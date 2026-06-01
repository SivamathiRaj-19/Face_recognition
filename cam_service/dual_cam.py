import cv2
import numpy as np
import time
import os
import csv
import glob

import sys, os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from queue import Queue
from tool.hailo_model import HAILO, rescale_network_outputs, SCRFDPostProc
from tool.camera_manager import get_camera_manager
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import math
import sys
from tool.database_interact import get_csv_users, get_server_users, create_user


os.environ['QT_QPA_PLATFORM'] = 'xcb'
try:
    active_user, inactive_user, active_user_attr, inactive_user_attr = get_server_users()
    csv_user = get_csv_users()
except:
    active_user, inactive_user, active_user_attr, inactive_user_attr = [], [], [], []
    csv_user = get_csv_users()

def rotate_image(image, angle, center = None):
    if center == None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    else:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

class CameraThread(QThread):
    frameReady = pyqtSignal(np.ndarray, int)
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_manager = get_camera_manager()
        self.user_id = f"dual_cam_cam_{camera_id}_{id(self)}"
        self.running = False
        
    def initialize_camera(self):
        try:
            config = {
                'format': 'XRGB8888',
                'size': (640, 480)
            }

            camera = self.camera_manager.get_camera(
                self.camera_id, 
                self.user_id, 
                config
            )
            
            if camera is None:
                return False
            
            time.sleep(0.5)
            return True
        except Exception as e:
            return False
    
    def run(self):
        if not self.initialize_camera():
            return
        
        self.setPriority(QThread.HighPriority)
        self.running = True
        while self.running:
            try:
                frame = self.camera_manager.capture_frame(self.camera_id, self.user_id)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.frameReady.emit(frame, self.camera_id)
                self.msleep(1)
            except Exception as e:
                break
        
        self.camera_manager.release_camera(self.camera_id, self.user_id)
    
    def stop(self):
        self.running = False
        try:
            self.camera_manager.release_camera(self.camera_id, self.user_id)
        except:
            pass

class FaceDetectionThread(QThread):
  
    faceDetected = pyqtSignal(list, np.ndarray, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_model = HAILO("/home/pi/KLTN_2025/tool/model/scrfd_500m.hef", task=0)
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.last_process_time = {0: 0, 1: 0}
        self.process_interval = 0.08 
        
                
        config = {
                    "nms_iou_thresh": 0.4,
                    "score_threshold": 0.6,
                    "anchors": {
                        "steps": [8, 16, 32],
                        "min_sizes": [
                            [16, 32],
                            [64, 128],
                            [256, 512]
                        ]
                    }
                }
                
        self.postprocessor = SCRFDPostProc(
            image_dims=(self.face_model.width, self.face_model.height),
            nms_iou_thresh=config["nms_iou_thresh"],
            score_threshold=config["score_threshold"],
            anchors=config["anchors"]
        )
        
    def add_frame(self, frame, camera_id):
      
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        self.frame_queue.put((frame.copy(), camera_id))
    
    def run(self):
        
        self.setPriority(QThread.HighPriority)
        self.running = True
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame, camera_id = self.frame_queue.get()
                    
                    current_time = time.time()
                    if current_time - self.last_process_time[camera_id] < self.process_interval:
                        continue
                    self.last_process_time[camera_id] = current_time
                    
                    faces = self.detect_faces(frame)
                    self.faceDetected.emit(faces, frame, camera_id)
                else:
                    self.msleep(5)  
            except Exception as e:
                pass
                
    def detect_faces(self, frame):
        try:                
            height, width = frame.shape[:2]
            # i = random.randint(1,20)
            # cv2.imwrite(f"/home/pi/KLTN_2025/tool/dd/{i}_org.jpg", frame)
            if self.face_model:
                frame_rz = cv2.resize(frame, (640,640))
                result = self.face_model(frame_rz)
                original_frame, infer_results = result
                
                # print(infer_results)
                
                rescaled_result = rescale_network_outputs(infer_results)

                
                results = self.postprocessor.main(rescaled_result)
                
                faces = []
                # print(results)
                if results['num_detections'] > 0:
                    boxes = results['detection_boxes'][0]
                    scores = results['detection_scores'][0]
                    landmarks = results['face_landmarks'][0]
                    
                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        if score > 0.6:
                            x1, y1, x2, y2 = box
                            x1, x2 = int(x1 * width), int(x2 * width)
                            y1, y2 = int(y1 * height), int(y2 * height)
                            
                            ## edit
                            landmarks_points = landmarks[i]
                            # x1,y1,x2,y2=0,0,0,0
                            for j in range(0, len(landmarks_points), 2):
                                x = int(landmarks_points[j] * width)
                                y = int(landmarks_points[j + 1] * height)
                                if j<3:
                                    x11 = int(landmarks_points[0] * width)
                                    y11 = int(landmarks_points[1] * height)
                                    x22 = int(landmarks_points[2] * width)
                                    y22 = int(landmarks_points[3] * height)
                                    if x11 != x22 or y11 != y22:
                                        try:
                                            angle = math.atan2(y22 - y11, x22 - x11)
                                            angle = angle*180/math.pi
                                            # cv2.circle(frame, (x1+int(abs(x1-x2)/2),y1+int(abs(y1-y2)/2)), 10, (0, 0, 255), -1)
                                        except:
                                            pass

                                else:
                                    # cv2.circle(image_with_boxes, (x, y), 10, (0, 255, 0), -1)
                                    pass
                            img = rotate_image(frame, int(angle), (x1+int(abs(x1-x2)/2),y1+int(abs(y1-y2)/2)))
                            img = img[y1:y2,x1:x2,:]
                            
                            
                            if x2 > x1 and y2 > y1:
                                # print([x1, y1, x2, y2])
                                faces.append([x1, y1, x2, y2, img])
                
                return faces if faces else []
            return []
        except Exception as e:
            print(e)
            return []
    
    def stop(self):
        self.running = False
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break

class Ui_MainWindow(object):
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("DualCameraWindow")
        MainWindow.resize(600, 1024)
        MainWindow.setWindowTitle("Dual Camera Face Recognition")
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.camera1_label = QtWidgets.QLabel(self.centralwidget)
        self.camera1_label.setGeometry(QtCore.QRect(0, 0, 600, 1024))
        self.camera1_label.setText("")
        self.camera1_label.setObjectName("camera1_label")
        
        self.camera2_label = QtWidgets.QLabel(self.centralwidget)
        self.camera2_label.setText("")
        self.camera2_label.setObjectName("camera2_label")
        
        button_width = 180
        button_height = 91
        spacing = 20
        total_width = button_width * 2 + spacing  
        start_x = int((600 - total_width) / 2)
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(start_x, 900, button_width, button_height))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")  
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(start_x + button_width + spacing, 900, button_width, button_height))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2") 
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.pushButton.clicked.connect(lambda: self.return_to_main(MainWindow))
        self.pushButton_2.clicked.connect(self.extract_features)
        self.pushButton_2.setEnabled(False) 
        
        self.camera_thread1 = None
        self.camera_thread2 = None
        self.face_detection_thread = None
        self.feature_thread1 = None
        self.feature_thread2 = None
        
        self.current_frame1 = None
        self.current_frame2 = None
        self.current_faces1 = []
        self.current_faces2 = []
        
        self.face_results1 = {}
        self.face_results2 = {}
        self.result_hold_time = 5 
        self.result_counters1 = {}
        self.result_counters2 = {}
        
        self.camera1 = None
        self.camera2 = None
        self.face_detection_model = None
        self.feature_model = HAILO("/home/pi/KLTN_2025/tool/model/arcface_mobilefacenet-2.hef", task=1, output_type={'arcface_mobilefacenet/fc1':'FLOAT32'})
        
        self.landmark_model = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        
        self.face_features = {}
        self.similarity_threshold = 0.7
        self.main_window = MainWindow
        MainWindow.closeEvent = self.closeEvent
        self.timer.start(16)
        
        self.is_capturing = False
        self.captured_images_cam1 = []
        self.captured_images_cam2 = []
        self.best_image_cam1 = None
        self.best_image_cam2 = None
        self.detected_face_cam1 = None
        self.detected_face_cam2 = None
        self.face_box_cam1 = None
        self.face_box_cam2 = None
        self.capture_start_time = 0
        self.capture_duration = 3
        self.max_frames_per_camera = 100
        self.face_name = ""
        self.user_id = ""
        self.capture_mode = False
        
        QtCore.QTimer.singleShot(300, self.show_input_dialog)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def style_message_box(self, msg_box):
        for button in msg_box.buttons():
            button.setMinimumHeight(60)
            button.setMinimumWidth(140)
            font = QtGui.QFont()
            font.setPointSize(18)
            button.setFont(font)
        
        font = QtGui.QFont()
        font.setPointSize(16)
        msg_box.setFont(font)
        
        return msg_box
    
    def style_dialog_buttons(self, button_box):
        for button in button_box.buttons():
            button.setMinimumHeight(60)
            button.setMinimumWidth(140)
            button_font = QtGui.QFont()
            button_font.setPointSize(18)
            button.setFont(button_font)
    
    def show_warning(self, title, message):
        msg = QMessageBox(self.main_window)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        self.style_message_box(msg)
        return msg.exec_()
    
    def show_critical(self, title, message):
        msg = QMessageBox(self.main_window)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        self.style_message_box(msg)
        return msg.exec_()
    
    def show_information(self, title, message):
        msg = QMessageBox(self.main_window)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        self.style_message_box(msg)
        return msg.exec_()
    
    def show_question(self, title, message, buttons=QMessageBox.Yes | QMessageBox.No):
        msg = QMessageBox(self.main_window)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(buttons)
        self.style_message_box(msg)
        return msg.exec_()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("DualCameraWindow", "Dual Camera Face Recognition"))
        self.pushButton.setText(_translate("DualCameraWindow", "Quay lại"))
        self.pushButton_2.setText(_translate("DualCameraWindow", "Xác nhận \nđăng ký"))

    def start_cameras(self):
        try:
            self.face_detection_thread = FaceDetectionThread(self.main_window)
            self.face_detection_thread.faceDetected.connect(self.on_face_detected)
            self.face_detection_thread.start()
            
            self.camera_thread1 = CameraThread(0, self.main_window)
            self.camera_thread1.frameReady.connect(self.on_frame_ready)
            self.camera_thread1.start()
            
            self.camera_thread2 = CameraThread(1, self.main_window)
            self.camera_thread2.frameReady.connect(self.on_frame_ready)
            self.camera_thread2.start()
            
            try:
                self.landmark_model = HAILO("/home/pi/KLTN_2025/tool/model/scrfd_500m.hef", task=0)
            except Exception as e:
                self.landmark_model = None
            
            self.load_face_features()
            
        except Exception as e:
            self.show_warning("Lỗi", f"Không thể khởi tạo system: {str(e)}")

    def on_frame_ready(self, frame, camera_id):
        try:
            height, width = frame.shape[:2]
            offset = 600
            shift = 120
            
            if camera_id == 0:
                frame1 = frame[:, ::-1]
                frame1_original = frame1.copy()
                left1 = max(int(width / 2 - offset + shift), 0)
                right1 = min(int(width / 2 + offset + shift), width)
                frame1 = frame1[:, left1:right1]
                self.current_frame1 = frame1.copy()
                
                if not hasattr(self, 'original_frame_cam1'):
                    self.original_frame_cam1 = None
                self.original_frame_cam1 = frame1_original
                
                if self.is_capturing and self.capture_mode:
                    if hasattr(self, 'faces_detected_cam1') and self.faces_detected_cam1:
                        if len(self.captured_images_cam1) < self.max_frames_per_camera:
                            self.captured_images_cam1.append(frame1.copy())
                            if (len(self.captured_images_cam1) >= self.max_frames_per_camera and 
                                len(self.captured_images_cam2) >= self.max_frames_per_camera):
                                QtCore.QTimer.singleShot(100, self.process_captured_images_dual)
                                
                if self.face_detection_thread and self.face_detection_thread.isRunning():
                    self.face_detection_thread.add_frame(frame1, camera_id)
                    
            elif camera_id == 1:
                frame2 = frame[:, ::-1]
                frame2_original = frame2.copy()
                left2 = max(int(width / 2 - offset - shift + 25), 0)
                right2 = min(int(width / 2 + offset - shift + 25), width)
                frame2 = frame2[:, left2:right2]
                self.current_frame2 = frame2.copy()
                
                if not hasattr(self, 'original_frame_cam2'):
                    self.original_frame_cam2 = None
                self.original_frame_cam2 = frame2_original
                
                if self.is_capturing and self.capture_mode:
                    if hasattr(self, 'faces_detected_cam2') and self.faces_detected_cam2:
                        if len(self.captured_images_cam2) < self.max_frames_per_camera:
                            self.captured_images_cam2.append(frame2.copy())
                            if (len(self.captured_images_cam1) >= self.max_frames_per_camera and 
                                len(self.captured_images_cam2) >= self.max_frames_per_camera):
                                QtCore.QTimer.singleShot(100, self.process_captured_images_dual)
                                
                if self.face_detection_thread and self.face_detection_thread.isRunning():
                    self.face_detection_thread.add_frame(frame2, camera_id)
                
        except Exception as e:
            pass

    def validate_single_face_in_frame(self, faces, frame_shape):

        if not faces or len(faces) == 0:
            return None
        
        frame_height, frame_width = frame_shape[:2]
        valid_faces = []
        
        for face in faces:
            if len(face) < 4:
                continue
                
            x1, y1, x2, y2 = map(float, face[:4])
            
            margin = 5
            if (x1 >= margin and y1 >= margin and 
                x2 <= frame_width - margin and y2 <= frame_height - margin):
                
                face_area = abs(x2 - x1) * abs(y2 - y1)
                valid_faces.append({
                    'coords': (x1, y1, x2, y2),
                    'area': face_area
                })
        
        if len(valid_faces) == 0:
            return None
        
        if len(valid_faces) == 1:
            return valid_faces[0]['coords']
        
        valid_faces.sort(key=lambda f: f['area'], reverse=True)
        
        largest = valid_faces[0]
        second_largest = valid_faces[1]
        
        if largest['area'] >= second_largest['area'] * 2:
            return largest['coords']
        
        return None
    
    def on_face_detected(self, faces, frame, camera_id):
        # try:
        if len(faces) != 0:
            if camera_id == 0:
                # print(faces)
                valid_face = self.validate_single_face_in_frame([faces[0][:-1]], frame.shape)
                
                if valid_face:
                    self.current_faces1 = [valid_face, faces[0][-1]]
                else:
                    self.current_faces1 = []
                
                if self.is_capturing and self.capture_mode:
                    if valid_face:
                        if not hasattr(self, 'faces_detected_cam1'):
                            self.faces_detected_cam1 = False
                        if not self.faces_detected_cam1:
                            self.faces_detected_cam1 = True
                        if hasattr(self, 'faces_detected_cam1') and self.faces_detected_cam1:
                            if len(self.captured_images_cam1) < self.max_frames_per_camera:
                                if not hasattr(self, 'captured_frames_with_faces_cam1'):
                                    self.captured_frames_with_faces_cam1 = []
                                self.captured_frames_with_faces_cam1.append({
                                    'frame': frame.copy(),
                                    'faces': [faces[0][-1]]
                                })
                                self.captured_images_cam1.append(frame.copy())
                            if (len(self.captured_images_cam1) >= self.max_frames_per_camera and 
                                len(self.captured_images_cam2) >= self.max_frames_per_camera):
                                QtCore.QTimer.singleShot(100, self.process_captured_images_dual)
                    else:
                        if not hasattr(self, 'warning_shown_cam1'):
                            self.warning_shown_cam1 = False
                        if not self.warning_shown_cam1:
                            self.warning_shown_cam1 = True
                            QtCore.QTimer.singleShot(2000, lambda: setattr(self, 'warning_shown_cam1', False))
                            
            elif camera_id == 1:
                valid_face = self.validate_single_face_in_frame(faces, frame.shape)
                
                if valid_face:
                    self.current_faces2 = [valid_face, faces[0][-1]]
                else:
                    self.current_faces2 = []
                
                if self.is_capturing and self.capture_mode:
                    if valid_face:
                        if not hasattr(self, 'faces_detected_cam2'):
                            self.faces_detected_cam2 = False
                        if not self.faces_detected_cam2:
                            self.faces_detected_cam2 = True
                        if hasattr(self, 'faces_detected_cam2') and self.faces_detected_cam2:
                            if len(self.captured_images_cam2) < self.max_frames_per_camera:
                                if not hasattr(self, 'captured_frames_with_faces_cam2'):
                                    self.captured_frames_with_faces_cam2 = []
                                self.captured_frames_with_faces_cam2.append({
                                    'frame': frame.copy(),
                                    'faces': [faces[0][-1]]
                                })
                                self.captured_images_cam2.append(frame.copy())
                            if (len(self.captured_images_cam1) >= self.max_frames_per_camera and 
                                len(self.captured_images_cam2) >= self.max_frames_per_camera):
                                QtCore.QTimer.singleShot(100, self.process_captured_images_dual)
                    else:
                        if not hasattr(self, 'warning_shown_cam2'):
                            self.warning_shown_cam2 = False
                        if not self.warning_shown_cam2:
                            self.warning_shown_cam2 = True
                            QtCore.QTimer.singleShot(2000, lambda: setattr(self, 'warning_shown_cam2', False))
                            

    def update_display(self):
        try:
            if (hasattr(self, 'camera_thread1') and self.camera_thread1 and 
                hasattr(self, 'camera_thread2') and self.camera_thread2):
                if self.current_frame1 is not None:
                    frame1 = self.current_frame1.copy()
                    self.draw_face_boxes(frame1, [self.current_faces1[0]], 
                                       getattr(self, 'face_results1', {}))

                    
                    self.display_frame(frame1, self.camera1_label)
                if self.current_frame2 is not None:
                    frame2 = self.current_frame2.copy()
                    self.draw_face_boxes(frame2, [self.current_faces2[0]],
                                       getattr(self, 'face_results2', {}))
                    self.display_frame(frame2, self.camera2_label)
            if hasattr(self, 'pushButton_2') and not self.is_capturing:
                if not self.pushButton_2.isEnabled():
                    has_captured_data = (hasattr(self, 'best_image_cam1') and hasattr(self, 'best_image_cam2') and 
                                        hasattr(self, 'detected_face_cam1') and hasattr(self, 'detected_face_cam2') and
                                        self.detected_face_cam1 is not None and self.detected_face_cam2 is not None and
                                        hasattr(self, 'user_id') and hasattr(self, 'face_name'))
                    if has_captured_data:
                        self.pushButton_2.setEnabled(True)
        except Exception as e:
            pass

    def draw_face_boxes(self, frame, faces, face_results):
        try:
            
            for face_coords in faces:
                x1, y1, x2, y2 = map(int, face_coords)
                # x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                
                frame_height, frame_width = frame.shape[:2]
                margin = 5
                is_fully_visible = (x1 >= margin and y1 >= margin and 
                                  x2 <= frame_width - margin and y2 <= frame_height - margin)
           
                x1_display = max(0, x1 )
                y1_display = max(0, y1)
                x2_display = min(frame.shape[1], x2 )
                y2_display = min(frame.shape[0], y2 )
                start_point = (x1, y1)
                end_point = (x2, y2)
                if self.is_capturing and self.capture_mode:
                    if is_fully_visible:
                        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 4)
                        cv2.putText(frame, "CAPTURING - VALID FACE", (x1_display, y1_display-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, start_point, end_point, (0, 165, 255), 3)
                        cv2.putText(frame, "FACE PARTIALLY OUT OF FRAME", (x1_display, y1_display-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        cv2.putText(frame, "Move closer to center", (frame.shape[1]//2 - 150, frame.shape[0] - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    cv2.rectangle(frame, start_point, end_point, (0, 255, 255), 2)
        except Exception as e:
            print(e)
            pass

    def display_frame(self, frame, label):

        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            label.setPixmap(pixmap.scaled(label.width(), label.height(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio))
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        except Exception as e:
            pass

    def load_face_features(self):
      
        csv_paths = ["./face_features.csv", "face_features.csv"]
        csv_path = next((path for path in csv_paths if os.path.exists(path)), None)
        
        if not csv_path:
            return
        
        try:
            with open(csv_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                
                id_idx = header.index('id') if 'id' in header else 0
                name_idx = header.index('name') if 'name' in header else 1
                features_idx = header.index('features') if 'features' in header else 2
                
                for row in reader:
                    if len(row) <= features_idx:
                        continue
                        
                    face_id = row[id_idx]
                    face_name = row[name_idx]
                    features_str = row[features_idx]
                    features = np.array([float(x) for x in features_str.split(',')])
                    
                    self.face_features[face_id] = {
                        'name': face_name,
                        'features': features
                    }
        except Exception as e:
            pass

    def find_best_match(self, current_features):

        if not self.face_features:
            return None
        
        current_norm = np.linalg.norm(current_features)
        current_normalized = current_features / current_norm if current_norm > 0 else current_features
        
        best_match = None
        best_similarity = 0
        
        for face_id, data in self.face_features.items():
            stored_features = data['features']
            min_length = min(len(current_features), len(stored_features))
            stored_vec = stored_features[:min_length]
            
            stored_norm = np.linalg.norm(stored_vec)
            stored_normalized = stored_vec / stored_norm if stored_norm > 0 else stored_vec
                
            l2_dist = np.linalg.norm(current_normalized[:min_length] - stored_normalized)
            similarity = 1 - (l2_dist / 2)
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'id': face_id,
                    'name': data['name'],
                    'similarity': similarity
                }
        
        return best_match

    def detect_faces(self, frame):
        try:
            if self.face_detection_thread and self.face_detection_thread.face_model:
                return self.face_detection_thread.detect_faces(frame)
            return []
        except Exception as e:
            return []

    def return_to_main(self, main_window):
        self.timer.stop()
        
        reply = self.show_question(
            'Xác nhận',
            'Bạn muốn quay lại màn hình Menu chứ?',
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Yes:  
            self.return_to_menu_from_main()
        elif reply == QMessageBox.No:  
            from tool.login import Ui_MainWindow as LoginWindow
            self.window = QtWidgets.QMainWindow()
            self.ui = LoginWindow()
            self.ui.setupUi(self.window)
            self.window.show()
            main_window.close()
    
    def return_to_menu_from_main(self):
        try:
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog("Đang chuyển màn hình...", None, 0, 0, self.main_window)
            progress.setWindowTitle("Vui lòng đợi")
            progress.setCancelButton(None)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()
            
            self.cleanup_resources_async(self.main_window)
            
        except Exception as e:
            self.show_warning("Lỗi", f"Không thể quay lại: {str(e)}")
    
    def extract_features(self):
        # try:
        if not hasattr(self, 'user_id') or not hasattr(self, 'face_name') or not self.user_id or not self.face_name:
            self.show_warning("Lỗi", "Chưa có thông tin người dùng!")
            return
        
        if not hasattr(self, 'best_image_cam1') or not hasattr(self, 'best_image_cam2'):
            self.show_warning("Lỗi", "Chưa có ảnh được capture!")
            return
        
        if not hasattr(self, 'detected_face_cam1') or not hasattr(self, 'detected_face_cam2'):
            self.show_warning("Lỗi", "Chưa phát hiện khuôn mặt trong ảnh đã capture!")
            return
        
        self.extract_and_save_features()
            
        # except Exception as e:
        #     self.show_critical("Lỗi", f"Lỗi khi trích xuất đặc trưng: {str(e)}")

    def cleanup_resources_async(self, MainWindow):
        class CleanupThread(QThread):
            def __init__(self, ui_instance):
                super().__init__()
                self.ui = ui_instance
                
            def run(self):
                try:
                    if hasattr(self.ui, 'timer'):
                        self.ui.timer.stop()
                    
                    if hasattr(self.ui, 'camera_thread1') and self.ui.camera_thread1:
                        self.ui.camera_thread1.stop()
                        self.ui.camera_thread1.wait(100)
                        self.ui.camera_thread1 = None
                        
                    if hasattr(self.ui, 'camera_thread2') and self.ui.camera_thread2:
                        self.ui.camera_thread2.stop()
                        self.ui.camera_thread2.wait(100)
                        self.ui.camera_thread2 = None
                    
                    if hasattr(self.ui, 'face_detection_thread') and self.ui.face_detection_thread:
                        self.ui.face_detection_thread.stop()
                        self.ui.face_detection_thread.wait(100)
                        self.ui.face_detection_thread = None
                        
                    if hasattr(self.ui, 'feature_thread1') and self.ui.feature_thread1:
                        self.ui.feature_thread1.stop()
                        self.ui.feature_thread1.wait(100)
                        self.ui.feature_thread1 = None
                        
                    if hasattr(self.ui, 'feature_thread2') and self.ui.feature_thread2:
                        self.ui.feature_thread2.stop()
                        self.ui.feature_thread2.wait(100)
                        self.ui.feature_thread2 = None
                    
                    if hasattr(self.ui, 'camera1') and self.ui.camera1:
                        try:
                            self.ui.camera1.stop()
                            self.ui.camera1.close()
                        except:
                            pass
                        self.ui.camera1 = None
                        
                    if hasattr(self.ui, 'camera2') and self.ui.camera2:
                        try:
                            self.ui.camera2.stop()
                            self.ui.camera2.close()
                        except:
                            pass
                        self.ui.camera2 = None
                    
                except Exception as e:
                    pass
        
        self.cleanup_thread = CleanupThread(self)
        self.cleanup_thread.finished.connect(lambda: self.switch_to_menu(MainWindow))
        self.cleanup_thread.start()

    def switch_to_menu(self, MainWindow):
        try:
            from tool.Mainmenu import Ui_MainWindow as MainMenu_UI
            self.window = QtWidgets.QMainWindow()
            self.ui = MainMenu_UI()
            self.ui.setupUi(self.window)
            self.window.show()
            
            if MainWindow and hasattr(MainWindow, 'close'):
                MainWindow.close()
                
        except Exception as e:
            pass

    def return_to_menu(self, MainWindow):
        try:
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog("Đang chuyển màn hình...", None, 0, 0, self.main_window)
            progress.setWindowTitle("Vui lòng đợi")
            progress.setCancelButton(None)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()
            
            self.cleanup_resources_async(MainWindow)
            
        except Exception as e:
            pass
    
    def closeEvent(self, event):
        import gc
        
        self.timer.stop()
        if hasattr(self, 'camera_thread1') and self.camera_thread1:
            self.camera_thread1.stop()
            self.camera_thread1 = None
            
        if hasattr(self, 'camera_thread2') and self.camera_thread2:
            self.camera_thread2.stop()
            self.camera_thread2 = None
        
        if hasattr(self, 'face_detection_thread') and self.face_detection_thread:
            self.face_detection_thread.stop()
            self.face_detection_thread = None
            
        if hasattr(self, 'feature_thread1') and self.feature_thread1:
            self.feature_thread1.stop()
            self.feature_thread1 = None
            
        if hasattr(self, 'feature_thread2') and self.feature_thread2:
            self.feature_thread2.stop()
            self.feature_thread2 = None
        if hasattr(self, 'camera1') and self.camera1:
            try:
                self.camera1.stop()
                self.camera1.close()
            except:
                pass
            self.camera1 = None
            
        if hasattr(self, 'camera2') and self.camera2:
            try:
                self.camera2.stop()
                self.camera2.close()
            except:
                pass
            self.camera2 = None
        
        gc.collect()
        
        event.accept()

    def get_csv_path(self):
        csv_path = "/home/pi/KLTN_2025/face_features.csv"
        if not os.path.exists(csv_path):
            csv_path = "/home/pi/KLTN_2025/tool/face_features.csv"
        return csv_path

    def is_id_exists(self, csv_path, id_to_check):
        if not os.path.exists(csv_path):
            return False
        
        try:
            with open(csv_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader, None)
                
                id_col = 0
                if header and 'id' in header:
                    id_col = header.index('id')
                
                for row in reader:
                    if len(row) > id_col:
                        id_str = row[id_col].strip()
                        if '_' in id_str:
                            base_id = id_str.split('_')[0]
                        else:
                            base_id = id_str
                        
                        if base_id == str(id_to_check):
                            return True
        except Exception as e:
            pass
        
        return False

    def get_next_id(self, csv_path):
        next_id = 1  
        
        if not os.path.exists(csv_path):
            return next_id
            
        try:
            all_ids = []
            with open(csv_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader, None)
                
                id_col = 0
                if header and 'id' in header:
                    id_col = header.index('id')
                    
                for row in reader:
                    if len(row) > id_col:
                        id_str = row[id_col].strip()
                        if '_' in id_str:
                            base_id = id_str.split('_')[0]
                        else:
                            base_id = id_str
                        
                        if base_id.isdigit():
                            all_ids.append(int(base_id))
            
            if all_ids:
                unique_ids = sorted(list(set(all_ids)))
                
                for i in range(1, unique_ids[-1] + 1):
                    if i not in unique_ids:
                        return i
                
                next_id = unique_ids[-1] + 1
            
        except Exception as e:
            pass
        return next_id

    def get_existing_ids(self, csv_path):
        existing_ids = []
        
        if not os.path.exists(csv_path):
            return existing_ids
            
        try:
            with open(csv_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader, None)
                
                id_col = 0
                if header and 'id' in header:
                    id_col = header.index('id')
                    
                for row in reader:
                    if len(row) > id_col:
                        id_str = row[id_col].strip()
                        if '_' in id_str:
                            base_id = id_str.split('_')[0]
                        else:
                            base_id = id_str
                        
                        if base_id.isdigit():
                            existing_ids.append(int(base_id))
            
            existing_ids = sorted(list(set(existing_ids)))
        except Exception as e:
            pass
        
        return existing_ids

    def show_input_dialog_for_button(self):
        from tool.learning import AlphaNeumericVirtualKeyboard
        
        class InputDialog(QDialog):
            def __init__(self, suggested_id, existing_ids, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Nhập thông tin người dùng")
                self.setMinimumWidth(600)  
                self.setMinimumHeight(700)
                
                self.main_layout = QtWidgets.QVBoxLayout(self)
                
                if existing_ids:
                    info_label = QtWidgets.QLabel(f"IDs đã tồn tại: {', '.join(map(str, existing_ids))}")
                    info_label.setStyleSheet("color: blue; font-size: 14px; margin: 10px;")
                    self.main_layout.addWidget(info_label)
                
                form_layout = QFormLayout()
                self.id_edit = QLineEdit(str(suggested_id))
                self.name_edit = QLineEdit()
                
                font = QtGui.QFont()
                font.setPointSize(16)
                self.id_edit.setFont(font)
                self.name_edit.setFont(font)
                self.id_edit.setMinimumHeight(50)
                self.name_edit.setMinimumHeight(50)
                
                self.id_edit.setToolTip(f"ID đề xuất: {suggested_id}")
                
                form_layout.addRow("ID người dùng:", self.id_edit)
                form_layout.addRow("Tên người dùng:", self.name_edit)
                
                self.main_layout.addLayout(form_layout)
                
                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                
                for button in buttons.buttons():
                    button.setMinimumHeight(60)
                    button.setMinimumWidth(140)
                    button_font = QtGui.QFont()
                    button_font.setPointSize(18)
                    button.setFont(button_font)
                
                spacer = QtWidgets.QSpacerItem(20, 300, QtWidgets.QSizePolicy.Minimum, 
                                              QtWidgets.QSizePolicy.Expanding)
                self.main_layout.addItem(spacer)
                
                self.main_layout.addWidget(buttons)
                
                self.screen_width = 600
                
                self.keyboard = AlphaNeumericVirtualKeyboard(None, 0, 400, self)
                self.keyboard.setFixedWidth(self.screen_width)
                
                self.keyboard_timer = QtCore.QTimer()
                self.keyboard_timer.setSingleShot(True)
                self.keyboard_timer.timeout.connect(self.adjust_keyboard)
                
                self.keyboard.hide()
                
                self.original_id_focus = self.id_edit.mousePressEvent
                self.original_name_focus = self.name_edit.mousePressEvent
                
                self.id_edit.mousePressEvent = self.create_id_focus_handler()
                self.name_edit.mousePressEvent = self.create_name_focus_handler()
                
                self.mousePressEvent = self.hide_keyboard_on_click
            
            def adjust_keyboard(self):
                if not self.keyboard.isHidden:
                    self.keyboard.setGeometry(0, 400, self.screen_width, 315)
                    self.keyboard.x_pos = 0
                    self.keyboard.y_pos = 400
                    self.keyboard.raise_()  
            
            def create_id_focus_handler(self):
                def handler(event):
                    if self.original_id_focus:
                        self.original_id_focus(event)
                    
                    mouse_event = QtGui.QMouseEvent(
                        QtCore.QEvent.Type.MouseButtonPress,
                        QtCore.QPoint(0, 0),
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.KeyboardModifier.NoModifier
                    )
                    
                    self.keyboard.display(self.id_edit, mouse_event, closeButtonEnable=True)
                    self.keyboard_timer.start(300)  
                    event.accept()
                return handler
            
            def create_name_focus_handler(self):
                def handler(event):
                    if self.original_name_focus:
                        self.original_name_focus(event)
                    
                    mouse_event = QtGui.QMouseEvent(
                        QtCore.QEvent.Type.MouseButtonPress,
                        QtCore.QPoint(0, 0),
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.KeyboardModifier.NoModifier
                    )
                    
                    self.keyboard.display(self.name_edit, mouse_event, closeButtonEnable=True)
                    self.keyboard_timer.start(300)
                    event.accept()
                return handler
            
            def hide_keyboard_on_click(self, event):
                if not self.keyboard.isHidden:
                    if not (self.id_edit.geometry().contains(event.pos()) or 
                            self.name_edit.geometry().contains(event.pos()) or
                            self.keyboard.geometry().contains(event.pos())):
                        self.keyboard.close_handler()
                
                QtWidgets.QDialog.mousePressEvent(self, event)
            
            def get_data(self):
                return self.id_edit.text().strip(), self.name_edit.text().strip()
        
        csv_path = self.get_csv_path()
        suggested_id = self.get_next_id(csv_path)
        existing_ids = self.get_existing_ids(csv_path)
        
        while True:
            dialog = InputDialog(suggested_id, existing_ids, self.main_window)
            if dialog.exec_() == QDialog.Accepted:
                user_id, face_name = dialog.get_data()
                if not user_id or not face_name:
                    self.show_warning("Thiếu thông tin", "Vui lòng nhập đầy đủ ID và tên!")
                    continue
                if self.is_id_exists(csv_path, user_id):
                    current_existing = self.get_existing_ids(csv_path)
                    reply = self.show_question(
                        "ID đã tồn tại",
                        f"ID {user_id} đã tồn tại trong hệ thống!\n\n"
                        f"IDs hiện có: {', '.join(map(str, current_existing))}\n"
                        f"ID đề xuất tiếp theo: {max(current_existing) + 1 if current_existing else 1}\n\n"
                        f"Bạn có muốn sử dụng ID khác không?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if reply == QtWidgets.QMessageBox.Yes:
                        suggested_id = max(current_existing) + 1 if current_existing else 1
                        continue
                    else:
                        return None, None
                return user_id.strip(), face_name.strip()
            else:
                return None, None

    def show_input_dialog(self):
        from tool.learning import AlphaNeumericVirtualKeyboard
        
        class InputDialog(QDialog):
            def __init__(self, suggested_id, existing_ids, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Nhập thông tin người dùng")
                self.setMinimumWidth(600)  
                self.setMinimumHeight(700)
                
                self.main_layout = QtWidgets.QVBoxLayout(self)
                
                if existing_ids:
                    info_label = QtWidgets.QLabel(f"IDs đã tồn tại: {', '.join(map(str, existing_ids))}")
                    info_label.setStyleSheet("color: blue; font-size: 14px; margin: 10px;")
                    self.main_layout.addWidget(info_label)
                
                form_layout = QFormLayout()
                self.id_edit = QLineEdit(str(suggested_id))
                self.name_edit = QLineEdit()
                
                font = QtGui.QFont()
                font.setPointSize(16)
                self.id_edit.setFont(font)
                self.name_edit.setFont(font)
                self.id_edit.setMinimumHeight(50)
                self.name_edit.setMinimumHeight(50)
                
                self.id_edit.setToolTip(f"ID đề xuất: {suggested_id}")
                
                form_layout.addRow("ID người dùng:", self.id_edit)
                form_layout.addRow("Tên người dùng:", self.name_edit)
                
                self.main_layout.addLayout(form_layout)
                
                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                
                for button in buttons.buttons():
                    button.setMinimumHeight(60)
                    button.setMinimumWidth(140)
                    button_font = QtGui.QFont()
                    button_font.setPointSize(18)
                    button.setFont(button_font)
                
                spacer = QtWidgets.QSpacerItem(20, 300, QtWidgets.QSizePolicy.Minimum, 
                                              QtWidgets.QSizePolicy.Expanding)
                self.main_layout.addItem(spacer)
                
                self.main_layout.addWidget(buttons)
                
                self.screen_width = 600
                
                self.keyboard = AlphaNeumericVirtualKeyboard(None, 0, 400, self)
                
                self.keyboard.setFixedWidth(self.screen_width)
                
                self.keyboard_timer = QtCore.QTimer()
                self.keyboard_timer.setSingleShot(True)
                self.keyboard_timer.timeout.connect(self.adjust_keyboard)
                
                self.keyboard.hide()
                
                self.original_id_focus = self.id_edit.mousePressEvent
                self.original_name_focus = self.name_edit.mousePressEvent
                
                self.id_edit.mousePressEvent = self.create_id_focus_handler()
                self.name_edit.mousePressEvent = self.create_name_focus_handler()
                
                self.mousePressEvent = self.hide_keyboard_on_click
            
            def adjust_keyboard(self):
                if not self.keyboard.isHidden:
                    self.keyboard.setGeometry(0, 400, self.screen_width, 315)
                    self.keyboard.x_pos = 0
                    self.keyboard.y_pos = 400
                    self.keyboard.raise_()  
            
            def create_id_focus_handler(self):
                def handler(event):
                    if self.original_id_focus:
                        self.original_id_focus(event)
                    
                    mouse_event = QtGui.QMouseEvent(
                        QtCore.QEvent.Type.MouseButtonPress,
                        QtCore.QPoint(0, 0),
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.KeyboardModifier.NoModifier
                    )
                    
                    self.keyboard.display(self.id_edit, mouse_event, closeButtonEnable=True)
                    
                    self.keyboard_timer.start(300)  
                    
                    event.accept()
                return handler
            
            def create_name_focus_handler(self):
                def handler(event):
                    if self.original_name_focus:
                        self.original_name_focus(event)
                    
                    mouse_event = QtGui.QMouseEvent(
                        QtCore.QEvent.Type.MouseButtonPress,
                        QtCore.QPoint(0, 0),
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.KeyboardModifier.NoModifier
                    )
                    
                    self.keyboard.display(self.name_edit, mouse_event, closeButtonEnable=True)
                    
                    self.keyboard_timer.start(300)
                    
                    event.accept()
                return handler
            
            def hide_keyboard_on_click(self, event):
                if not self.keyboard.isHidden:
                    if not (self.id_edit.geometry().contains(event.pos()) or 
                            self.name_edit.geometry().contains(event.pos()) or
                            self.keyboard.geometry().contains(event.pos())):
                        self.keyboard.close_handler()
                
                QtWidgets.QDialog.mousePressEvent(self, event)
            
            def get_data(self):
                return self.id_edit.text().strip(), self.name_edit.text().strip()
        
        csv_path = self.get_csv_path()
        suggested_id = self.get_next_id(csv_path)
        
        existing_ids = self.get_existing_ids(csv_path)
        
        while True:
            dialog = InputDialog(suggested_id, existing_ids, self.main_window)
            if dialog.exec_() == QDialog.Accepted:
                user_id, face_name = dialog.get_data()
                if not user_id or not face_name:
                    self.show_warning("Thiếu thông tin", "Vui lòng nhập đầy đủ ID và tên!")
                    continue
                try:
                    list_1 = [i.replace("_Camera2","").replace("_Camera1","") for i in active_user.keys()]
                    list_2 = [i.replace("_Camera2","").replace("_Camera1","") for i in inactive_user.keys()]
                except:
                    list_1, list_2 = [], []
                list_3 = [i.replace("_Camera2","").replace("_Camera1","") for i in csv_user['name'].tolist()]
                if (face_name in list_1 or face_name in list_2 or face_name in list_3):
                    self.show_warning("Thông tin không hợp lệ", "Tên người dùng đã tồn tại, vui lòng đổi tên người dùng!")
                    continue
                if self.is_id_exists(csv_path, user_id):
                    current_existing = self.get_existing_ids(csv_path)
                    reply = self.show_question(
                        "ID đã tồn tại",
                        f"ID {user_id} đã tồn tại trong hệ thống!\n\n"
                        f"IDs hiện có: {', '.join(map(str, current_existing))}\n"
                        f"ID đề xuất tiếp theo: {max(current_existing) + 1 if current_existing else 1}\n\n"
                        f"Bạn có muốn sử dụng ID khác không?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if reply == QtWidgets.QMessageBox.Yes:
                        suggested_id = max(current_existing) + 1 if current_existing else 1
                        continue
                    else:
                        self.return_to_menu(self.main_window)
                        return
                break
            else:
                self.return_to_menu(self.main_window)
                return
    
        self.face_name = face_name.strip()
        self.user_id = user_id.strip()
        
        self.start_cameras()
        self.start_capture_with_face_detection()

    def start_capture_with_face_detection(self):
        self.is_capturing = False
        self.capture_mode = False
        self.capture_start_time = time.time()
        self.captured_images_cam1 = []
        self.captured_images_cam2 = []
        self.captured_frames_with_faces_cam1 = []
        self.captured_frames_with_faces_cam2 = []
        self.faces_detected_cam1 = False
        self.faces_detected_cam2 = False
        
        if hasattr(self, 'pushButton_2'):
            self.pushButton_2.setEnabled(False)
        
        self.show_information("Thông báo", 
            f"Vui lòng đặt khuôn mặt vào khung hình của cả 2 camera.\n\nSau khi nhấn OK, hệ thống sẽ bắt đầu thu thập {self.max_frames_per_camera} khung hình từ mỗi camera.")
        
        self.is_capturing = True
        self.capture_mode = True

    def start_capture_process(self):
        self.is_capturing = True
        self.capture_mode = True
        self.capture_start_time = time.time()
        self.captured_images_cam1 = []
        self.captured_images_cam2 = []
        
        if hasattr(self, 'pushButton_2'):
            self.pushButton_2.setEnabled(False)
        
        self.show_information("Thông báo", 
            f"Vui lòng đặt khuôn mặt vào khung hình của cả 2 camera!\nSẽ thu thập {self.max_frames_per_camera} frame từ mỗi camera.")
        
    def process_captured_images_dual(self):
        self.is_capturing = False
        self.capture_mode = False
        
        if self.captured_images_cam1:
            # try:
            best_frame_cam1, best_face_cam1, best_face_box_cam1 = self.find_best_face_frame(self.captured_images_cam1, "Camera 1")
            
            if best_frame_cam1 is not None and best_face_cam1 is not None:
                self.best_image_cam1 = best_frame_cam1
                self.detected_face_cam1 = best_face_cam1
                self.face_box_cam1 = best_face_box_cam1
                
                display_img1 = self.best_image_cam1.copy()
                cv2.putText(display_img1, f"Best Face - Camera 1 ({len(self.captured_images_cam1)} frames)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.display_frame(display_img1, self.camera1_label)
            else:
                self.show_warning("Lỗi Camera 1", "Không tìm thấy ảnh nào có khuôn mặt từ Camera 1!")
        
        if self.captured_images_cam2:
            best_frame_cam2, best_face_cam2, best_face_box_cam2 = self.find_best_face_frame(self.captured_images_cam2, "Camera 2")
            
            if best_frame_cam2 is not None and best_face_cam2 is not None:
                self.best_image_cam2 = best_frame_cam2
                self.detected_face_cam2 = best_face_cam2
                self.face_box_cam2 = best_face_box_cam2
                
                display_img2 = self.best_image_cam2.copy()
                cv2.putText(display_img2, f"Best Face - Camera 2 ({len(self.captured_images_cam2)} frames)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.display_frame(display_img2, self.camera2_label)
            else:
                self.show_warning("Lỗi Camera 2", "Không tìm thấy ảnh nào có khuôn mặt từ Camera 2!")
        
        if (hasattr(self, 'detected_face_cam1') and self.detected_face_cam1 is not None and
            hasattr(self, 'detected_face_cam2') and self.detected_face_cam2 is not None):
            if hasattr(self, 'pushButton_2'):
                self.pushButton_2.setEnabled(True)
        else:
            if hasattr(self, 'pushButton_2'):
                self.pushButton_2.setEnabled(False)
        
        self.stop_cameras()

    def extract_face_features_for_storage(self, face_img):
        if self.feature_model:
            face_img = face_img[0]
            tensor_output = self.feature_model(face_img)
            
            return tensor_output
        else:
            raise Exception("Model không khả dụng")

    def find_best_feature_with_cosine_filtering(self, feature_vectors, similarity_threshold=0.85):
        if not feature_vectors or len(feature_vectors) == 0:
            return None, None, None
        
        features_array = np.array(feature_vectors)
        norms = np.linalg.norm(features_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        features_normalized = features_array / norms
        
        similarity_matrix = np.dot(features_normalized, features_normalized.T)
        
        avg_similarities = similarity_matrix.mean(axis=1)
        
        valid_mask = avg_similarities >= similarity_threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            best_index = np.argmax(avg_similarities)
            return features_array[best_index], best_index, {
                'method': 'fallback',
                'total_vectors': len(features_array),
                'valid_vectors': 0
            }
        
        valid_vectors = features_array[valid_indices]
        average_vector = np.mean(valid_vectors, axis=0)
        average_vector = average_vector / (np.linalg.norm(average_vector) + 1e-8)
        valid_vectors_norm = valid_vectors / (np.linalg.norm(valid_vectors, axis=1, keepdims=True) + 1e-8)
        similarities_to_avg = np.dot(valid_vectors_norm, average_vector)
        best_in_valid_idx = np.argmax(similarities_to_avg)
        best_global_idx = valid_indices[best_in_valid_idx]
        best_feature = features_array[best_global_idx]
        
        cluster_info = {
            'method': 'cosine_filtering',
            'threshold': similarity_threshold,
            'total_vectors': len(features_array),
            'valid_vectors': len(valid_indices),
            'outliers_removed': len(features_array) - len(valid_indices),
            'best_similarity_to_avg': float(similarities_to_avg[best_in_valid_idx]),
            'avg_similarity': float(avg_similarities[best_global_idx])
        }
        return best_feature, best_global_idx, cluster_info

    def extract_and_save_features(self):
        csv_path = self.get_csv_path()
        if not os.path.exists("faces"):
            os.makedirs("faces")
        
        if hasattr(self, 'user_id') and self.user_id:
            old_images_pattern = f"faces/{self.user_id}_*.jpg"
            old_images = glob.glob(old_images_pattern)
            
            if old_images:
                for old_img in old_images:
                    try:
                        os.remove(old_img)
                    except Exception as e:
                        pass
            
            feature_vectors_cam1 = []
            frame_data_list_cam1 = []
            
            if hasattr(self, 'captured_frames_with_faces_cam1') and self.captured_frames_with_faces_cam1:
                
                for frame_data in self.captured_frames_with_faces_cam1:
                    frame = frame_data['frame']
                    faces = frame_data['faces']
                    
                    if not faces:
                        continue
                    
                    img, features = self.extract_face_features_for_storage(faces)
                    if features is not None:
                        feature_vectors_cam1.append(features)
                        frame_data_list_cam1.append({
                            'frame': frame,
                            'face_crop': faces,
                        })
            
            best_feature_cam1 = None
            best_frame_data_cam1 = None
            
            if len(feature_vectors_cam1) >= 3:
                best_feature_cam1, best_index, cluster_info = self.find_best_feature_with_cosine_filtering(
                    feature_vectors_cam1, 
                    similarity_threshold=0.85
                )
                if best_index is not None and best_index < len(frame_data_list_cam1):
                    best_frame_data_cam1 = frame_data_list_cam1[best_index]
            elif len(feature_vectors_cam1) > 0:
                best_feature_cam1 = feature_vectors_cam1[0]
                best_frame_data_cam1 = frame_data_list_cam1[0] if frame_data_list_cam1 else None
            
            if best_feature_cam1 is not None:
                if best_frame_data_cam1 is not None:
                    if hasattr(self, 'original_frame_cam1') and self.original_frame_cam1 is not None:
                        best_face_path_cam1 = f"faces/{self.user_id}_{self.face_name}_cam1_best.jpg"
                        cv2.imwrite(best_face_path_cam1, self.original_frame_cam1)
                    else:
                        best_face_path_cam1 = f"faces/{self.user_id}_{self.face_name}_cam1_best.jpg"
                        cv2.imwrite(best_face_path_cam1, best_frame_data_cam1['frame'])
                    
                    face_crop_path_cam1 = f"faces/{self.user_id}_{self.face_name}_cam1_crop.jpg"
                    cv2.imwrite(face_crop_path_cam1, best_frame_data_cam1['face_crop'][0])
                
                features_str_cam1 = ','.join(map(str, best_feature_cam1.tolist()))
                is_new_file = not os.path.exists(csv_path)
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if is_new_file:
                        header = ['id', 'name', 'features']
                        writer.writerow(header)
                    row_cam1 = [f"{self.user_id}", f"{self.face_name}_Camera1", features_str_cam1]
                    writer.writerow(row_cam1)
                    features = best_feature_cam1.tolist()
                    try:
                        print("Upload state:", True if create_user(f"{self.face_name}_Camera1",features) else False)
                    except:
                        print("Can not connect to server")
        if hasattr(self, 'best_image_cam2') and self.best_image_cam2 is not None and hasattr(self, 'face_box_cam2') and self.face_box_cam2 is not None:
            
            feature_vectors_cam2 = []
            frame_data_list_cam2 = []
            
            if hasattr(self, 'captured_frames_with_faces_cam2') and self.captured_frames_with_faces_cam2:
                
                for frame_data in self.captured_frames_with_faces_cam2:
                    try:
                        frame = frame_data['frame']
                        faces = frame_data['faces']
                        
                        if not faces:
                            continue
                        
                        img, features = self.extract_face_features_for_storage(faces)
                        if features is not None:
                            feature_vectors_cam2.append(features)
                            frame_data_list_cam2.append({
                                'frame': frame,
                                'face_crop': faces,
                            })
                            
                    except Exception as e:
                        continue
            
            best_feature_cam2 = None
            best_frame_data_cam2 = None
            
            if len(feature_vectors_cam2) >= 3:
                best_feature_cam2, best_index, cluster_info = self.find_best_feature_with_cosine_filtering(
                    feature_vectors_cam2, 
                    similarity_threshold=0.85
                )
                if best_index is not None and best_index < len(frame_data_list_cam2):
                    best_frame_data_cam2 = frame_data_list_cam2[best_index]
            elif len(feature_vectors_cam2) > 0:
                best_feature_cam2 = feature_vectors_cam2[0]
                best_frame_data_cam2 = frame_data_list_cam2[0] if frame_data_list_cam2 else None
            
            if best_feature_cam2 is not None:
                if best_frame_data_cam2 is not None:
                    if hasattr(self, 'original_frame_cam2') and self.original_frame_cam2 is not None:
                        best_face_path_cam2 = f"faces/{self.user_id}_{self.face_name}_cam2_best.jpg"
                        cv2.imwrite(best_face_path_cam2, self.original_frame_cam2)
                    else:
                        best_face_path_cam2 = f"faces/{self.user_id}_{self.face_name}_cam2_best.jpg"
                        cv2.imwrite(best_face_path_cam2, best_frame_data_cam2['frame'])
                    
                    face_crop_path_cam2 = f"faces/{self.user_id}_{self.face_name}_cam2_crop.jpg"
                    cv2.imwrite(face_crop_path_cam2, best_frame_data_cam2['face_crop'][0])
                
                features_str_cam2 = ','.join(map(str, best_feature_cam2.tolist()))
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    row_cam2 = [f"{int(self.user_id)+1}", f"{self.face_name}_Camera2", features_str_cam2]
                    features = best_feature_cam2.tolist()
                    try:
                        print("Upload state:", True if create_user(f"{self.face_name}_Camera2",features) else False)
                    except:
                        print("Can not connect to server")
                    writer.writerow(row_cam2)
        
        cam1_success = hasattr(self, 'best_image_cam1') and self.best_image_cam1 is not None and hasattr(self, 'face_box_cam1') and self.face_box_cam1 is not None
        cam2_success = hasattr(self, 'best_image_cam2') and self.best_image_cam2 is not None and hasattr(self, 'face_box_cam2') and self.face_box_cam2 is not None
        if cam1_success and cam2_success:
            message = f"Đã đăng ký thành công khuôn mặt của {self.face_name} từ cả 2 camera!\nID: {self.user_id}\n"
        elif cam1_success:
            message = f"Đã đăng ký thành công khuôn mặt của {self.face_name} từ Camera 1!\nID: {self.user_id}\n"
        elif cam2_success:
            message = f"Đã đăng ký thành công khuôn mặt của {self.face_name} từ Camera 2!\nID: {self.user_id}\n"
        else:
            message = "Không thể đăng ký do không có khuôn mặt nào được phát hiện!"
        self.show_information("Thành công", message)
        QtCore.QTimer.singleShot(2000, lambda: self.return_to_main(self.main_window))
        # except Exception as e:
        #     self.show_warning("Lỗi", f"Lỗi khi trích xuất đặc trưng: {str(e)}")
            
    def stop_cameras(self):
        try:
            if hasattr(self, 'timer'):
                self.timer.stop()
            
            if hasattr(self, 'camera_thread1') and self.camera_thread1:
                self.camera_thread1.stop()
                self.camera_thread1 = None
                
            if hasattr(self, 'camera_thread2') and self.camera_thread2:
                self.camera_thread2.stop()
                self.camera_thread2 = None
            
            if hasattr(self, 'face_detection_thread') and self.face_detection_thread:
                self.face_detection_thread.stop()
                self.face_detection_thread = None
                
            if hasattr(self, 'feature_thread1') and self.feature_thread1:
                self.feature_thread1.stop()
                self.feature_thread1 = None
                
            if hasattr(self, 'feature_thread2') and self.feature_thread2:
                self.feature_thread2.stop()
                self.feature_thread2 = None
        except Exception as e:
            pass
    
    def find_best_face_frame(self, captured_images, camera_name):
        if not captured_images:
            return None, None, None
        
        best_frame = None
        best_face = None
        best_face_box = None
        best_score = 0
        
        frames_with_faces = None
        if camera_name == "Camera 1" and hasattr(self, 'captured_frames_with_faces_cam1'):
            frames_with_faces = self.captured_frames_with_faces_cam1
        elif camera_name == "Camera 2" and hasattr(self, 'captured_frames_with_faces_cam2'):
            frames_with_faces = self.captured_frames_with_faces_cam2
        
        if frames_with_faces and len(frames_with_faces) > 0:
            for i, frame_data in enumerate(frames_with_faces):
                try:
                    frame = frame_data['frame']
                    faces = frame_data['faces']
                    if not faces:
                        continue

                    best_frame = frame.copy()
                    best_face = faces
                    best_face_box = (0, 0, 0, 0)
                except Exception as e:
                    continue
            if best_frame is not None:
                return best_frame, best_face, best_face_box

def main():
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
import sys
import os
import cv2
import numpy as np
import csv
import time
import gc
import math
from queue import Queue
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QApplication

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from tool.hailo_model import HAILO, rescale_network_outputs, SCRFDPostProc
from tool.camera_manager import get_camera_manager
from tool.database_interact import compare_and_update, get_server_users, save_checkin, get_user
from tool.anti_spoofing import AntiSpoofing

os.environ['QT_QPA_PLATFORM'] = 'xcb'
active_user, inactive_user, active_user_attr, inactive_user_attr = get_server_users()

class DatabaseSyncThread(QThread):
    trigger = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.time_old = time.time()
        self.reload_time = 1 * 60  # 1 min
        self.first_sync_done = False
    
    def run(self):
        global active_user, inactive_user, active_user_attr, inactive_user_attr
        self.running = True
        self.trigger.emit(1)
        while self.running:
            if time.time() - self.time_old > self.reload_time:
                self.time_old = time.time()
                try:
                    compare_and_update()
                    self.trigger.emit(1)
                    active_user, inactive_user, active_user_attr, inactive_user_attr = get_server_users()
                except Exception as e:
                    print(e)
            self.msleep(33)
    
    def stop(self):
        self.running = False

class DisplayThread(QThread):
    frameReady = pyqtSignal(np.ndarray, int)
    
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_manager = get_camera_manager()
        self.user_id = f"display_cam_{camera_id}_{id(self)}"
        self.running = False
        
    def initialize_camera(self):
        try:
            config = {
                'format': 'XRGB8888',
                'size': (1920, 1080)
            }
            camera = self.camera_manager.get_camera(self.camera_id, self.user_id, config)
            if camera is None:
                return False
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"[DisplayThread] Lỗi khởi tạo camera {self.camera_id}: {e}")
            return False
    
    def run(self):
        if not self.initialize_camera():
            return
        
        self.setPriority(QThread.HighestPriority)
        self.running = True
        
        while self.running:
            try:
                frame = self.camera_manager.capture_frame(self.camera_id, self.user_id)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.frameReady.emit(frame.copy(), self.camera_id)
                self.msleep(33)
            except Exception as e:
                print(f"[DisplayThread] Lỗi camera {self.camera_id}: {e}")
                break
        
        self.camera_manager.release_camera(self.camera_id, self.user_id)
    
    def stop(self):
        self.running = False
        try:
            self.camera_manager.release_camera(self.camera_id, self.user_id)
        except:
            pass

class CameraThread(QThread):
    frameReady = pyqtSignal(np.ndarray, int, float)
    
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_manager = get_camera_manager()
        self.user_id = f"process_cam_{camera_id}_{id(self)}"
        self.running = False
        
    def initialize_camera(self):
        try:
            config = {
                'format': 'XRGB8888',
                'size': (1920, 1080)
            }
            camera = self.camera_manager.get_camera(self.camera_id, self.user_id, config)
            if camera is None:
                return False
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"[ProcessThread] Lỗi khởi tạo camera {self.camera_id}: {e}")
            return False
    
    def run(self):
        if not self.initialize_camera():
            return
        
        self.setPriority(QThread.NormalPriority)
        self.running = True
        
        while self.running:
            try:
                frame = self.camera_manager.capture_frame(self.camera_id, self.user_id)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    timestamp = time.time()
                    self.frameReady.emit(frame.copy(), self.camera_id, timestamp)
                    
                self.msleep(50)
            except Exception as e:
                print(f"[ProcessThread] Lỗi camera {self.camera_id}: {e}")
                break
        
        self.camera_manager.release_camera(self.camera_id, self.user_id)
    
    def stop(self):
        self.running = False
        try:
            self.camera_manager.release_camera(self.camera_id, self.user_id)
        except:
            pass

class FaceDetectionThread(QThread):
    faceDetected = pyqtSignal(list, np.ndarray, int, object)  # faces, frame, camera_id, landmarks
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_model = HAILO("/home/pi/KLTN_2025/tool/model/scrfd_500m.hef", task=0)
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        
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
        self.running = True
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame, camera_id = self.frame_queue.get()
                    faces, landmarks_normalized = self.detect_faces(frame)
                    self.faceDetected.emit(faces, frame, camera_id, landmarks_normalized)
                else:
                    self.msleep(10)
            except Exception as e:
                pass
                
    def detect_faces(self, frame):
        try:                
            height, width = frame.shape[:2]
            if self.face_model:
                frame_rz = cv2.resize(frame,(640,640))
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
                            
                            landmarks_points = landmarks[i]
                            for j in range(0, len(landmarks_points), 2):
                                if j < 3:
                                    x11 = int(landmarks_points[0] * width)
                                    y11 = int(landmarks_points[1] * height)
                                    x22 = int(landmarks_points[2] * width)
                                    y22 = int(landmarks_points[3] * height)
                                    if x11 != x22 or y11 != y22:
                                        try:
                                            angle = math.atan2(y22 - y11, x22 - x11)
                                            angle = angle * 180 / math.pi
                                        except:
                                            pass
                            
                            img = rotate_image(frame, int(angle), (x1 + int(abs(x1 - x2) / 2), y1 + int(abs(y1 - y2) / 2)))
                            img = img[y1:y2, x1:x2, :]
                            
                            if x2 > x1 and y2 > y1:
                                faces.append([x1, y1, x2, y2, img, landmarks_points])
                
                first_landmarks = faces[0][5] if faces else None
                return (faces if faces else [], first_landmarks)
            return ([], None)
        except Exception as e:
            print(e)
            return ([], None)
    
    def stop(self):
        self.running = False
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break


class AntiSpoofingThread(QThread):
    livenessResult = pyqtSignal(str, bool, float, str)  
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.anti_spoof = AntiSpoofing(history_size=60, min_frames=30)
        self.task_queue = Queue(maxsize=10)
        self.running = False
        
        self.face_liveness_state = {}  
        self.real_duration_threshold = 1
    
    def add_task(self, face_id, landmarks):
        if self.task_queue.full():
            try:
                self.task_queue.get_nowait()
            except:
                pass
        self.task_queue.put((face_id, landmarks))
    
    def run(self):
        self.running = True
        while self.running:
            try:
                if not self.task_queue.empty():
                    face_id, landmarks = self.task_queue.get()
                    is_real, score, details = self.anti_spoof.check(landmarks, face_id)
                    
                    current_time = time.time()
                    if face_id not in self.face_liveness_state:
                        self.face_liveness_state[face_id] = {
                            'is_real': False,
                            'real_start_time': None,
                            'confirmed_real': False,
                            'last_update': current_time,
                            'max_score': 0.0,  
                            'start_time': current_time,  
                            'collecting_done': False  
                        }
                    
                    state = self.face_liveness_state[face_id]
                    state['last_update'] = current_time
                    
                    
                    if score > state['max_score']:
                        state['max_score'] = score
                    
                    
                    if 'Collecting' not in details:
                        state['collecting_done'] = True
                    
                    if is_real:
                        if state['real_start_time'] is None:
                            state['real_start_time'] = current_time
                        real_duration = current_time - state['real_start_time']
                        if real_duration >= self.real_duration_threshold:
                            state['confirmed_real'] = True
                            state['is_real'] = True
                    else:
                        
                        state['is_real'] = False
                    
                    
                    if not state['collecting_done']:
                        
                        display_score = score
                        display_details = details
                    else:
                        
                        display_score = state['max_score']
                        if state['confirmed_real']:
                            display_details = "✓ Xác thực thành công"
                        elif is_real:
                            
                            elapsed = current_time - state['real_start_time'] if state['real_start_time'] else 0
                            progress = min(elapsed / self.real_duration_threshold * 100, 99)
                            display_details = f"Đang xác thực... {progress:.0f}%"
                        else:
                            display_details = "Vui lòng cười để xác thực"
                    
                    self.livenessResult.emit(face_id, state['confirmed_real'], display_score, display_details)
                else:
                    self.msleep(10)
            except Exception as e:
                print(f"Error: {e}")
                pass
    
    def is_face_confirmed_real(self, face_id):
        if face_id in self.face_liveness_state:
            return self.face_liveness_state[face_id].get('confirmed_real', False)
        return False
    
    def reset_face(self, face_id):
        if face_id in self.face_liveness_state:
            del self.face_liveness_state[face_id]
        self.anti_spoof.reset(face_id)
    
    def cleanup_old_faces(self, timeout=10.0):
        current_time = time.time()
        to_remove = []
        for face_id, state in self.face_liveness_state.items():
            if current_time - state['last_update'] > timeout:
                to_remove.append(face_id)
        
        for face_id in to_remove:
            self.reset_face(face_id)
    
    def reset_state(self):
        self.face_liveness_state.clear()
        self.anti_spoof.face_histories.clear()
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                break
    
    def stop(self):
        self.running = False
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                break


class FeatureExtractionThread(QThread):
    featuresExtracted = pyqtSignal(np.ndarray, int, tuple, np.ndarray)   
    
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.feature_model = HAILO("/home/pi/KLTN_2025/tool/model/arcface_mobilefacenet-2.hef", task=1, output_type={'arcface_mobilefacenet/fc1':'FLOAT32'})
            
        self.task_queue = Queue(maxsize=20)
        self.running = False
        
        self.features_extracted_count = 0
        self.features_start_time = time.time()
        
        self.kept_count = 0
        self.enhanced_count = 0
        self.rejected_count = 0
    
    def add_task(self, straight_face, face_coords):
        self.task_queue.put((straight_face.copy(), face_coords))
    
    def run(self):
        self.running = True
        while self.running:
            if not self.task_queue.empty():
                straight_face, face_coords = self.task_queue.get()
                features = self.extract_features(straight_face)
                
                if features is not None:
                    self.features_extracted_count += 1
                    current_time = time.time()
                    elapsed = current_time - self.features_start_time
                    
                    if elapsed >= 1.0:
                        features_per_sec = self.features_extracted_count / elapsed
                        print(f"[CAM {self.camera_id}] Features extracted: {self.features_extracted_count} images in {elapsed:.2f}s = {features_per_sec:.2f} images/sec")
                        self.features_extracted_count = 0
                        self.features_start_time = current_time
                    feature_vec = features[-1].flatten() if hasattr(features[-1], 'flatten') else np.array(features[-1]).flatten()
                    self.featuresExtracted.emit(feature_vec, self.camera_id, face_coords, straight_face)
            else:
                self.msleep(1)
    
    def calculate_blur_score(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def calculate_brightness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def enhance_image_light(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def is_good_quality(self, img, blur_thresh=200, bright_range=(40, 210)):
        blur = self.calculate_blur_score(img)
        bright = self.calculate_brightness(img)
        is_good = blur > blur_thresh and bright_range[0] < bright < bright_range[1]
        return is_good, blur, bright
    
    def should_enhance(self, blur, bright, blur_good=200, bright_good=(40, 210),
                       blur_min=100, bright_min=(30, 220)):
        if blur < blur_min or bright < bright_min[0] or bright > bright_min[1]:
            return 'reject'
        elif blur < blur_good or bright < bright_good[0] or bright > bright_good[1]:
            return 'enhance'
        else:
            return 'keep'
    
    def extract_features(self, face_img):
        face_img = cv2.medianBlur(face_img, 5)
        return self.feature_model(face_img)
    
    def stop(self):
        self.running = False
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                break

class RecognitionThread(QThread):
    recognitionComplete = pyqtSignal(str, dict, int, list, float)
    
    OUTLIER_THRESHOLD = 0.85
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_queue = Queue(maxsize=10)
        self.running = False
        self.face_features = {}
        self.similarity_threshold = 0.6
        self.check_in_local = []
        
    def set_face_features(self, face_features):
        self.face_features = face_features.copy()
    
    def set_params(self, similarity_threshold):
        self.similarity_threshold = similarity_threshold
    
    def add_task(self, face_id, vectors, camera_id, coords, straight_face):
        if self.task_queue.full():
            try:
                self.task_queue.get_nowait()
            except:
                pass
        self.task_queue.put((face_id, vectors, camera_id, coords, straight_face))
    
    def run(self):
        self.running = True
        while self.running:
            if not self.task_queue.empty():
                face_id, vectors, camera_id, coords, straight_face = self.task_queue.get()
                
                filtered_vector = self.filter_and_average_vectors(vectors)
                result, best_similarity = self.compare_with_database(filtered_vector)
                
                coords_list = list(coords) if isinstance(coords, tuple) else coords
                
                if result is None:
                    result = {}
                else:
                    print(result)
                    print("Vector shape:", filtered_vector.shape)
                    self.check_in_local.append({
                        "user_id": active_user[result['name']]['id'],
                        "name": result['name'],
                        "camera_vector": filtered_vector.tolist(),
                        "user_vector": get_user(active_user[result['name']]['id'])['features'],
                        "accuracy_rate": result['similarity'],
                        "processing_speed": 120
                    })
                
                if len(self.check_in_local) > 0:
                    try:
                        print(save_checkin(
                            user_id=self.check_in_local[0]['user_id'],
                            name=self.check_in_local[0]['name'],
                            camera_vector=self.check_in_local[0]['camera_vector'],
                            user_vector=self.check_in_local[0]['user_vector'],
                            accuracy_rate=self.check_in_local[0]['accuracy_rate'],
                            processing_speed=self.check_in_local[0]['processing_speed'],
                        ))
                        self.check_in_local.pop(0)
                    except:
                        print("Can not connect to server")
                
                self.recognitionComplete.emit(face_id, result, camera_id, [coords_list, straight_face], best_similarity)
            else:
                self.msleep(5)
    
    def filter_and_average_vectors(self, vectors_list):
        
        try:
            n = len(vectors_list)
            if n < 1: return np.zeros(512)
            if n == 1: return vectors_list[0]
            if n < 4: return np.mean(vectors_list, axis=0)
            
            vecs = np.array(vectors_list)
            mean_vec = np.mean(vecs, axis=0)
            mean_norm = np.linalg.norm(mean_vec)
            if mean_norm < 1e-6: return mean_vec
            
            norms = np.linalg.norm(vecs, axis=1)
            sims = np.dot(vecs, mean_vec) / (norms * mean_norm + 1e-10)
            
            good_mask = sims >= self.OUTLIER_THRESHOLD
            good_vecs = vecs[good_mask]
            
            if len(good_vecs) < 2: return mean_vec
            if len(good_vecs) < n:
                print(f"[Filter] Loại {n-len(good_vecs)}/{n} outliers")
            return np.mean(good_vecs, axis=0)
        except:
            return np.mean(vectors_list, axis=0)
    
    def compare_with_database(self, avg_vec):
        
        try:
            avg_vec = np.array(avg_vec).flatten()
            
            avg_norm = np.linalg.norm(avg_vec)
            if avg_norm < 1e-6: return None, 0.0
            
            ids, vecs, names = [], [], []
            for fid, data in self.face_features.items():
                feat = np.array(data['features']).flatten()
                vec_norm = np.linalg.norm(feat)
                if vec_norm > 1e-6:
                    ids.append(fid)
                    vecs.append(feat)
                    names.append(data['name'])
            
            if not vecs: return None, 0.0
            
            mat = np.array(vecs)
            sims = np.dot(mat, avg_vec) / (np.linalg.norm(mat, axis=1) * avg_norm + 1e-10)
            best_idx = np.argmax(sims)
            best_sim = float(sims[best_idx])
            print(f"Best: {names[best_idx]} = {best_sim:.4f}")
            if best_sim >= self.similarity_threshold:
                return {'id': ids[best_idx], 'name': names[best_idx], 'similarity': best_sim, 
                       'validated': True, 'confirmed_dual': True}, best_sim
            return None, best_sim
        except:
            return None, 0.0
    
    def stop(self):
        self.running = False
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                break

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("DualRecognitionWindow")
        MainWindow.resize(600, 1024) 
        MainWindow.setWindowTitle("Dual Camera Face Recognition")
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.camera1_label = QtWidgets.QLabel(self.centralwidget)
        self.camera1_label.setGeometry(QtCore.QRect(0, 0, 600, 824))
        self.camera1_label.setText("Camera 1")
        self.camera1_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.camera1_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.camera1_label.setObjectName("camera1_label")
        
        self.camera2_label = QtWidgets.QLabel(self.centralwidget)
        self.camera2_label.setText("Camera 2")
        self.camera2_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.camera2_label.setObjectName("camera2_label")
        
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(0, 824, 500, 200))
        self.listView.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.listView.setObjectName("listView")
        
        self.face_image_label = QtWidgets.QLabel(self.centralwidget)
        self.face_image_label.setGeometry(QtCore.QRect(10, 834, 120, 161))
        self.face_image_label.setText("")
        self.face_image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.face_image_label.setStyleSheet("""
            background-color:
            border: 2px solid
        """)
        self.face_image_label.setObjectName("face_image_label")
        
        self.info_text = QtWidgets.QTextEdit(self.centralwidget)
        self.info_text.setGeometry(QtCore.QRect(140, 824, 281, 181))
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("""
            background-color:
            color:
            font-size: 16px; 
            font-weight: bold;
            border: 2px solid
            padding: 10px;
        """)
        self.info_text.setObjectName("info_text")
        
        self.Menu = QtWidgets.QPushButton(self.centralwidget)
        self.Menu.setGeometry(QtCore.QRect(500, 824, 100, 200))
        self.Menu.setStyleSheet("#btn_circle {\n"
        "    border: 2px solid #0078d7;\n"
        "    border-radius: 25px;\n"
        "    background-color: #00aaff;\n"
        "    color: white;\n"
        "}\n"
        "#Menu:hover {\n"
        "    background-color: #0088cc;\n"
        "}\n"
        "")
        self.Menu.setObjectName("Menu")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.Menu.setIcon(QtGui.QIcon("/home/pi/KLTN_2025/tool/icon/menu.jpg"))
        self.Menu.setIconSize(QtCore.QSize(90, 90))
        self.Menu.clicked.connect(lambda: self.return_to_login(MainWindow))
        
        self.display_thread1 = None
        self.display_thread2 = None
        
        self.camera_thread1 = None
        self.camera_thread2 = None
        self.face_detection_thread1 = None
        self.face_detection_thread2 = None 
        self.feature_thread1 = None
        self.feature_thread2 = None
        self.recognition_thread = None
        self.anti_spoofing_thread = None  
        
        self.current_frame1 = None
        self.current_frame2 = None
        self.current_faces1 = []
        self.current_faces2 = []
        
        self.latest_recognition1 = {} 
        self.latest_recognition2 = {}  
        
        self.face_liveness_confirmed = {} 
        self.face_liveness_status = {}  
        
        
        self.pending_recognition = {}
        
        self.confirmed_faces = {}
        self.confirmed_faces_timeout = 10.0
        
        self.authenticated_faces = set()
        
        self.feature_buffers = {}
        self.similarity_threshold = 0.6
        self.straight_face = None
        
        self.face_presence1 = {}
        self.face_presence2 = {}
        self.face_absence_threshold = 3.0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        
        self.face_features = {}
        
        self.current_user_name = None
        self.current_user_id = None
        
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        
        self.main_window = MainWindow
        MainWindow.closeEvent = self.closeEvent
        self.start_cameras()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("DualRecognitionWindow", "Dual Camera Face Recognition"))
        self.Menu.setText(_translate("DualRecognitionWindow", ""))

    def start_cameras(self):
        self.face_detection_thread1 = FaceDetectionThread(self.main_window)
        self.face_detection_thread1.faceDetected.connect(self.on_face_detected)
        self.face_detection_thread1.start()
        
        self.face_detection_thread2 = FaceDetectionThread(self.main_window)
        self.face_detection_thread2.faceDetected.connect(self.on_face_detected)
        self.face_detection_thread2.start()
        
        self.anti_spoofing_thread = AntiSpoofingThread(self.main_window)
        self.anti_spoofing_thread.livenessResult.connect(self.on_liveness_result)
        self.anti_spoofing_thread.start()
        
        self.feature_thread1 = FeatureExtractionThread(0, self.main_window)
        self.feature_thread1.featuresExtracted.connect(self.on_features_extracted)
        self.feature_thread1.start()
        
        self.feature_thread2 = FeatureExtractionThread(1, self.main_window)
        self.feature_thread2.featuresExtracted.connect(self.on_features_extracted)
        self.feature_thread2.start()
        
        self.recognition_thread = RecognitionThread(self.main_window)
        self.recognition_thread.recognitionComplete.connect(self.on_recognition_complete)
        self.recognition_thread.start()
        
        self.camera_thread1 = CameraThread(0, self.main_window)
        self.camera_thread1.frameReady.connect(self.on_frame_ready)
        self.camera_thread1.start()
        
        self.camera_thread2 = CameraThread(1, self.main_window)
        self.camera_thread2.frameReady.connect(self.on_frame_ready)
        self.camera_thread2.start()

        self.display_thread1 = DisplayThread(0, self.main_window)
        self.display_thread1.frameReady.connect(self.on_display_frame)
        self.display_thread1.start()
        
        self.display_thread2 = DisplayThread(1, self.main_window)
        self.display_thread2.frameReady.connect(self.on_display_frame)
        self.display_thread2.start()
        
        self.sync_database = DatabaseSyncThread(self.main_window)
        self.sync_database.trigger.connect(self.load_face_features)
        self.sync_database.start()
        
        self.timer.start(16)

    def on_display_frame(self, frame, camera_id):
        height, width = frame.shape[:2]
        offset = 600
        shift = 120
        
        if camera_id == 0:
            frame = frame[:, ::-1]
            left = max(int(width / 2 - offset + shift), 0)
            right = min(int(width / 2 + offset + shift), width)
            frame = frame[:, left:right]
            self.current_frame1 = frame
        elif camera_id == 1:
            frame = frame[:, ::-1]
            left = max(int(width / 2 - offset - shift + 25), 0)
            right = min(int(width / 2 + offset - shift + 25), width)
            frame = frame[:, left:right]
            self.current_frame2 = frame


    def on_frame_ready(self, frame, camera_id, timestamp=None):
       
        if timestamp is None:
            timestamp = time.time()
            
        height, width = frame.shape[:2]
        offset = 600
        shift = 120
        detect_interval = 2
        if camera_id == 0:
            frame1 = frame[:, ::-1]  
            left1 = max(int(width / 2 - offset + shift), 0)
            right1 = min(int(width / 2 + offset + shift ), width)
            frame1 = frame1[:, left1:right1]
            if self.face_detection_thread1 and self.face_detection_thread1.isRunning():
                self.face_detection_thread1.add_frame(frame1, camera_id)

        elif camera_id == 1:
            frame2 = frame[:, ::-1]
            left2 = max(int(width / 2 - offset - shift+25), 0)
            right2 = min(int(width / 2 + offset - shift+25), width)
            frame2 = frame2[:, left2:right2]
            if self.face_detection_thread2 and self.face_detection_thread2.isRunning():
                self.face_detection_thread2.add_frame(frame2, camera_id)

    def on_face_detected(self, faces, frame, camera_id, landmarks_normalized=None):
        try:
            face_boxes = [[f[0], f[1], f[2], f[3]] for f in faces] if faces else []
            tracked_faces = self.update_face_tracking(face_boxes, camera_id)
            
            if camera_id == 0:
                self.current_faces1 = [coords for _, coords in tracked_faces]
                if not self.current_faces1:
                    self.immediate_cleanup_camera_results(camera_id)
            elif camera_id == 1:
                self.current_faces2 = [coords for _, coords in tracked_faces]
                if not self.current_faces2:
                    self.immediate_cleanup_camera_results(camera_id)
            
            if landmarks_normalized is not None and self.anti_spoofing_thread and self.anti_spoofing_thread.isRunning():
                for face_id, _ in tracked_faces:
                    self.anti_spoofing_thread.add_task(face_id, landmarks_normalized)
            
            max_concurrent_faces = 1
            
            unique_face_ids = set()
            for buffer_key in self.feature_buffers.keys():
                if '_cam' in buffer_key:
                    base_face_id = buffer_key.rsplit('_cam', 1)[0]
                    unique_face_ids.add(base_face_id)
            active_buffers = len(unique_face_ids)
            
            unconfirmed_faces = [(fid, coords) for fid, coords in tracked_faces 
                                if fid not in self.confirmed_faces and fid not in self.authenticated_faces]
            
            if len(unconfirmed_faces) > 1:
                largest_unconfirmed = self.filter_largest_face(unconfirmed_faces)
                confirmed_faces = [(fid, coords) for fid, coords in tracked_faces if fid in self.confirmed_faces]
                tracked_faces = confirmed_faces + largest_unconfirmed
            
            for i, (face_id, face_coords) in enumerate(tracked_faces):
                if face_id in self.authenticated_faces:
                    continue
                    
                if face_id in self.confirmed_faces:
                    confirmed = self.confirmed_faces[face_id]
                    current_time = time.time()
                    
                    if camera_id == 0:
                        self.face_presence1[face_id] = current_time
                    else:
                        self.face_presence2[face_id] = current_time
                    
                    result = {
                        'id': confirmed['id'],
                        'name': confirmed['name'],
                        'similarity': confirmed.get('final_similarity', 1.0),
                        'validated': True,
                        'cached': True,
                        'confirmed_dual': confirmed.get('confirmed_dual', False)
                    }
                    
                    if camera_id == 0:
                        self.latest_recognition1[face_id] = {
                            'result': result,
                            'coords': face_coords,
                            'timestamp': current_time
                        }
                    else:
                        self.latest_recognition2[face_id] = {
                            'result': result,
                            'coords': face_coords,
                            'timestamp': current_time
                        }
                    continue
                
                buffer_key_this_cam = f"{face_id}_cam{camera_id}"
                face_needs_collection = buffer_key_this_cam in self.feature_buffers
                
                if '_face_' in face_id:
                    face_number = face_id.split('_face_')[-1]
                    same_face_in_buffer = any(
                        f"_face_{face_number}_cam" in key 
                        for key in self.feature_buffers.keys()
                    )
                else:
                    same_face_in_buffer = False
                
                if face_needs_collection:
                    should_extract = True
                elif same_face_in_buffer:
                    should_extract = True
                else:
                    if active_buffers < max_concurrent_faces:
                        should_extract = True
                    else:
                        should_extract = False
                
                if should_extract:
                    straight_face = faces[0][4] if len(faces) > 0 and len(faces[0]) > 4 else None
                    
                    if straight_face is not None and straight_face.size > 0:
                        if camera_id == 0 and self.feature_thread1 and self.feature_thread1.isRunning():
                            self.feature_thread1.add_task(straight_face, (face_id, face_coords))
                        elif camera_id == 1 and self.feature_thread2 and self.feature_thread2.isRunning():
                            self.feature_thread2.add_task(straight_face, (face_id, face_coords))

        except:
            pass
        
    def evaluate_average_features(self, face_id, avg_vec):
        result, best_sim = self.recognition_thread.compare_with_database(avg_vec)
        if result:
            t = time.time()
            coords = [0, 0, 100, 100]
            self.latest_recognition1[face_id] = {'result': result.copy(), 'coords': coords, 'timestamp': t}
            self.latest_recognition2[face_id] = {'result': result.copy(), 'coords': coords, 'timestamp': t}
            self.confirmed_faces[face_id] = {'id': result['id'], 'name': result['name'], 
                                            'final_similarity': best_sim, 'confirmed_dual': True, 'timestamp': t}
            return True
        return False
    
    def on_features_extracted(self, features, camera_id, face_data, face_img):
        if isinstance(face_data, tuple) and len(face_data) == 2:
            face_id, face_coords = face_data
        else:
            face_id = f"face_{hash(str(face_data))}"
            face_coords = face_data
        
        current_time = time.time()
        if camera_id == 0:
            self.face_presence1[face_id] = current_time
        else:
            self.face_presence2[face_id] = current_time
        
        if face_id in self.confirmed_faces:
            return
        
        buffer_key = f"{face_id}_cam{camera_id}"
        if buffer_key not in self.feature_buffers:
            self.feature_buffers[buffer_key] = {'vectors': [], 'start_time': current_time, 'face_id': face_id}
        
        buffer = self.feature_buffers[buffer_key]
        buffer['vectors'].append(features.copy())
        
        cam0_vecs = self.feature_buffers.get(f"{face_id}_cam0", {}).get('vectors', [])
        cam1_vecs = self.feature_buffers.get(f"{face_id}_cam1", {}).get('vectors', [])
        total = len(cam0_vecs) + len(cam1_vecs)
        elapsed = current_time - buffer['start_time']
        
        if elapsed >= 3.0 or total >= 21:
            if face_id in self.authenticated_faces:
                self.feature_buffers.pop(f"{face_id}_cam0", None)
                self.feature_buffers.pop(f"{face_id}_cam1", None)
                return
            
            all_vecs = cam0_vecs + cam1_vecs
            print(all_vecs[0].shape)
            if len(all_vecs) >= 3 and self.recognition_thread and self.recognition_thread.isRunning():
                self.recognition_thread.add_task(face_id, all_vecs, camera_id, face_coords, face_img)
            
            self.feature_buffers.pop(f"{face_id}_cam0", None)
            self.feature_buffers.pop(f"{face_id}_cam1", None)
    
    def on_liveness_result(self, face_id, is_confirmed_real, score, details):
        self.face_liveness_confirmed[face_id] = is_confirmed_real
        self.face_liveness_status[face_id] = {
            'is_real': is_confirmed_real,
            'score': score,
            'details': details
        }
        
        
        if is_confirmed_real and face_id in self.pending_recognition:
            pending = self.pending_recognition.pop(face_id)
            result = pending['result']
            coords = pending['coords']
            straight_face = pending['straight_face']
            camera_id = pending['camera_id']
            similarity = pending['similarity']
            current_time = time.time()
            
            print(f"[SUCCESS - FROM PENDING] {result['name']} (sim={similarity:.3f})")
            
            
            result['validated'] = True
            result['confirmed_dual'] = True
            result['liveness_failed'] = False
            
            self.latest_recognition1[face_id] = {
                'result': result.copy(),
                'coords': coords,
                'timestamp': current_time
            }
            self.latest_recognition2[face_id] = {
                'result': result.copy(),
                'coords': coords,
                'timestamp': current_time
            }
            
            confirmed_data = {
                'id': result['id'],
                'name': result['name'],
                'final_similarity': similarity,
                'confirmed_dual': True,
                'timestamp': current_time
            }
            self.confirmed_faces[face_id] = confirmed_data
            self.authenticated_faces.add(face_id)
            self.straight_face = straight_face
            
            if camera_id == 0 and self.current_frame1 is not None:
                self.save_authenticated_face(straight_face, result['name'], result['id'], 0)
            elif camera_id == 1 and self.current_frame2 is not None:
                self.save_authenticated_face(straight_face, result['name'], result['id'], 1)
    
    def is_face_liveness_confirmed(self, face_id):
        return self.face_liveness_confirmed.get(face_id, False)
    
    def on_recognition_complete(self, face_id, result, camera_id, coords_and_face, similarity):
        current_time = time.time()
        coords, straight_face = coords_and_face
        is_liveness_ok = self.is_face_liveness_confirmed(face_id)
        liveness_status = self.face_liveness_status.get(face_id, {})
        
        if result and isinstance(result, dict) and 'id' in result:
            if is_liveness_ok:
                print(f"[Sucess] {result['name']} (sim={similarity:.3f}") 
                self.latest_recognition1[face_id] = {
                    'result': result.copy(),
                    'coords': coords,
                    'timestamp': current_time
                }
                self.latest_recognition2[face_id] = {
                    'result': result.copy(),
                    'coords': coords,
                    'timestamp': current_time
                }
                
                confirmed_data = {
                    'id': result['id'],
                    'name': result['name'],
                    'final_similarity': similarity,
                    'confirmed_dual': True,
                    'timestamp': current_time
                }
                self.confirmed_faces[face_id] = confirmed_data
                
                self.authenticated_faces.add(face_id)
                
                if camera_id == 0 and self.current_frame1 is not None:
                    self.save_authenticated_face(straight_face, result['name'], result['id'], 0)
                elif camera_id == 1 and self.current_frame2 is not None:
                    self.save_authenticated_face(straight_face, result['name'], result['id'], 1)
            else:
                liveness_score = liveness_status.get('score', 0.0)
                liveness_details = liveness_status.get('details', 'Unknown')
                print(f"[WAITING LIVENESS] {result['name']} (sim={similarity:.3f}) - score={liveness_score:.2f}")
                
                
                self.pending_recognition[face_id] = {
                    'result': result.copy(),
                    'coords': coords,
                    'straight_face': straight_face,
                    'camera_id': camera_id,
                    'similarity': similarity,
                    'timestamp': current_time
                }
                
                spoofing_result = {
                    'id': result['id'],
                    'name': f"[Đang chờ...] {result['name']}",
                    'similarity': similarity,
                    'validated': False,
                    'cached': False,
                    'liveness_failed': True
                }
                
                if camera_id == 0:
                    self.latest_recognition1[face_id] = {
                        'result': spoofing_result,
                        'coords': coords,
                        'timestamp': current_time
                    }
                else:
                    self.latest_recognition2[face_id] = {
                        'result': spoofing_result,
                        'coords': coords,
                        'timestamp': current_time
                    }
        else:
            unknown_result = {
                'id': None,
                'name': 'Unknown',
                'similarity': 0.0,
                'validated': False,
                'cached': False
            }
            
            if camera_id == 0:
                self.latest_recognition1[face_id] = {
                    'result': unknown_result,
                    'coords': coords,
                    'timestamp': current_time
                }
            else:
                self.latest_recognition2[face_id] = {
                    'result': unknown_result,
                    'coords': coords,
                    'timestamp': current_time
                }
    
    def cleanup_old_recognition_results(self):
        current_time = time.time()
        cleanup_threshold = 8.0
        confirmed_cleanup_threshold = 15.0
        
        keys_to_remove = []
        for face_id, data in self.latest_recognition1.items():
            result = data.get('result', {})
            is_confirmed = result.get('validated', False) and result.get('name') != 'Unknown'
            threshold = confirmed_cleanup_threshold if is_confirmed else cleanup_threshold
            
            if current_time - data['timestamp'] > threshold:
                keys_to_remove.append(face_id)
        for key in keys_to_remove:
            self.latest_recognition1.pop(key, None)
        
        keys_to_remove = []
        for face_id, data in self.latest_recognition2.items():
            result = data.get('result', {})
            is_confirmed = result.get('validated', False) and result.get('name') != 'Unknown'
            threshold = confirmed_cleanup_threshold if is_confirmed else cleanup_threshold
            
            if current_time - data['timestamp'] > threshold:
                keys_to_remove.append(face_id)
        for key in keys_to_remove:
            self.latest_recognition2.pop(key, None)
        
        face_absence_threshold_extended = self.face_absence_threshold * 3
        keys_to_remove = []
        for face_id, last_seen in self.face_presence1.items():
            if current_time - last_seen > face_absence_threshold_extended:
                keys_to_remove.append(face_id)
        for key in keys_to_remove:
            self.face_presence1.pop(key, None)
        
        keys_to_remove = []
        for face_id, last_seen in self.face_presence2.items():
            if current_time - last_seen > face_absence_threshold_extended:
                keys_to_remove.append(face_id)
        for key in keys_to_remove:
            self.face_presence2.pop(key, None)
        
        confirmed_timeout_extended = self.confirmed_faces_timeout * 2
        keys_to_remove = []
        for face_id, confirmed_data in self.confirmed_faces.items():
            cam1_absent = face_id not in self.face_presence1 or (current_time - self.face_presence1.get(face_id, 0)) > face_absence_threshold_extended
            cam2_absent = face_id not in self.face_presence2 or (current_time - self.face_presence2.get(face_id, 0)) > face_absence_threshold_extended
            
            if cam1_absent and cam2_absent:
                if current_time - confirmed_data['timestamp'] > confirmed_timeout_extended:
                    keys_to_remove.append(face_id)
        
        for key in keys_to_remove:
            self.confirmed_faces.pop(key, None)
            if key in self.authenticated_faces:
                self.authenticated_faces.remove(key)
        
        
        
        auth_timeout = self.face_absence_threshold  
        keys_to_remove = []
        for face_id in list(self.authenticated_faces):
            cam1_absent = face_id not in self.face_presence1 or (current_time - self.face_presence1.get(face_id, 0)) > auth_timeout
            cam2_absent = face_id not in self.face_presence2 or (current_time - self.face_presence2.get(face_id, 0)) > auth_timeout
            
            if cam1_absent and cam2_absent:
                keys_to_remove.append(face_id)
        
        for key in keys_to_remove:
            self.authenticated_faces.discard(key)
            self.confirmed_faces.pop(key, None)
            self.face_liveness_confirmed.pop(key, None)
            self.face_liveness_status.pop(key, None)
            self.pending_recognition.pop(key, None)
            
            if hasattr(self, 'anti_spoofing_thread') and self.anti_spoofing_thread:
                self.anti_spoofing_thread.reset_face(key)
            print(f"[CLEANUP] Reset face: {key}")
        
        keys_to_remove = []
        for buffer_key, buffer_data in self.feature_buffers.items():
            buffer_age = current_time - buffer_data['start_time']
            face_id = buffer_data.get('face_id', '')
            
            face_absent = (face_id not in self.face_presence1 or (current_time - self.face_presence1.get(face_id, 0)) > self.face_absence_threshold) and \
                         (face_id not in self.face_presence2 or (current_time - self.face_presence2.get(face_id, 0)) > self.face_absence_threshold)
            
            if face_absent or buffer_age > 30.0:
                keys_to_remove.append(buffer_key)
        
        for key in keys_to_remove:
            self.feature_buffers.pop(key, None)
        
        
        keys_to_remove = []
        for face_id, pending in self.pending_recognition.items():
            pending_age = current_time - pending['timestamp']
            face_absent = (face_id not in self.face_presence1 or (current_time - self.face_presence1.get(face_id, 0)) > self.face_absence_threshold) and \
                         (face_id not in self.face_presence2 or (current_time - self.face_presence2.get(face_id, 0)) > self.face_absence_threshold)
            
            if face_absent or pending_age > 15.0:
                keys_to_remove.append(face_id)
        
        for key in keys_to_remove:
            self.pending_recognition.pop(key, None)
            print(f"[CLEANUP] Removed pending recognition for {key}")

    def immediate_cleanup_camera_results(self, camera_id):
        if camera_id == 0:
            if hasattr(self, 'latest_recognition1'):
                self.latest_recognition1.clear()
            if hasattr(self, 'face_presence1'):
                self.face_presence1.clear()
        elif camera_id == 1:
            if hasattr(self, 'latest_recognition2'):
                self.latest_recognition2.clear()
            if hasattr(self, 'face_presence2'):
                self.face_presence2.clear()
        
        if not self.current_faces1 and not self.current_faces2:
            current_time = time.time()
            self.feature_buffers.clear()
            if hasattr(self, 'confirmed_faces'):
                self.confirmed_faces.clear()
            if hasattr(self, 'authenticated_faces'):
                self.authenticated_faces.clear()
            if hasattr(self, 'face_liveness_confirmed'):
                self.face_liveness_confirmed.clear()
            if hasattr(self, 'face_liveness_status'):
                self.face_liveness_status.clear()
            if hasattr(self, 'anti_spoofing_thread') and self.anti_spoofing_thread:
                self.anti_spoofing_thread.reset_state()
            if hasattr(self, 'pending_recognition'):
                self.pending_recognition.clear()
    
    def update_display(self):
        self.cleanup_old_recognition_results()
        
        if self.current_frame1 is not None:
            frame1 = self.current_frame1.copy()
            self.display_frame(frame1, self.camera1_label)
        
        if self.current_frame2 is not None:
            frame2 = self.current_frame2.copy()
            self.display_frame(frame2, self.camera2_label)
        
        self.update_info_area()
    
    def update_info_area(self):
        current_time = time.time()
        
        latest_confirmed = None
        latest_time = 0
        
        
        for face_id, recognition_data in self.latest_recognition1.items():
            result = recognition_data.get('result', {})
            if result.get('confirmed_dual') and result.get('name') != 'Unknown':
                timestamp = recognition_data.get('timestamp', 0)
                if timestamp > latest_time:
                    latest_time = timestamp
                    latest_confirmed = result
        
        if not latest_confirmed:
            for face_id, recognition_data in self.latest_recognition2.items():
                result = recognition_data.get('result', {})
                if result.get('confirmed_dual') and result.get('name') != 'Unknown':
                    timestamp = recognition_data.get('timestamp', 0)
                    if timestamp > latest_time:
                        latest_time = timestamp
                        latest_confirmed = result
        
        if latest_confirmed:
            self.current_user_name = latest_confirmed.get('name', 'Unknown')
            self.current_user_id = latest_confirmed.get('id', 'N/A')
            
            
            if self.straight_face is not None:
                self.display_face_image(self.straight_face)
            else:
                face_img = self.load_user_face_image(self.current_user_id)
                if face_img is not None:
                    self.display_face_image(face_img)
                else:
                    self.face_image_label.clear()
                    self.face_image_label.setStyleSheet("background-color: #ffffff; border: 2px solid #333;")
        else:
            self.current_user_name = None
            self.current_user_id = None
            self.face_image_label.clear()
            self.face_image_label.setStyleSheet("background-color: #ffffff; border: 2px solid #333;")
        
        self.update_info_text()
    
    def load_user_face_image(self, user_id):
        try:
            if '_cam' in user_id:
                base_id = user_id.split('_cam')[0]
            else:
                base_id = user_id
            
            authenticated_folder = "/home/pi/KLTN_2025/authenticated_faces"
            if not os.path.exists(authenticated_folder):
                return None
            
            authenticated_images = [f for f in os.listdir(authenticated_folder) 
                                   if f.startswith(base_id) and f.endswith('.jpg')]
            
            if not authenticated_images:
                return None
            
            authenticated_images.sort(reverse=True)
            
            image_path = os.path.join(authenticated_folder, authenticated_images[0])
            face_img = cv2.imread(image_path)
            
            return face_img
        except Exception as e:
            print(f"[ERROR] Error loading face image for {user_id}: {e}")
            return None
    
    def display_face_image(self, face_img):
        try:
            target_w = self.face_image_label.width()
            target_h = self.face_image_label.height()
            
            h, w = face_img.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            face_resized = cv2.resize(face_img, (new_w, new_h))
            
            rgb_image = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.face_image_label.setPixmap(pixmap.scaled(target_w, target_h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.face_image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        except Exception as e:
            print(f"Error displaying face image: {e}")
    
    def update_info_text(self):
        from datetime import datetime
        now = datetime.now()
        date_str, time_str = now.strftime("%d/%m/%Y"), now.strftime("%H:%M:%S")
        
        
        if self.current_user_name and self.current_user_id:
            text = f"<b>Tên:</b> {self.current_user_name}<br><b>ID:</b> {self.current_user_id}<br><span style='color: green;'>✓ Xác thực thành công</span><hr>{date_str}<br>{time_str}"
            self.info_text.setStyleSheet("background-color: #e8f5e9; color: #000; border: 2px solid #4caf50; padding: 10px; font-size: 16px;")
        else:
            
            pending_name = None
            for face_id, pending in self.pending_recognition.items():
                result = pending.get('result', {})
                if result.get('name'):
                    pending_name = result['name']
                    break
            
            if pending_name:
                
                text = f"<b>Tên:</b> ---<br><b>ID:</b> ---<br><span style='color: orange;'> Vui lòng cười để xác thực bước cuối!</span><hr>{date_str}<br>{time_str}"
            else:
                
                text = f"<b>Tên:</b> ---<br><b>ID:</b> ---<hr>{date_str}<br>{time_str}"
            self.info_text.setStyleSheet("background-color: #fff; color: #888; border: 2px solid #ccc; padding: 10px; font-size: 16px;")
        
        self.info_text.setHtml(text)
    
    def update_clock(self):
        if hasattr(self, 'info_text'):
            self.update_info_text()

    def display_frame(self, frame, label):
        target_width = label.width()
        target_height = label.height()
        
        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def load_face_features(self,number):
        print('Loading')
        self.face_features = {}
        csv_path = "/home/pi/KLTN_2025/face_features.csv"
        if not os.path.exists(csv_path):
            csv_path = "face_features.csv" 
            if not os.path.exists(csv_path):
                return
        if os.stat(csv_path).st_size != 0:
            with open(csv_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                
                id_idx = header.index('id') if 'id' in header else 0
                name_idx = header.index('name') if 'name' in header else 1
                features_idx = header.index('features') if 'features' in header else 2
                for row in reader:
                    if len(row) > features_idx:
                        face_id = row[id_idx]
                        face_name = row[name_idx]
                        features_str = row[features_idx]
                        features = np.array([float(x) for x in features_str.split(',')])
                        
                        self.face_features[face_id] = {
                            'name': face_name,
                            'features': features
                        }
        
        if hasattr(self, 'recognition_thread'):
            self.recognition_thread.set_face_features(self.face_features)
            self.recognition_thread.set_params(self.similarity_threshold)

    def cleanup_resources_async(self, MainWindow):
        try:
            print("Bắt đầu cleanup tài nguyên...")
            
            if hasattr(self, 'timer'):
                self.timer.stop()
            
            threads_to_stop = []
            for attr in ['camera_thread1', 'camera_thread2',
                        'face_detection_thread1', 'face_detection_thread2',
                        'feature_thread1', 'feature_thread2', 'recognition_thread']:
                thread = getattr(self, attr, None)
                if thread:
                    threads_to_stop.append((attr, thread))
            
            for attr, thread in threads_to_stop:
                if hasattr(thread, 'stop'):
                    if hasattr(thread, 'running'):
                        thread.running = False
                    print(f"Đang dừng {attr}...")
            
            for attr, thread in threads_to_stop:
                if thread.isRunning():
                    thread.wait(50)
         
            self.switch_to_login(MainWindow)
            
        except Exception as e:
            print(f"Lỗi cleanup: {e}")
            import traceback
            traceback.print_exc()
            self.switch_to_login(MainWindow)
        
    def switch_to_login(self, MainWindow):
        try:
            if hasattr(self, 'progress'):
                self.progress.close()
            
            from tool.login import Ui_MainWindow as LoginWindow
            self.window = QtWidgets.QMainWindow()
            self.ui = LoginWindow()
            self.ui.setupUi(self.window)
            self.window.show()
            MainWindow.close()
        except Exception as e:
            print(f"Lỗi chuyển màn hình: {str(e)}")
            if hasattr(self, 'progress'):
                self.progress.close()
            try:
                QMessageBox.critical(MainWindow, "Lỗi", f"Không thể chuyển màn hình: {str(e)}")
            except:
                pass

    def return_to_login(self, MainWindow):
        try:
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            
            self.progress = QProgressDialog("Đang chuyển màn hình...", "", 0, 0, MainWindow)
            self.progress.setWindowTitle("Vui lòng đợi")
            self.progress.setWindowModality(Qt.ApplicationModal)
            self.progress.setCancelButton(None)
            self.progress.setMinimumDuration(0)
            self.progress.show()
            QApplication.processEvents()
            
            self.cleanup_resources_async(MainWindow)
            
        except Exception as e:
            print(f"Lỗi return_to_login: {str(e)}")
            if hasattr(self, 'progress'):
                self.progress.close()
            try:
                QMessageBox.critical(MainWindow, "Lỗi", f"Không thể chuyển màn hình: {str(e)}")
            except:
                pass

    def closeEvent(self, event):
        import gc
        print("[CLEANUP] Bắt đầu dừng tất cả threads...")
        self.timer.stop()
        
        if hasattr(self, 'clock_timer'):
            self.clock_timer.stop()
            
        if hasattr(self, 'sync_database'):
            self.sync_database.stop()
        
        if hasattr(self, 'display_thread1') and self.display_thread1:
            print("[CLEANUP] Dừng DisplayThread 1...")
            self.display_thread1.stop()
            self.display_thread1 = None
            
        if hasattr(self, 'display_thread2') and self.display_thread2:
            print("[CLEANUP] Dừng DisplayThread 2...")
            self.display_thread2.stop()
            self.display_thread2 = None
        
        if hasattr(self, 'camera_thread1') and self.camera_thread1:
            print("[CLEANUP] Dừng CameraThread 1...")
            self.camera_thread1.stop()
            self.camera_thread1 = None
            
        if hasattr(self, 'camera_thread2') and self.camera_thread2:
            print("[CLEANUP] Dừng CameraThread 2...")
            self.camera_thread2.stop()
            self.camera_thread2 = None
        
        if hasattr(self, 'face_detection_thread1') and self.face_detection_thread1:
            print("[CLEANUP] Dừng FaceDetectionThread 1...")
            self.face_detection_thread1.stop()
            self.face_detection_thread1 = None
            
        if hasattr(self, 'face_detection_thread2') and self.face_detection_thread2:
            print("[CLEANUP] Dừng FaceDetectionThread 2...")
            self.face_detection_thread2.stop()
            self.face_detection_thread2 = None
            
        if hasattr(self, 'feature_thread1') and self.feature_thread1:
            self.feature_thread1.stop()
            self.feature_thread1 = None
            
        if hasattr(self, 'feature_thread2') and self.feature_thread2:
            self.feature_thread2.stop()
            self.feature_thread2 = None
        
        if hasattr(self, 'anti_spoofing_thread') and self.anti_spoofing_thread:
            print("[CLEANUP] AntiSpoofingThread...")
            self.anti_spoofing_thread.stop()
            self.anti_spoofing_thread = None
        
        gc.collect()
        
        event.accept()

    def calculate_face_distance(self, f1, f2):
        c1, c2 = ((f1[0]+f1[2])/2, (f1[1]+f1[3])/2), ((f2[0]+f2[2])/2, (f2[1]+f2[3])/2)
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    
    def update_face_tracking(self, faces, camera_id):
        updated_faces = []
        for i, face_coords in enumerate(faces):
            face_coords = tuple(map(int, face_coords))
            face_id = f"cam{camera_id}_face_{i}"
            updated_faces.append((face_id, face_coords))
        return updated_faces

    def filter_largest_face(self, tracked_faces):
        if len(tracked_faces) <= 1: return tracked_faces
        fid, coords = max(tracked_faces, key=lambda f: abs(f[1][2]-f[1][0])*abs(f[1][3]-f[1][1]))
        return [(fid, coords)]

    def save_authenticated_face(self, straight_face, person_name, person_id, camera_id):
        save_dir = "/home/pi/KLTN_2025/authenticated_faces"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{person_id}_{person_name}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        cv2.imwrite(filepath, straight_face)
        return True

def rotate_image(image, angle, center=None):
    if center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    else:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
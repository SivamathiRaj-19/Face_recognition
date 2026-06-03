import cv2
import numpy as np
import time
import os
import csv
import glob
import sys
import math
import argparse
import threading
import queue
import signal

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from tool.hailo_model import HAILO
from hailo_service.hailo_model import SCRFD_HAILO
from cam_service.cam import RTSP_URL
from tool.database_interact import create_user

def rotate_image(image, angle, center = None):
    if center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    else:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

class CameraThread(threading.Thread):
    def __init__(self, camera_id, frame_queue, rtsp_url):
        super().__init__()
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.running = False
        self.frame_queue = frame_queue
        self.daemon = True
        
    def initialize_camera(self):
        try:
            print(f"Initializing camera {self.camera_id} at {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            return self.cap.isOpened()
        except Exception as e:
            print(f"Error initializing camera {self.camera_id}: {e}")
            return False
    
    def run(self):
        if not self.initialize_camera():
            print(f"Failed to initialize camera {self.camera_id}")
            return
        
        self.running = True
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Use a non-blocking put to drop frames if processing is too slow
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                else:
                    print(f"Warning: Failed to fetch frame from {self.rtsp_url}. Retrying...")
                    time.sleep(0.1)
            except Exception as e:
                print(f"Camera {self.camera_id} error: {e}")
                break
        
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def stop(self):
        self.running = False


class FaceRegistrationApp:
    def __init__(self, user_id, face_name, max_frames=100):
        self.user_id = user_id
        self.face_name = face_name
        self.max_frames = max_frames
        
        self.running = True
        self.frame_queue_cam1 = queue.Queue(maxsize=2)
        self.frame_queue_cam2 = queue.Queue(maxsize=2)
        
        self.captured_cam1 = []
        self.captured_cam2 = []
        
        print("Loading Models...")
        self.face_detector = SCRFD_HAILO("./models/scrfd_500m.hef", nms_iou_thresh=0.4, score_threshold=0.6)
        self.feature_model = HAILO("./models/arcface_mobilefacenet-2.hef", task=1, output_type={'arcface_mobilefacenet/fc1':'FLOAT32'})
        
        # Use RTSP_URL for both cameras (modify the second URL if you have a stream2)
        self.cam1_thread = CameraThread(0, self.frame_queue_cam1, RTSP_URL)
        
        rtsp_url2 = RTSP_URL.replace("stream1", "stream2") # Example for second camera
        self.cam2_thread = CameraThread(1, self.frame_queue_cam2, rtsp_url2)

    def validate_single_face_in_frame(self, detections, frame_shape):
        if detections is None or detections['num_detections'] == 0:
            return None
            
        frame_height, frame_width = frame_shape[:2]
        valid_faces = []
        
        boxes = detections['detection_boxes']
        scores = detections['detection_scores']
        landmarks = detections['face_landmarks']
        
        for i in range(detections['num_detections']):
            if scores[i] <= 0.6:
                continue
                
            x1, y1, x2, y2 = map(int, boxes[i])
            margin = 5
            if (x1 >= margin and y1 >= margin and 
                x2 <= frame_width - margin and y2 <= frame_height - margin):
                
                face_area = abs(x2 - x1) * abs(y2 - y1)
                valid_faces.append({
                    'coords': (x1, y1, x2, y2),
                    'landmarks': landmarks[i],
                    'area': face_area
                })
                
        if len(valid_faces) == 0:
            return None
        if len(valid_faces) == 1:
            return valid_faces[0]
            
        valid_faces.sort(key=lambda f: f['area'], reverse=True)
        if valid_faces[0]['area'] >= valid_faces[1]['area'] * 2:
            return valid_faces[0]
        return None

    def process_face(self, frame, valid_face):
        x1, y1, x2, y2 = valid_face['coords']
        landm = valid_face['landmarks']
        
        # Calculate rotation angle using eyes
        x11, y11 = int(landm[0]), int(landm[1]) # Left eye
        x22, y22 = int(landm[2]), int(landm[3]) # Right eye
        
        angle = 0
        if x11 != x22 or y11 != y22:
            try:
                angle_rad = math.atan2(y22 - y11, x22 - x11)
                angle = angle_rad * 180 / math.pi
            except:
                pass
                
        # Rotate and crop
        center = (x1 + int(abs(x1 - x2) / 2), y1 + int(abs(y1 - y2) / 2))
        img_rotated = rotate_image(frame, int(angle), center)
        
        # Ensure bounds
        h, w = img_rotated.shape[:2]
        crop_y1, crop_y2 = max(0, y1), min(h, y2)
        crop_x1, crop_x2 = max(0, x1), min(w, x2)
        
        face_crop = img_rotated[crop_y1:crop_y2, crop_x1:crop_x2, :]
        return face_crop

    def extract_face_features(self, face_img):
        tensor_output = self.feature_model([face_img])
        return tensor_output

    def find_best_feature_with_cosine_filtering(self, feature_vectors, similarity_threshold=0.85):
        if not feature_vectors or len(feature_vectors) == 0:
            return None, None
        
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
            return features_array[best_index], best_index
        
        valid_vectors = features_array[valid_indices]
        average_vector = np.mean(valid_vectors, axis=0)
        average_vector = average_vector / (np.linalg.norm(average_vector) + 1e-8)
        
        valid_vectors_norm = valid_vectors / (np.linalg.norm(valid_vectors, axis=1, keepdims=True) + 1e-8)
        similarities_to_avg = np.dot(valid_vectors_norm, average_vector)
        
        best_in_valid_idx = np.argmax(similarities_to_avg)
        best_global_idx = valid_indices[best_in_valid_idx]
        
        return features_array[best_global_idx], best_global_idx

    def save_features(self, camera_name, best_feature, best_frame, best_crop, id_offset=0):
        csv_path = "./face_features.csv"
        os.makedirs("faces", exist_ok=True)
        
        uid = str(int(self.user_id) + id_offset)
        display_name = f"{self.face_name}_{camera_name}"
        
        cv2.imwrite(f"faces/{uid}_{self.face_name}_{camera_name}_best.jpg", best_frame)
        cv2.imwrite(f"faces/{uid}_{self.face_name}_{camera_name}_crop.jpg", best_crop)
        
        features_str = ','.join(map(str, best_feature.tolist()))
        is_new_file = not os.path.exists(csv_path)
        
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow(['id', 'name', 'features'])
            writer.writerow([uid, display_name, features_str])
            
        try:
            print(f"[{camera_name}] Uploading to server...")
            success = create_user(display_name, best_feature.tolist())
            print(f"[{camera_name}] Upload state:", True if success else False)
        except Exception as e:
            print(f"[{camera_name}] Cannot connect to server:", e)

    def process_camera_captures(self, captured_list, camera_name, id_offset):
        if not captured_list:
            print(f"No frames captured for {camera_name}")
            return
            
        print(f"Processing {len(captured_list)} frames from {camera_name}...")
        feature_vectors = []
        
        for item in captured_list:
            face_crop = item['crop']
            try:
                features = self.extract_face_features(face_crop)
                if features is not None:
                    feature_vectors.append(features)
            except Exception as e:
                pass
                
        if len(feature_vectors) > 0:
            best_feature, best_idx = self.find_best_feature_with_cosine_filtering(feature_vectors)
            if best_feature is not None:
                best_item = captured_list[best_idx]
                self.save_features(camera_name, best_feature, best_item['frame'], best_item['crop'], id_offset)
                print(f"Successfully registered {self.face_name} on {camera_name}")
            else:
                print(f"Failed to find valid feature cluster for {camera_name}")
        else:
            print(f"No valid features extracted for {camera_name}")

    def run(self):
        self.cam1_thread.start()
        self.cam2_thread.start()
        print("Cameras started. Please position your face in front of both cameras.")
        
        while self.running:
            frame1, frame2 = None, None
            
            try:
                frame1 = self.frame_queue_cam1.get_nowait()
            except queue.Empty:
                pass
                
            try:
                frame2 = self.frame_queue_cam2.get_nowait()
            except queue.Empty:
                pass

            if frame1 is not None and len(self.captured_cam1) < self.max_frames:
                frame1_flipped = cv2.flip(frame1, 1)
                detections = self.face_detector(frame1_flipped)
                valid_face = self.validate_single_face_in_frame(detections, frame1_flipped.shape)
                
                if valid_face:
                    crop = self.process_face(frame1_flipped, valid_face)
                    if crop.size > 0:
                        self.captured_cam1.append({
                            'frame': frame1_flipped.copy(),
                            'crop': crop
                        })
                        x1, y1, x2, y2 = valid_face['coords']
                        cv2.rectangle(frame1_flipped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame1_flipped, f"Cam1: {len(self.captured_cam1)}/{self.max_frames}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame1_flipped, "Please center your face", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                cv2.imshow("Camera 1", frame1_flipped)

            if frame2 is not None and len(self.captured_cam2) < self.max_frames:
                frame2_flipped = cv2.flip(frame2, 1)
                detections = self.face_detector(frame2_flipped)
                valid_face = self.validate_single_face_in_frame(detections, frame2_flipped.shape)
                
                if valid_face:
                    crop = self.process_face(frame2_flipped, valid_face)
                    if crop.size > 0:
                        self.captured_cam2.append({
                            'frame': frame2_flipped.copy(),
                            'crop': crop
                        })
                        x1, y1, x2, y2 = valid_face['coords']
                        cv2.rectangle(frame2_flipped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame2_flipped, f"Cam2: {len(self.captured_cam2)}/{self.max_frames}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame2_flipped, "Please center your face", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                cv2.imshow("Camera 2", frame2_flipped)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if len(self.captured_cam1) >= self.max_frames and len(self.captured_cam2) >= self.max_frames:
                print("Capture complete. Processing features...")
                self.process_camera_captures(self.captured_cam1, "Camera1", 0)
                self.process_camera_captures(self.captured_cam2, "Camera2", 1)
                break
                
        self.stop()

    def stop(self):
        self.running = False
        self.cam1_thread.stop()
        self.cam2_thread.stop()
        cv2.destroyAllWindows()


def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Dual Camera Face Registration")
    parser.add_argument("--id", type=str, required=True, help="User ID to register")
    parser.add_argument("--name", type=str, required=True, help="User Name to register")
    parser.add_argument("--frames", type=int, default=100, help="Number of valid frames to capture per camera")
    
    args = parser.parse_args()
    
    app = FaceRegistrationApp(user_id=args.id, face_name=args.name, max_frames=args.frames)
    app.run()
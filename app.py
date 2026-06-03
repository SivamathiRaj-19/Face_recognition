import cv2
import numpy as np
import os
import math
import sys

# Ensure local imports work properly
root_path = os.path.abspath(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from hailo_service import HAILO, SCRFD_HAILO
from cam_service import RTSP_URL

DB_PATH = "face_db.npy"

def load_db():
    if os.path.exists(DB_PATH):
        try:
            return np.load(DB_PATH, allow_pickle=True).item()
        except:
            return {}
    return {}

def save_db(db):
    np.save(DB_PATH, db)
    print(f"[INFO] Database saved to {DB_PATH}")

def rotate_image(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def align_and_crop(frame, box, landmarks):
    x1, y1, x2, y2 = map(int, box)
    
    # Landmarks format: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
    # Indices 0-1: Left eye, Indices 2-3: Right eye
    left_eye_x, left_eye_y = int(landmarks[0]), int(landmarks[1])
    right_eye_x, right_eye_y = int(landmarks[2]), int(landmarks[3])
    
    angle = 0
    if left_eye_x != right_eye_x or left_eye_y != right_eye_y:
        angle_rad = math.atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
        angle = angle_rad * 180.0 / math.pi
        
    center = (x1 + int(abs(x1 - x2) / 2), y1 + int(abs(y1 - y2) / 2))
    img_rotated = rotate_image(frame, angle, center)
    
    h, w = img_rotated.shape[:2]
    crop_y1, crop_y2 = max(0, y1), min(h, y2)
    crop_x1, crop_x2 = max(0, x1), min(w, x2)
    
    return img_rotated[crop_y1:crop_y2, crop_x1:crop_x2, :]

def main():
    print("[INFO] Loading Face DB...")
    face_db = load_db()
    print(f"[INFO] Loaded {len(face_db)} registered faces.")

    print("[INFO] Initializing Models...")
    face_detector = SCRFD_HAILO("./models/scrfd_500m.hef", nms_iou_thresh=0.4, score_threshold=0.6)
    feature_model = HAILO("./models/arcface_mobilefacenet-2.hef", task=1, output_type={'arcface_mobilefacenet/fc1': 'FLOAT32'})

    print(f"[INFO] Connecting to Camera Stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("[ERROR] Failed to connect to camera.")
        return

    print("\n-----------------------------------------")
    print("Commands:")
    print(" - Press 's' to save the detected face.")
    print(" - Press 'q' to quit.")
    print("-----------------------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to fetch frame. Retrying...")
            continue
            
        display_frame = frame.copy()
        detections = face_detector(frame)
        
        valid_faces = []
        if detections is not None and detections['num_detections'] > 0:
            boxes = detections['detection_boxes']
            scores = detections['detection_scores']
            landmarks = detections['face_landmarks']
            
            for i in range(detections['num_detections']):
                if scores[i] > 0.6:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Score: {scores[i]:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Draw landmarks
                    landm = landmarks[i]
                    for j in range(5):
                        lx, ly = int(landm[j * 2]), int(landm[j * 2 + 1])
                        cv2.circle(display_frame, (lx, ly), 2, (0, 0, 255), -1)
                        
                    valid_faces.append({
                        'box': boxes[i],
                        'landmarks': landmarks[i]
                    })
                    
        cv2.imshow("Face Registration Application", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            if not valid_faces:
                print("[WARNING] No valid face detected to save!")
                continue
                
            # If multiple faces, we pick the largest one
            if len(valid_faces) > 1:
                valid_faces.sort(key=lambda f: abs(f['box'][2] - f['box'][0]) * abs(f['box'][3] - f['box'][1]), reverse=True)
                
            best_face = valid_faces[0]
            print("[INFO] Processing face...")
            
            face_crop = align_and_crop(frame, best_face['box'], best_face['landmarks'])
            
            if face_crop.size == 0:
                print("[ERROR] Crop failed due to out-of-bounds.")
                continue
                
            # Get feature embedding
            print("[INFO] Extracting features...")
            embedding = feature_model([face_crop])
            
            if embedding is not None:
                # Pause stream to ask for input
                name = input("\n[INPUT] Enter name for this face: ").strip()
                if name:
                    face_db[name] = embedding.flatten()
                    save_db(face_db)
                    print(f"[SUCCESS] Registered {name} successfully!\n")
                else:
                    print("[WARNING] Name cannot be empty. Registration aborted.\n")
            else:
                print("[ERROR] Failed to extract features.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

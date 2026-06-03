import cv2
import numpy as np
from hailo_service.hailo_model import SCRFD_HAILO

RTSP_URL = "rtsp://admin:Triton123@192.168.1.103:554/Streaming/Channels/101"
HEF_PATH = "./models/scrfd_500m.hef"
HEF2_PATH = "./models/arcface_mobilefacenet-2.hef"

def main():
    # Initialize our custom SCRFD face detector
    print("Loading Hailo model...")
    face_detector = SCRFD_HAILO(
        model_path=HEF_PATH, 
        nms_iou_thresh=0.4, 
        score_threshold=0.5
    )

    # Open the RTSP stream using OpenCV
    print(f"Connecting to stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Stream running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to fetch frame. Retrying...")
            continue
            
        # 1. Feed frame to our bounding box logic
        detections = face_detector(frame)
        
        # 2. Parse and draw results
        if detections is not None:
            boxes = detections["detection_boxes"]
            scores = detections["detection_scores"]
            landmarks = detections["face_landmarks"]
            
            for i in range(detections["num_detections"]):
                box = boxes[i]
                score = scores[i]
                landm = landmarks[i]
                
                # Draw bounding box
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw confidence score
                label = f"{score:.2f}"
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw the 5 facial landmarks
                for j in range(5):
                    lx, ly = int(landm[j * 2]), int(landm[j * 2 + 1])
                    cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

        # 3. Display the frame
        cv2.imshow("Hailo SCRFD Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

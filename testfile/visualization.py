import cv2

def draw_detections(frame, detections):
    """Draws bounding boxes, confidence, names, and landmarks on the frame."""
    out_frame = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        name = det.get("name", "Unknown")
        landmarks = det["landmarks"]
        
        # Draw BBox
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Label
        label = f"{name} {conf:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(out_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw Landmarks
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, (lx, ly) in enumerate(landmarks):
            cv2.circle(out_frame, (int(lx), int(ly)), 2, colors[i], -1)
            
    return out_frame
    
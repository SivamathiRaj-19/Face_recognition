import cv2
import logging
from ..sample.scrfd_detector import SCRFDDetector

logging.basicConfig(level=logging.INFO)

def main():
    detector = SCRFDDetector(hef_path='../models/scrfd_500m.hef', conf_thresh=0.6)
    
    img = cv2.imread('test_image.jpg')
    if img is None:
        print("Image not found.")
        return
        
    detections = detector.detect(img)
    print(detections)
    
    out_img = detector.draw(img, detections)
    cv2.imshow("Hailo SCRFD", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    detector.close()

if __name__ == "__main__":
    main()
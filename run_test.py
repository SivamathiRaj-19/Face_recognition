import cv2
import os
import numpy as np
import logging

# Import the wrapper we built
from wrapper.scrfd_detector import SCRFDDetector

# Set up clean logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    # 1. Define paths
    hef_path = 'models/scrfd_500m.hef'
    image_path = 'test_image.jpg'

    # Check if the model exists to prevent Hailo API crashes
    if not os.path.exists(hef_path):
        logging.error(f"Model not found at {hef_path}.")
        logging.error("Please place your compiled 'scrfd_500m.hef' in the 'models/' folder.")
        return

    # 2. Initialize the detector
    logging.info("Initializing Hailo SCRFD Detector and locking PCIe device...")
    try:
        detector = SCRFDDetector(
            hef_path=hef_path,
            input_size=(640, 640),
            conf_thresh=0.5,
            iou_thresh=0.4
        )
        logging.info("Model loaded onto Hailo hardware successfully!")
    except Exception as e:
        logging.error(f"Failed to initialize detector: {e}")
        return

    # 3. Load test image
    img = cv2.imread(image_path)
    if img is None:
        logging.warning(f"'{image_path}' not found.")
        logging.warning("Creating a blank 1920x1080 image just to test the tensor pipeline...")
        # Create a dummy image (BGR)
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(img, "Dummy Data (No Faces)", (600, 500), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # 4. Run Inference
    logging.info("Running preprocessing, Hailo inference, and postprocessing...")
    detections = detector.detect(img)
    
    # 5. Print Results
    logging.info(f"Detected {len(detections)} face(s).")
    for i, det in enumerate(detections):
        # Format the bbox and confidence for easy reading
        bbox = [int(x) for x in det['bbox']]
        conf = det['confidence']
        logging.info(f"  Face {i+1} -> BBox: {bbox}, Confidence: {conf:.2%}")

    # 6. Draw and Save Output
    out_img = detector.draw(img, detections)
    output_filename = 'test_output.jpg'
    cv2.imwrite(output_filename, out_img)
    logging.info(f"Saved visual output to '{output_filename}'.")

    # 7. Cleanup
    detector.close()
    logging.info("Hailo hardware released. Test complete.")

if __name__ == "__main__":
    main()
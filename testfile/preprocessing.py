import cv2
import numpy as np

def preprocess_image(frame: np.ndarray, input_size: tuple = (640, 640)):
    """
    Applies letterbox resizing to maintain aspect ratio, and converts BGR to RGB.
    Returns the padded image and the scaling parameters.
    """
    img_h, img_w = frame.shape[:2]
    target_w, target_h = input_size
    
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    # Hailo typically requires RGB
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Pad to target size
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = rgb_img
    
    # Scale params to map bounding boxes back to original image
    scale_params = {
        'scale': scale,
        'pad_w': pad_w,
        'pad_h': pad_h
    }
    
    return canvas, scale_params
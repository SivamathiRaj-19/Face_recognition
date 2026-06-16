import numpy as np

def decode_outputs(outputs: dict, anchors: dict, strides: list, conf_thresh: float):
    """
    Decodes network outputs into bounding boxes and landmarks.
    Expects outputs mapped by stride, containing 'scores', 'bboxes', 'landmarks'.
    """
    all_boxes, all_scores, all_landmarks = [], [], []

    for stride in strides:
        scores = outputs[stride]['scores']
        bboxes = outputs[stride]['bboxes']
        kps = outputs[stride]['landmarks']
        
        anchor_centers = anchors[stride]
        
        # Flatten Hailo outputs (typically NHWC) -> (N, ...)
        scores = scores.reshape(-1)
        bboxes = bboxes.reshape(-1, 4)
        kps = kps.reshape(-1, 10)
        
        # Filter by confidence early to save compute
        valid_mask = scores > conf_thresh
        valid_scores = scores[valid_mask]
        valid_boxes = bboxes[valid_mask]
        valid_kps = kps[valid_mask]
        valid_anchors = anchor_centers[valid_mask]
        
        if len(valid_scores) == 0:
            continue
            
        # Decode BBoxes (distance -> coordinates)
        # Bbox is [left, top, right, bottom] distances
        x1 = valid_anchors[:, 0] - valid_boxes[:, 0] * stride
        y1 = valid_anchors[:, 1] - valid_boxes[:, 1] * stride
        x2 = valid_anchors[:, 0] + valid_boxes[:, 2] * stride
        y2 = valid_anchors[:, 1] + valid_boxes[:, 3] * stride
        decoded_boxes = np.stack([x1, y1, x2, y2], axis=-1)
        
        # Decode Landmarks (distance -> coordinates)
        decoded_kps = np.zeros_like(valid_kps)
        for i in range(5):
            decoded_kps[:, i*2] = valid_anchors[:, 0] + valid_kps[:, i*2] * stride
            decoded_kps[:, i*2+1] = valid_anchors[:, 1] + valid_kps[:, i*2+1] * stride
            
        all_boxes.append(decoded_boxes)
        all_scores.append(valid_scores)
        all_landmarks.append(decoded_kps)

    if not all_boxes:
        return np.array([]), np.array([]), np.array([])

    return np.vstack(all_boxes), np.concatenate(all_scores), np.vstack(all_landmarks)

def scale_back(boxes, landmarks, scale_params):
    """Maps decoded coordinates back to original image scale."""
    scale = scale_params['scale']
    pad_w = scale_params['pad_w']
    pad_h = scale_params['pad_h']
    
    # Unpad and unscale boxes
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
    
    # Unpad and unscale landmarks
    for i in range(5):
        landmarks[:, i*2] = (landmarks[:, i*2] - pad_w) / scale
        landmarks[:, i*2+1] = (landmarks[:, i*2+1] - pad_h) / scale
        
    return boxes, landmarks
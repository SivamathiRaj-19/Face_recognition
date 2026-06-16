import numpy as np

def generate_anchors(width: int, height: int, strides: list) -> dict:
    """Generates anchor centers for given strides."""
    anchors = {}
    for stride in strides:
        num_grid_w = width // stride
        num_grid_h = height // stride
        
        # Create grid centers
        x, y = np.meshgrid(np.arange(num_grid_w), np.arange(num_grid_h))
        x = (x.flatten() * stride) + (stride // 2)
        y = (y.flatten() * stride) + (stride // 2)
        
        # Typically SCRFD uses 1 or 2 anchors per location. Assuming 1 for standard 500m.
        # Shape: (num_grid_w * num_grid_h, 2)
        anchors[stride] = np.stack([x, y], axis=-1)
    return anchors

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4):
    """Applies Non-Maximum Suppression (NMS)."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
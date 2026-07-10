from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import time

import cv2
import numpy as np
from loguru import logger

# Relative imports avoid the circular-import issue that occurs when
# this module is imported via hailo_service/__init__.py
from .object_detection_utils import ObjectDetectionUtils
from . import utils

# ---------------------------------------------------------------------------
# Label file — required by ObjectDetectionUtils constructor
# ---------------------------------------------------------------------------

_LABELS_PATH = str(Path(__file__).parent / "label.txt")


# ---------------------------------------------------------------------------
# HAILO — generic async inference wrapper (detection OR recognition)
# ---------------------------------------------------------------------------


class HAILO:
    """
    Generic Hailo inference wrapper.

    Args:
        model_path (str): Path to the .hef model file.
        task (int): 0 = object detection (returns raw output dict),
                    1 = recognition (returns embedding array).
        output_type (dict | None): Per-layer output format overrides,
            e.g. ``{'arcface_mobilefacenet/fc1': 'FLOAT32'}``.
    """

    def __init__(self, model_path, task=0, output_type=None):
        # Lazy import avoids circular dependency through __init__.py
        from .hailo_model_manager import HailoModelManager

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.task = task
        self.utils = ObjectDetectionUtils(_LABELS_PATH)

        vdevice = HailoModelManager.get_device()

        hailo_inference = utils.HailoAsyncInference(
            model_path,
            self.input_queue,
            self.output_queue,
            batch_size=1,
            output_type=output_type,
            send_original_frame=True,
            vdevice=vdevice,
        )
        self.height, self.width, _ = hailo_inference.get_input_shape()

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hailo_generic")
        self._executor.submit(hailo_inference.run)

    def __call__(self, frame):
        # Accept both a raw frame and a single-item list (feature_model([crop]))
        if isinstance(frame, list):
            if len(frame) == 0:
                return None
            frame = frame[0]

        if frame is None:
            raise ValueError("Input frame is None")

        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = self.utils.preprocess(processed, self.width, self.height)

        self.input_queue.put(([frame], [processed]))

        timeout_start = time.monotonic()
        while self.output_queue.empty():
            if time.monotonic() - timeout_start > 0.5:
                logger.warning("Hailo inference timeout — skipping frame.")
                return [] if self.task == 0 else None
            time.sleep(0.001)

        _original_frame, result = self.output_queue.get()
        return result

    def extract_boxes_only(self, detections: dict, image: np.ndarray, min_score: float = 0.45):
        boxes = detections["detection_boxes"]
        scores = detections["detection_scores"]

        img_height, img_width = image.shape[:2]
        size = max(img_height, img_width)
        padding_length = int(abs(img_height - img_width) / 2)
        scaled_box_list = []

        for idx in range(detections["num_detections"]):
            if scores[idx] >= min_score:
                scaled_box = self.utils.denormalize_and_rm_pad(
                    boxes[idx], size, padding_length, img_height, img_width
                )
                scaled_box_list.append(scaled_box)

        return scaled_box_list


# ---------------------------------------------------------------------------
# SCRFD_HAILO — SCRFD face detector with built-in postprocessing
# ---------------------------------------------------------------------------


class SCRFD_HAILO:
    """
    Hailo-accelerated SCRFD face detector.

    Runs SCRFD 500M on the Hailo device and returns structured detections
    with pixel-coordinate bounding boxes and facial landmarks.

    Args:
        model_path (str): Path to the SCRFD .hef file.
        nms_iou_thresh (float): IoU threshold for NMS. Default 0.40.
        score_threshold (float): Minimum detection confidence. Default 0.60.
    """

    _ANCHORS = {
        "steps": [8, 16, 32],
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
    }

    def __init__(self, model_path, nms_iou_thresh=0.40, score_threshold=0.60):
        from .hailo_model_manager import HailoModelManager

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.utils = ObjectDetectionUtils(_LABELS_PATH)

        vdevice = HailoModelManager.get_device()

        hailo_inference = utils.HailoAsyncInference(
            model_path,
            self.input_queue,
            self.output_queue,
            batch_size=1,
            output_type=None,       # SCRFD uses native UINT8 output
            send_original_frame=True,
            vdevice=vdevice,
        )
        self.height, self.width, _ = hailo_inference.get_input_shape()

        self.postproc = SCRFDPostProc(
            image_dims=(self.height, self.width),
            nms_iou_thresh=nms_iou_thresh,
            score_threshold=score_threshold,
            anchors=self._ANCHORS,
        )

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hailo_scrfd")
        self._executor.submit(hailo_inference.run)

    def __call__(self, frame):
        if frame is None:
            return None

        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = self.utils.preprocess(processed, self.width, self.height)

        self.input_queue.put(([frame], [processed]))

        timeout_start = time.monotonic()
        while self.output_queue.empty():
            if time.monotonic() - timeout_start > 0.5:
                logger.warning("SCRFD inference timeout — skipping frame.")
                return None
            time.sleep(0.001)

        _original_frame, raw_output = self.output_queue.get()

        # Step 1: dequantize the raw UINT8 Hailo outputs
        rescaled = rescale_network_outputs(raw_output)

        # Step 2: anchor-based decoding + NMS
        detections = self.postproc.main(rescaled)

        # detections['detection_boxes']  shape (1, N, 4)  — normalised [0,1]
        # detections['detection_scores'] shape (1, N)
        # detections['face_landmarks']   shape (1, N, 10) — normalised [0,1]

        h, w = frame.shape[:2]
        n_detected = detections["detection_boxes"].shape[1]

        if n_detected > 0:
            boxes = detections["detection_boxes"][0].copy()       # (N, 4)
            scores = detections["detection_scores"][0].copy()     # (N,)
            landmarks = detections["face_landmarks"][0].copy()    # (N, 10)

            # Denormalise → pixel coordinates
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * w, 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * h, 0, h)
            landmarks[:, 0::2] = np.clip(landmarks[:, 0::2] * w, 0, w)
            landmarks[:, 1::2] = np.clip(landmarks[:, 1::2] * h, 0, h)

            return {
                "detection_boxes": boxes,
                "detection_scores": scores,
                "face_landmarks": landmarks,
                "num_detections": int(n_detected),
            }

        return {
            "detection_boxes": np.empty((0, 4), dtype=np.float32),
            "detection_scores": np.empty((0,), dtype=np.float32),
            "face_landmarks": np.empty((0, 10), dtype=np.float32),
            "num_detections": 0,
        }


# ---------------------------------------------------------------------------
# NMS helpers (used by SCRFDPostProc)
# ---------------------------------------------------------------------------


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_a[:, None] + area_b - area_inter)


def non_max_suppression(
    prediction_boxes: np.ndarray,
    prediction_scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    classes = np.ones_like(prediction_scores)
    predictions = np.concatenate(
        [prediction_boxes, prediction_scores[:, np.newaxis], classes[:, np.newaxis]],
        axis=1,
    )
    rows, _ = predictions.shape
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes) - np.eye(rows)

    keep = np.ones(rows, dtype=bool)
    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]


# ---------------------------------------------------------------------------
# SCRFDPostProc — anchor-based decoding + NMS for SCRFD outputs
# ---------------------------------------------------------------------------


class SCRFDPostProc:
    NUM_CLASSES = 1
    NUM_LANDMARKS = 10
    LABEL_OFFSET = 1

    def __init__(self, image_dims, nms_iou_thresh, score_threshold, anchors):
        self._image_dims = image_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._num_branches = len(anchors["steps"])
        self.anchors = anchors
        if anchors is None:
            raise ValueError("Missing detection anchors metadata")
        self._anchors = self.extract_anchors(anchors["min_sizes"], anchors["steps"])

    def collect_box_class_predictions(self, output_branches):
        box_list, class_list, landmark_list = [], [], []
        n = self._num_branches
        assert len(output_branches) % n == 0
        nodes_per_branch = len(output_branches) // n

        for branch_index in range(0, len(output_branches), nodes_per_branch):
            batch = output_branches[branch_index].shape[0]
            box_list.append(output_branches[branch_index].reshape(batch, -1, 4))
            class_list.append(
                output_branches[branch_index + 1].reshape(batch, -1, self.NUM_CLASSES)
            )
            if nodes_per_branch > 2:
                landmark_list.append(
                    output_branches[branch_index + 2].reshape(batch, -1, 10)
                )

        boxes = np.concatenate(box_list, axis=1)
        classes = np.concatenate(class_list, axis=1)
        landmarks = np.concatenate(landmark_list, axis=1) if landmark_list else None
        return boxes, classes, landmarks

    def extract_anchors(self, min_sizes, steps):
        anchors = []
        for stride, min_size in zip(steps, min_sizes):
            height = self._image_dims[0] // stride
            width = self._image_dims[1] // stride
            num_anchors = len(min_size)

            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            centers[:, 0] /= self._image_dims[0]
            centers[:, 1] /= self._image_dims[1]

            if num_anchors > 1:
                centers = np.stack([centers] * num_anchors, axis=1).reshape((-1, 2))

            scales = np.ones_like(centers) * stride
            scales[:, 0] /= self._image_dims[0]
            scales[:, 1] /= self._image_dims[1]
            anchors.append(np.concatenate([centers, scales], axis=1))

        return np.concatenate(anchors, axis=0)

    def _decode_boxes(self, box_detections, anchors):
        x1 = anchors[:, 0] - box_detections[:, 0] * anchors[:, 2]
        y1 = anchors[:, 1] - box_detections[:, 1] * anchors[:, 3]
        x2 = anchors[:, 0] + box_detections[:, 2] * anchors[:, 2]
        y2 = anchors[:, 1] + box_detections[:, 3] * anchors[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _decode_landmarks(self, landmarks_detections, anchors):
        preds = []
        for i in range(0, self.NUM_LANDMARKS, 2):
            px = anchors[:, 0] + landmarks_detections[:, i] * anchors[:, 2]
            py = anchors[:, 1] + landmarks_detections[:, i + 1] * anchors[:, 3]
            preds.extend([px, py])
        return np.stack(preds, axis=-1)

    def main(self, endnodes):
        box_preds, class_preds, landmark_preds = self.collect_box_class_predictions(endnodes)

        batch_size, num_proposals = box_preds.shape[:2]
        tiled_anchors = np.tile(self._anchors[np.newaxis, :, :], [batch_size, 1, 1])
        flat_anchors = tiled_anchors.reshape(-1, 4)

        decoded_boxes = self._decode_boxes(box_preds.reshape(-1, 4), flat_anchors)
        detection_boxes = decoded_boxes.reshape(batch_size, num_proposals, 4)

        decoded_landmarks = self._decode_landmarks(
            landmark_preds.reshape(-1, 10), flat_anchors
        )
        decoded_landmarks = decoded_landmarks.reshape(batch_size, num_proposals, 10)

        detection_boxes = np.expand_dims(detection_boxes, axis=2)
        nmsed_boxes, nmsed_scores, nmsed_landmarks = self._batch_multiclass_nms(
            boxes=detection_boxes,
            scores=class_preds,
            landmarks=decoded_landmarks,
        )

        return {
            "detection_boxes": nmsed_boxes,
            "detection_scores": nmsed_scores,
            "num_detections": nmsed_scores.size,
            "face_landmarks": nmsed_landmarks,
        }

    def _batch_multiclass_nms(self, boxes, scores, landmarks):
        assert boxes.shape[0] == 1, "Batch size must be 1"
        batch_boxes = boxes[0, :, 0, :]
        batch_scores = scores[0, :, 0]
        batch_landmarks = landmarks[0]

        mask = batch_scores >= self._score_threshold
        filtered_boxes = batch_boxes[mask]
        filtered_scores = batch_scores[mask]
        filtered_landmarks = batch_landmarks[mask]

        if len(filtered_boxes) > 0:
            keep = non_max_suppression(filtered_boxes, filtered_scores, self._nms_iou_thresh)
            nmsed_boxes = filtered_boxes[keep]
            nmsed_scores = filtered_scores[keep]
            nmsed_landmarks = filtered_landmarks[keep]
        else:
            nmsed_boxes = np.empty((0, 4))
            nmsed_scores = np.empty((0,))
            nmsed_landmarks = np.empty((0, 10))

        return (
            np.expand_dims(nmsed_boxes, axis=0),
            np.expand_dims(nmsed_scores, axis=0),
            np.expand_dims(nmsed_landmarks, axis=0),
        )


# ---------------------------------------------------------------------------
# rescale_network_outputs — SCRFD UINT8 dequantization
# ---------------------------------------------------------------------------


def rescale_network_outputs(outputs):
    box_layers = {"scrfd_500m/conv27", "scrfd_500m/conv33", "scrfd_500m/conv39"}
    class_layers = {"scrfd_500m/conv26", "scrfd_500m/conv32", "scrfd_500m/conv38"}
    landmark_layers = {"scrfd_500m/conv25", "scrfd_500m/conv34", "scrfd_500m/conv40"}

    rescaled = []
    for output_name, output in outputs.items():
        output = output.astype(np.float32)

        if output_name in box_layers:
            output = output / 32.0
        elif output_name in class_layers:
            output = output / 255.0
        elif output_name in landmark_layers:
            output = (output - 113) / 29.0
        else:
            raise ValueError(f"Unknown SCRFD output layer: {output_name!r}")

        rescaled.append(output.reshape(1, -1, output.shape[-1]))

    return rescaled
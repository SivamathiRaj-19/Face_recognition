from concurrent.futures import ThreadPoolExecutor
import queue
from tool.object_detection_utils import ObjectDetectionUtils
from tool import utils
import time
import cv2
import numpy as np


class HAILO():
    def __init__(self, model_path, task = 0, output_type=None):
        executor = ThreadPoolExecutor(2)
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        batch_size = 1
        self.utils = ObjectDetectionUtils("label.txt")

        hailo_inference= utils.HailoAsyncInference(
            model_path, 
            self.input_queue, self.output_queue, batch_size, output_type=output_type, send_original_frame=True
        )
        self.height, self.width, _ = hailo_inference.get_input_shape()

        self.task=task
        executor.submit(hailo_inference.run)


    def __call__(self, frame):
        
        if frame is None:
            raise ValueError("Input Frame is None")
        
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = self.utils.preprocess(processed_frame, self.width, self.height)

        self.input_queue.put(([frame], [processed_frame]))
        
        # Tối ưu hóa: Thêm timeout và non-blocking check
        timeout_start = time.time()
        while self.output_queue.empty():
            if time.time() - timeout_start > 0.1:  # Timeout 100ms thay vì vô hạn
                print("Hailo inference timeout, skipping frame")
                return [] if self.task == 0 else None
            time.sleep(0.001)  # Sleep ngắn để giảm CPU usage

        result = self.output_queue.get()
        return result

    def extract_boxes_only(self, detections: dict, image: np.ndarray, min_score: float = 0.45):
        """
        Extract bounding boxes without drawing them on the image.
        """
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']

        img_height, img_width = image.shape[:2]
        size = max(img_height, img_width)
        padding_length = int(abs(img_height - img_width) / 2)
        scaled_box_list = []
        
        for idx in range(detections['num_detections']):
            if scores[idx] >= min_score:
                scaled_box = self.utils.denormalize_and_rm_pad(boxes[idx], size, padding_length, img_height, img_width)
                scaled_box_list.append(scaled_box)

        return scaled_box_list

def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:

        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        area_a = box_area(boxes_a.T)
        area_b = box_area(boxes_b.T)

        top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
        bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

        area_inter = np.prod(
            np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
            
        return area_inter / (area_a[:, None] + area_b - area_inter)


def non_max_suppression(prediction_boxes: np.ndarray, prediction_scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    classes = np.ones_like(prediction_scores) # Note: Only supports one class
    
    # Reshape our values to expected shape [x1, y1, x2, y2, score, class]
    predictions = np.concatenate([prediction_boxes, prediction_scores[:, np.newaxis], classes[:, np.newaxis]], axis=1)
    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]

class SCRFDPostProc(object):
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
        box_predictors_list = []
        class_predictors_list = []
        landmarks_predictors_list = []
        num_branches = self._num_branches
        assert len(output_branches) % num_branches == 0, "All branches must have the same number of output nodes"
        num_output_nodes_per_branch = len(output_branches) // num_branches
        
        for branch_index in range(0, len(output_branches), num_output_nodes_per_branch):
            num_of_batches = output_branches[branch_index].shape[0]
            box_predictors_list.append(output_branches[branch_index].reshape(num_of_batches, -1, 4))
            class_predictors_list.append(
                output_branches[branch_index + 1].reshape(num_of_batches, -1, self.NUM_CLASSES)
            )

            if num_output_nodes_per_branch > 2:
                landmarks_predictors_list.append(
                    output_branches[branch_index + 2].reshape(num_of_batches, -1, 10)
                )

        box_predictors = np.concatenate(box_predictors_list, axis=1)
        class_predictors = np.concatenate(class_predictors_list, axis=1)
        landmarks_predictors = np.concatenate(landmarks_predictors_list, axis=1) if landmarks_predictors_list else None
        return box_predictors, class_predictors, landmarks_predictors

    def extract_anchors(self, min_sizes, steps):
        anchors = []
        for stride, min_size in zip(steps, min_sizes):
            height = self._image_dims[0] // stride
            width = self._image_dims[1] // stride
            num_anchors = len(min_size)

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers[:, 0] /= self._image_dims[0]
            anchor_centers[:, 1] /= self._image_dims[1]
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            anchor_scales = np.ones_like(anchor_centers, dtype=np.float32) * stride
            anchor_scales[:, 0] /= self._image_dims[0]
            anchor_scales[:, 1] /= self._image_dims[1]
            anchor = np.concatenate([anchor_centers, anchor_scales], axis=1)
            anchors.append(anchor)
        return np.concatenate(anchors, axis=0)

    def _decode_landmarks(self, landmarks_detections, anchors):
        preds = []
        for i in range(0, self.NUM_LANDMARKS, 2):
            px = anchors[:, 0] + landmarks_detections[:, i] * anchors[:, 2]
            py = anchors[:, 1] + landmarks_detections[:, i + 1] * anchors[:, 3]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _decode_boxes(self, box_detections, anchors):
        x1 = anchors[:, 0] - box_detections[:, 0] * anchors[:, 2]
        y1 = anchors[:, 1] - box_detections[:, 1] * anchors[:, 3]
        x2 = anchors[:, 0] + box_detections[:, 2] * anchors[:, 2]
        y2 = anchors[:, 1] + box_detections[:, 3] * anchors[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def main(self, endnodes):
        box_predictions, classes_predictions, landmarks_predictors = self.collect_box_class_predictions(endnodes)
        additional_fields = {}

        detection_scores = classes_predictions
        batch_size, num_proposals = box_predictions.shape[:2]

        tiled_anchor_boxes = np.tile(self._anchors[np.newaxis, :, :], [batch_size, 1, 1])
        tiled_anchors_boxlist = tiled_anchor_boxes.reshape(-1, 4)

        decoded_boxes = self._decode_boxes(box_predictions.reshape(-1, 4), tiled_anchors_boxlist)
        detection_boxes = decoded_boxes.reshape(batch_size, num_proposals, 4)

        decoded_landmarks = self._decode_landmarks(
            landmarks_predictors.reshape(-1, 10), tiled_anchors_boxlist
        )
        decoded_landmarks = decoded_landmarks.reshape(batch_size, num_proposals, 10)
        detection_boxes = np.expand_dims(detection_boxes, axis=2)

        nmsed_boxes, nmsed_scores, nmsed_landmarks = (
            self._batch_multiclass_nms(
                boxes=detection_boxes,
                scores=detection_scores,
                landmarks=decoded_landmarks
            )
        )

        num_detections = nmsed_scores.size
        results = {
            "detection_boxes": nmsed_boxes,
            "detection_scores": nmsed_scores,
            "num_detections": num_detections,
            "face_landmarks": nmsed_landmarks
        }

        return results

    def _batch_multiclass_nms(self, boxes, scores, landmarks):
        assert boxes.shape[0] == 1, "Batch size must be 1"

        batch_boxes = boxes[0, :, 0, :]
        batch_scores = scores[0, :, 0]
        batch_landmarks = landmarks[0]

        score_mask = batch_scores >= self._score_threshold
        filtered_boxes = batch_boxes[score_mask]
        filtered_scores = batch_scores[score_mask]
        filtered_landmarks = batch_landmarks[score_mask]

        if len(filtered_boxes) > 0:
            keep_indices = non_max_suppression(
                filtered_boxes, 
                filtered_scores, 
                iou_threshold=self._nms_iou_thresh
            )

            nmsed_boxes = filtered_boxes[keep_indices]
            nmsed_scores = filtered_scores[keep_indices]
            nmsed_landmarks = filtered_landmarks[keep_indices]
        else:
            nmsed_boxes = np.empty((0, 4))
            nmsed_scores = np.empty((0,))
            nmsed_landmarks = np.empty((0, 10))

        nmsed_boxes = np.expand_dims(nmsed_boxes, axis=0)
        nmsed_scores = np.expand_dims(nmsed_scores, axis=0)
        nmsed_landmarks = np.expand_dims(nmsed_landmarks, axis=0)
        
        return nmsed_boxes, nmsed_scores, nmsed_landmarks


def rescale_network_outputs(outputs):

    box_layer_names = ["scrfd_500m/conv27", "scrfd_500m/conv33", "scrfd_500m/conv39"]
    class_layer_names = ["scrfd_500m/conv26", "scrfd_500m/conv32", "scrfd_500m/conv38"]
    landmark_layer_names = ["scrfd_500m/conv25", "scrfd_500m/conv34", "scrfd_500m/conv40"]

    rescaled_outputs = []
    for output_name, output in outputs.items():
        output = output.astype(np.float32)

        if output_name in box_layer_names:
            downscale_factor = 32
            output = output / downscale_factor
        elif output_name in class_layer_names:
            output = output / 255
        elif output_name in landmark_layer_names:
            zero_point = 113
            scale = 29
            output = (output - zero_point) / scale
        else:
            raise ValueError(f"Unknown output name: {output_name}")
        
        reshaped = output.reshape(1, -1, output.shape[-1])
        rescaled_outputs.append(reshaped)

    return rescaled_outputs
    
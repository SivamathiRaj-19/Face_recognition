import logging
from contextlib import ExitStack
import numpy as np

# Hailo platform API (v4.x+)
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, FormatType)

from .preprocessing import preprocess_image
from .postprocessing import decode_outputs, scale_back
from .visualization import draw_detections
from .utils import generate_anchors, nms

logger = logging.getLogger(__name__)

class SCRFDDetector:
    def __init__(self, hef_path: str, input_size: tuple = (640, 640), 
                 conf_thresh: float = 0.5, iou_thresh: float = 0.4):
        self.hef_path = hef_path
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.strides = [8, 16, 32]
        
        # Precompute anchors
        self.anchors = generate_anchors(input_size[0], input_size[1], self.strides)
        
        # Hailo Context Management
        self.exit_stack = ExitStack()
        self.target = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None
        
        self.load_model()
        logger.info(f"SCRFD model loaded from {hef_path}")

    def load_model(self):
        """Initializes the Hailo VDevice and configures VStreams."""
        try:
            hef = HEF(self.hef_path)
            self.target = self.exit_stack.enter_context(VDevice())
            
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.target.configure(hef, configure_params)[0]
            network_group_params = self.network_group.create_params()

            # Prepare VStreams
            input_vstream_info = hef.get_input_vstream_infos()
            output_vstream_info = hef.get_output_vstream_infos()

            input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.UINT8)
            output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

            # Create context for VStreams
            self.input_vstreams = self.exit_stack.enter_context(
                self.network_group.InferVStreamGroup(input_vstreams_params, output_vstreams_params)
            )
            
            # Extract input name for feed dictionary mapping
            self.input_name = input_vstream_info[0].name
            
            # Map outputs to strides based on tensor shape/names
            # Note: You may need to adjust these strings based on your specific SCRFD HEF compilation
            self.output_mapping = {}
            for info in output_vstream_info:
                name = info.name
                # Deduce stride logically or map via hardcoded strings based on Hailo Parser
                if 'stride8' in name or '6400' in name or info.shape[1] == 80: stride = 8
                elif 'stride16' in name or '1600' in name or info.shape[1] == 40: stride = 16
                else: stride = 32
                
                if stride not in self.output_mapping:
                    self.output_mapping[stride] = {}
                    
                if 'score' in name or info.shape[-1] <= 2: self.output_mapping[stride]['scores'] = name
                elif 'bbox' in name or info.shape[-1] == 4: self.output_mapping[stride]['bboxes'] = name
                else: self.output_mapping[stride]['landmarks'] = name
                
        except Exception as e:
            logger.error(f"Failed to load HEF model: {e}")
            self.close()
            raise e

    def preprocess(self, frame: np.ndarray):
        """Prepares standard frame for Hailo infer."""
        tensor, scale_params = preprocess_image(frame, self.input_size)
        # Expand dims for batch size 1
        return np.expand_dims(tensor, axis=0), scale_params

    def infer(self, input_tensor: np.ndarray) -> dict:
        """Executes inference via Hailo VStreams."""
        input_data = {self.input_name: input_tensor}
        self.input_vstreams.set_input(input_data)
        
        # Get infer results
        self.input_vstreams.wait_for_ready()
        infer_results = self.input_vstreams.get_output()
        
        # Map raw Hailo outputs back to our expected dict structure
        structured_outputs = {8: {}, 16: {}, 32: {}}
        for stride in self.strides:
            maps = self.output_mapping[stride]
            structured_outputs[stride]['scores'] = infer_results[maps['scores']]
            structured_outputs[stride]['bboxes'] = infer_results[maps['bboxes']]
            structured_outputs[stride]['landmarks'] = infer_results[maps['landmarks']]
            
        return structured_outputs

    def decode(self, outputs: dict, scale_params: dict):
        """Decodes raw tensors into coordinates, and scales them back."""
        boxes, scores, kps = decode_outputs(outputs, self.anchors, self.strides, self.conf_thresh)
        if len(boxes) > 0:
            boxes, kps = scale_back(boxes, kps, scale_params)
        return boxes, scores, kps

    def apply_nms(self, boxes, scores):
        """Wraps NMS utility."""
        return nms(boxes, scores, self.iou_thresh)

    def detect(self, frame: np.ndarray) -> list:
        """Main pipeline sequence for detection."""
        try:
            # 1. Preprocess
            input_tensor, scale_params = self.preprocess(frame)
            
            # 2. Infer
            raw_outputs = self.infer(input_tensor)
            
            # 3. Decode & Scale back
            boxes, scores, kps = self.decode(raw_outputs, scale_params)
            
            if len(boxes) == 0:
                return []
                
            # 4. NMS
            keep_indices = self.apply_nms(boxes, scores)
            
            # 5. Format Output
            detections = []
            for idx in keep_indices:
                pt_kps = kps[idx].reshape(-1, 2)
                landmarks_list = [[float(pt[0]), float(pt[1])] for pt in pt_kps]
                
                det = {
                    "bbox": [float(boxes[idx][0]), float(boxes[idx][1]), 
                             float(boxes[idx][2]), float(boxes[idx][3])],
                    "confidence": float(scores[idx]),
                    "landmarks": landmarks_list,
                    "name": "Unknown"  # Placeholder until recognizer is attached
                }
                detections.append(det)
                
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def draw(self, frame, detections):
        """Exposes visualization utility natively."""
        return draw_detections(frame, detections)

    def close(self):
        """Closes all Hailo hardware contexts."""
        self.exit_stack.close()

    def __del__(self):
        self.close()
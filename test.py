import cv2
import numpy as np
import threading
import time
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType

class HailoSCRFDRTSPWrapper:
    def __init__(self, hef_path, conf_threshold=0.5, nms_threshold=0.4):
        self.hef_path = hef_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load the HEF model
        self.hef = HEF(self.hef_path)
        
        # SCRFD anchor specifications
        self.strides = [8, 16, 32]
        self.anchor_num = 2
        
        # Threading controls for RTSP streaming
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def _generate_anchors(self, img_shape):
        anchors_list = []
        for stride in self.strides:
            feat_h = int(np.ceil(img_shape[0] / stride))
            feat_w = int(np.ceil(img_shape[1] / stride))
            
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            
            anchor_centers = np.stack([shift_x, shift_y], axis=-1).reshape(-1, 2)
            anchor_centers = np.repeat(anchor_centers, self.anchor_num, axis=0)
            anchors_list.append(anchor_centers)
        return anchors_list

    def post_process(self, raw_outputs, img_shape):
        all_boxes, all_scores = [], []
        anchors = self._generate_anchors(img_shape)
        
        for i, stride in enumerate(self.strides):
            # Dynamic matching based on output layer names
            cls_out = raw_outputs[f'cls_{stride}'].astype(np.float32)
            bbox_out = raw_outputs[f'bbox_{stride}'].astype(np.float32)
            
            scores = cls_out.reshape(-1, 1)
            bbox_offsets = bbox_out.reshape(-1, 4)
            stride_anchors = anchors[i]
            
            valid_indices = np.where(scores > self.conf_threshold)[0]
            if len(valid_indices) == 0:
                continue
                
            scores = scores[valid_indices]
            bbox_offsets = bbox_offsets[valid_indices]
            stride_anchors = stride_anchors[valid_indices]
            
            x1 = stride_anchors[:, 0] - bbox_offsets[:, 0] * stride
            y1 = stride_anchors[:, 1] - bbox_offsets[:, 1] * stride
            x2 = stride_anchors[:, 0] + bbox_offsets[:, 2] * stride
            y2 = stride_anchors[:, 1] + bbox_offsets[:, 3] * stride
            
            all_boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            all_scores.append(scores)
            
        if not all_boxes:
            return np.array([]), np.array([])
            
        all_boxes = np.vstack(all_boxes)
        all_scores = np.vstack(all_scores).flatten()
        
        indices = cv2.dnn.NMSBoxes(
            bboxes=all_boxes.tolist(), 
            scores=all_scores.tolist(), 
            score_threshold=self.conf_threshold, 
            nms_threshold=self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return all_boxes[indices], all_scores[indices]
        return np.array([]), np.array([])

    def _rtsp_reader(self, rtsp_url):
        """Background thread designed to continuously pull the latest frames from the network."""
        cap = cv2.VideoCapture(rtsp_url)
        
        # Optional optimization for low-latency network streams
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to fetch frame from RTSP stream. Retrying...")
                time.sleep(0.1)
                continue
            
            with self.lock:
                self.frame = frame.copy()
        
        cap.release()

    def start_stream_inference(self, rtsp_url):
        self.running = True
        
        # Start background thread for RTSP video capture ingestion
        reader_thread = threading.Thread(target=self._rtsp_reader, args=(rtsp_url,), daemon=True)
        reader_thread.start()
        
        print("Connecting to RTSP Stream...")
        # Wait until the stream establishes and buffers the first frame
        while self.frame is None:
            time.sleep(0.1)
            
        input_w, input_h = 640, 640
        
        # Open Hailo VDevice hardware framework
        with VDevice() as target_device:
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            network_group = target_device.configure(self.hef, configure_params)[0]
            
            input_vstream_params = InputVStreamParams.make_from_network_group(network_group, format_type=FormatType.FLOAT32)
            output_vstream_params = OutputVStreamParams.make_from_network_group(network_group, format_type=FormatType.FLOAT32)
            
            with HailoStreamInterface.create_input_vstreams(network_group, input_vstream_params) as input_vstreams, \
                 HailoStreamInterface.create_output_vstreams(network_group, output_vstream_params) as output_vstreams:
                
                with network_group.activate_context():
                    print("Processing Stream. Press 'q' to quit.")
                    while self.running:
                        # Safely grab the most recent frame fetched by the reader thread
                        with self.lock:
                            current_frame = self.frame.copy()
                            
                        h_orig, w_orig = current_frame.shape[:2]
                        
                        # Preprocess frame
                        resized_img = cv2.resize(current_frame, (input_w, input_h))
                        input_data = np.expand_dims(resized_img, axis=0)
                        
                        # Write frame tensor to Hailo architecture
                        input_vstreams[0].send(input_data)
                        
                        # Collect model's raw multi-scale anchor predictions
                        raw_outputs = {}
                        for out_stream in output_vstreams:
                            layer_name = out_stream.name.split('/')[-1]
                            raw_outputs[layer_name] = out_stream.recv()
                            
                        # Run postprocess coordinate mapping logic
                        boxes, scores = self.post_process(raw_outputs, (input_h, input_w))
                        
                        # Map scaled coordinates back to raw RTSP aspect ratio size
                        scale_x = w_orig / input_w
                        scale_y = h_orig / input_h
                        
                        # Draw bounding boxes onto frame output rendering
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = box
                            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                            
                            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(current_frame, f"{score:.2f}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Render the output
                        cv2.imshow("Hailo Live RTSP Detection", current_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False
                            break
                            
        cv2.destroyAllWindows()

# --- Live Execution Example ---
if __name__ == "__main__":
    # Standard format: "rtsp://username:password@ip_address:port/h264_stream"
    RTSP_URL = "rtsp://admin:Tiandy%40123@192.168.1.103:554/Streaming/Channels/101"
    
    wrapper = HailoSCRFDRTSPWrapper(hef_path="./models/scrfd_500m.hef")
    wrapper.start_stream_inference(RTSP_URL)

import queue
from tool.object_detection_utils import ObjectDetectionUtils
from tool import utils
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
"""
hailo_service/hailo_wrapper.py
================================
High-level inference worker that wraps HailoAsyncInference for the two
model roles used in the Smart Locker System:

  • Face Detection   — SCRFD 500M  (multi-output dict, uint8 → post-processed)
  • Face Recognition — ArcFace MobileFaceNet  (single float32 embedding output)

Usage::

    from hailo_service.hailo_wrapper import HailoInferenceWorker

    detector   = HailoInferenceWorker.create_detector("models/scrfd_500m.hef")
    recognizer = HailoInferenceWorker.create_recognizer("models/arcface_mobilefacenet-2.hef")

    # Push a preprocessed frame and get a result (blocking, with timeout)
    detector.infer(original_frame, preprocessed_frame)   # non-blocking push
    result = detector.get_result(timeout=0.5)            # returns (orig, raw_output)
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from hailo_service.hailo_model_manager import HailoModelManager
from hailo_service.utils import HailoAsyncInference


class HailoInferenceWorker:
    """
    Manages one HEF model's async inference loop in a background thread.

    The worker owns:
      - An ``input_queue``  — callers push preprocessed frames here.
      - An ``output_queue`` — results appear here after inference.
      - A ``ThreadPoolExecutor`` running the Hailo inference loop.

    The caller never touches the executor directly; use ``infer()`` to
    submit frames and ``get_result()`` to retrieve outputs.

    Args:
        hef_path (str): Absolute path to the .hef model file.
        output_type (Optional[Dict[str, str]]): Per-layer output format map
            (e.g. ``{'arcface_mobilefacenet/fc1': 'FLOAT32'}``).
            Pass None to use the model's native output format (e.g. UINT8).
        batch_size (int): Inference batch size. Defaults to 1.
        input_queue_size (int): Maximum items in the input queue. Defaults to 4.
        output_queue_size (int): Maximum items in the output queue. Defaults to 4.
    """

    def __init__(
        self,
        hef_path: str,
        output_type: Optional[Dict[str, str]] = None,
        batch_size: int = 1,
        input_queue_size: int = 4,
        output_queue_size: int = 4,
    ) -> None:
        self._hef_path = hef_path
        self._stopped = False

        # Thread-safe queues
        self._input_queue: queue.Queue = queue.Queue(maxsize=input_queue_size)
        self._output_queue: queue.Queue = queue.Queue(maxsize=output_queue_size)

        # Retrieve the shared VDevice
        vdevice = HailoModelManager.get_device()

        # Build the Hailo async inference object
        self._hailo_inference = HailoAsyncInference(
            hef_path=hef_path,
            input_queue=self._input_queue,
            output_queue=self._output_queue,
            batch_size=batch_size,
            output_type=output_type,
            send_original_frame=True,
            vdevice=vdevice,
        )

        # Expose input shape so the caller can preprocess correctly
        h, w, _ = self._hailo_inference.get_input_shape()
        self.input_height: int = h
        self.input_width: int = w

        # Start the inference loop in a background thread
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hailo_{hef_path[-20:]}")
        self._future = self._executor.submit(self._hailo_inference.run)
        self._future.add_done_callback(self._on_inference_thread_exit)

        logger.info(
            f"HailoInferenceWorker ready: {hef_path} | "
            f"input shape: ({h}, {w})"
        )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def create_detector(cls, hef_path: str) -> "HailoInferenceWorker":
        """
        Create a worker pre-configured for SCRFD face detection.

        The SCRFD model outputs multiple UINT8 tensors that are post-processed
        by ``hailo_model.py`` (rescale_network_outputs + SCRFDPostProc).

        Args:
            hef_path (str): Path to the SCRFD .hef file.

        Returns:
            HailoInferenceWorker: Configured for detection.
        """
        logger.info(f"Creating SCRFD face detector from: {hef_path}")
        # SCRFD uses native UINT8 output — do NOT override output_type
        return cls(hef_path=hef_path, output_type=None)

    @classmethod
    def create_recognizer(cls, hef_path: str) -> "HailoInferenceWorker":
        """
        Create a worker pre-configured for ArcFace face recognition.

        The ArcFace model requires FLOAT32 output for the embedding layer so
        that cosine similarity comparisons are numerically stable.

        Args:
            hef_path (str): Path to the ArcFace .hef file.

        Returns:
            HailoInferenceWorker: Configured for recognition.
        """
        logger.info(f"Creating ArcFace recognizer from: {hef_path}")
        output_type = {"arcface_mobilefacenet/fc1": "FLOAT32"}
        return cls(hef_path=hef_path, output_type=output_type)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def infer(
        self,
        original_frame: np.ndarray,
        preprocessed_frame: np.ndarray,
    ) -> bool:
        """
        Submit a frame for inference (non-blocking).

        If the input queue is full (backpressure), the oldest pending item is
        discarded before the new one is enqueued so the queue never blocks the
        caller's thread.

        Args:
            original_frame (np.ndarray): The raw camera frame (for passthrough).
            preprocessed_frame (np.ndarray): The model-ready frame (resized,
                colour-converted, etc.).

        Returns:
            bool: True if enqueued successfully, False if the worker is stopped.
        """
        if self._stopped:
            return False

        item = ([original_frame], [preprocessed_frame])

        try:
            self._input_queue.put_nowait(item)
        except queue.Full:
            # Drop oldest item to make room (keeps latency low)
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait(item)
            except queue.Full:
                logger.warning("HailoInferenceWorker input queue full — frame dropped.")
                return False

        return True

    def get_result(
        self,
        timeout: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, Any]]:
        """
        Retrieve the next inference result (blocking with timeout).

        Args:
            timeout (float): Maximum seconds to wait. Defaults to 0.5 s.

        Returns:
            Optional[Tuple[np.ndarray, Any]]: ``(original_frame, raw_output)``
                where raw_output is a numpy array (single output) or dict
                (multiple outputs).  Returns None on timeout.
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_output_queue(self) -> None:
        """Discard all pending items in the output queue."""
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def stop(self, wait: bool = True) -> None:
        """
        Gracefully stop the inference worker.

        Sends a sentinel None to the input_queue to signal the run loop
        to exit, then optionally waits for the background thread to finish.

        Args:
            wait (bool): If True (default), block until the thread exits.
        """
        if self._stopped:
            return

        self._stopped = True
        logger.info(f"Stopping HailoInferenceWorker: {self._hef_path}")

        # Unblock the run loop
        try:
            self._input_queue.put(None, timeout=2.0)
        except queue.Full:
            logger.warning("Input queue full during shutdown — forcing sentinel.")
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            self._input_queue.put(None)

        if wait:
            self._executor.shutdown(wait=True)

    def _on_inference_thread_exit(self, future) -> None:
        """Callback fired when the inference thread exits (normal or error)."""
        exc = future.exception()
        if exc is not None:
            logger.error(
                f"Hailo inference thread for '{self._hef_path}' "
                f"exited with exception: {exc}"
            )
        else:
            logger.info(f"Hailo inference thread exited cleanly: {self._hef_path}")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "HailoInferenceWorker":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

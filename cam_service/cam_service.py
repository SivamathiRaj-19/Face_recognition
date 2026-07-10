
from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class RTSPCameraService:
    def __init__(
        self,
        rtsp_url: str,
        frame_size: Tuple[int, int] = (640, 640),
        queue_maxsize: int = 4,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
    ) -> None:
        self._url = rtsp_url
        self._frame_size = frame_size
        self._reconnect_base = reconnect_base_delay
        self._reconnect_max = reconnect_max_delay

        # Leaky frame queue: consumer always gets the freshest frame
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend_name: str = "unknown"
        self._frames_captured: int = 0
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "RTSPCameraService":
        if self._thread is not None and self._thread.is_alive():
            logger.warning("RTSPCameraService is already running.")
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CaptureThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"RTSPCameraService started → {self._url}")
        return self

    def stop(self) -> None:
        """Signal the capture thread to exit and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("RTSPCameraService stopped.")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent frame without blocking.

        Returns:
            np.ndarray: BGR frame resized to ``frame_size``, or None if the
                queue is empty (no frame available yet).
        """
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def is_connected(self) -> bool:
        """True if the capture thread currently has an open camera handle."""
        return self._connected

    @property
    def backend(self) -> str:
        """Name of the currently active OpenCV capture backend."""
        return self._backend_name

    @property
    def frames_captured(self) -> int:
        """Total frames successfully captured since start."""
        return self._frames_captured

    # ------------------------------------------------------------------
    # Backend fallback chain
    # ------------------------------------------------------------------

    def _try_open_gstreamer(self) -> Optional[cv2.VideoCapture]:
        """
        Attempt to open the RTSP stream via GStreamer.

        Uses a GStreamer pipeline string that leverages hardware-accelerated
        H.264 decode (via ``omxh264dec`` on Raspberry Pi or ``avdec_h264``
        as fallback) and pre-scales to the target resolution.

        Returns:
            cv2.VideoCapture if successful, else None.
        """
        w, h = self._frame_size
        # Try hardware decoder first (RPi / Jetson), then software
        pipelines = [
            # RPi / V4L2 H.264 hardware decode
            (
                f"rtspsrc location={self._url} latency=100 protocols=tcp ! "
                "rtph264depay ! h264parse ! "
                "v4l2h264dec ! "
                f"videoscale ! video/x-raw,format=BGR,width={w},height={h} ! "
                "appsink drop=true max-buffers=2 sync=false"
            ),
            # Software decode (any platform with GStreamer + libav)
            (
                f"rtspsrc location={self._url} latency=100 protocols=tcp ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! videoscale ! "
                f"video/x-raw,format=BGR,width={w},height={h} ! "
                "appsink drop=true max-buffers=2 sync=false"
            ),
        ]

        for pipeline in pipelines:
            try:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        logger.success("Camera backend: GStreamer (hardware pipeline)")
                        return cap
                cap.release()
            except Exception as exc:
                logger.debug(f"GStreamer pipeline attempt failed: {exc}")

        return None

    def _try_open_ffmpeg(self) -> Optional[cv2.VideoCapture]:
        """
        Attempt to open the RTSP stream via FFmpeg.

        Sets buffer size and flags for low-latency TCP transport.

        Returns:
            cv2.VideoCapture if successful, else None.
        """
        try:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            # Low-latency FFmpeg tuning
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)

            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    logger.success("Camera backend: FFmpeg")
                    return cap
            cap.release()
        except Exception as exc:
            logger.debug(f"FFmpeg backend failed: {exc}")

        return None

    def _try_open_opencv(self) -> Optional[cv2.VideoCapture]:
        """
        Open the RTSP stream using OpenCV's default backend.

        This is the last-resort fallback — always works if OpenCV was
        built with any network stream support.

        Returns:
            cv2.VideoCapture if successful, else None.
        """
        try:
            cap = cv2.VideoCapture(self._url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    logger.success("Camera backend: OpenCV (default)")
                    return cap
            cap.release()
        except Exception as exc:
            logger.debug(f"OpenCV backend failed: {exc}")

        return None

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Try all backends in priority order and return the first working one.

        Priority: GStreamer → FFmpeg → OpenCV

        Returns:
            cv2.VideoCapture on success, None if all backends fail.
        """
        logger.info(f"Connecting to RTSP stream: {self._url}")

        cap = self._try_open_gstreamer()
        if cap is not None:
            self._backend_name = "gstreamer"
            return cap

        logger.warning("GStreamer unavailable — trying FFmpeg…")
        cap = self._try_open_ffmpeg()
        if cap is not None:
            self._backend_name = "ffmpeg"
            return cap

        logger.warning("FFmpeg unavailable — falling back to OpenCV…")
        cap = self._try_open_opencv()
        if cap is not None:
            self._backend_name = "opencv"
            return cap

        logger.error("All camera backends failed for URL: {}", self._url)
        return None

    # ------------------------------------------------------------------
    # Capture loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Main capture loop executed in the background thread.

        Reads frames, resizes them, and puts them into the frame queue.
        On failure, releases the capture handle and re-connects with
        exponential backoff up to ``reconnect_max_delay`` seconds.
        """
        delay = self._reconnect_base

        while not self._stop_event.is_set():
            cap = self._open_camera()

            if cap is None:
                self._connected = False
                logger.warning(
                    f"Reconnect in {delay:.1f}s (all backends failed)…"
                )
                self._stop_event.wait(delay)
                delay = min(delay * 2, self._reconnect_max)
                continue

            # Successful connection — reset backoff
            self._connected = True
            delay = self._reconnect_base
            consecutive_failures = 0
            logger.info(f"Stream connected via {self._backend_name}.")

            while not self._stop_event.is_set():
                ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        logger.warning(
                            f"⚠️  Stream read failed {consecutive_failures} "
                            "consecutive times — reconnecting…"
                        )
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                w, h = self._frame_size
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                # Non-blocking put: drop oldest frame if queue is full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self._frame_queue.put_nowait(frame)
                    self._frames_captured += 1
                except queue.Full:
                    pass  # Another producer slipped in — silently skip

            self._connected = False
            cap.release()
            logger.info("Camera handle released.")

            if not self._stop_event.is_set():
                logger.info(f"Waiting {delay:.1f}s before reconnect…")
                self._stop_event.wait(delay)
                delay = min(delay * 2, self._reconnect_max)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "RTSPCameraService":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return (
            f"RTSPCameraService(url={self._url!r}, "
            f"backend={self._backend_name!r}, "
            f"status={status}, "
            f"frames={self._frames_captured})"
        )

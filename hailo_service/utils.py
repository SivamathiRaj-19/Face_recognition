"""
hailo_service/utils.py
======================
Low-level Hailo runtime helpers for the Smart Locker System.

Key changes vs original:
  - VDevice is NO LONGER created at module level (moved to hailo_model_manager).
    HailoAsyncInference now accepts an injected `vdevice` argument.
  - Added cosine_similarity() for embedding matching.
  - Added preprocess_face_crop() for ArcFace input normalization.
  - All public functions carry full docstrings.
"""

from __future__ import annotations

import queue
from functools import partial
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from loguru import logger

from hailo_platform import FormatType, HEF

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".png", ".bmp", ".jpeg")


# ---------------------------------------------------------------------------
# HailoAsyncInference
# ---------------------------------------------------------------------------


class HailoAsyncInference:
    """
    Runs a single HEF model asynchronously inside a dedicated inference loop.

    The caller feeds pre-processed frames into `input_queue` and reads
    results from `output_queue`.  The run loop is designed to be executed
    in its own thread (e.g. via ``ThreadPoolExecutor.submit(instance.run)``).

    Args:
        hef_path (str): Path to the .hef model file.
        input_queue (queue.Queue): Source of (original_batch, preprocessed_batch)
            tuples when send_original_frame=True, or just preprocessed_batch lists.
        output_queue (queue.Queue): Destination for (original_frame, result) tuples.
        batch_size (int): Inference batch size. Defaults to 1.
        input_type (Optional[str]): FormatType name for the input stream
            (e.g. 'UINT8'). None uses the model default.
        output_type (Optional[Dict[str, str]]): Mapping of output layer name →
            FormatType name (e.g. {'layer': 'FLOAT32'}). None uses model default.
        send_original_frame (bool): When True, input_queue items are
            (original_batch, preprocessed_batch) tuples so the original frame
            is forwarded alongside the inference result.
        vdevice: Shared VDevice instance provided by HailoModelManager.
            Must not be None.
    """

    def __init__(
        self,
        hef_path: str,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        batch_size: int = 1,
        input_type: Optional[str] = None,
        output_type: Optional[Dict[str, str]] = None,
        send_original_frame: bool = False,
        vdevice=None,
    ) -> None:
        if vdevice is None:
            raise ValueError(
                "vdevice must be provided. Use HailoModelManager.get_device()."
            )

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.target = vdevice
        self.send_original_frame = send_original_frame
        self.output_type = output_type

        self.hef = HEF(hef_path)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

        # Resolve output names once so the callback and binding factory
        # always use the same list without touching private attributes.
        self._output_infos = self.hef.get_output_vstream_infos()
        self._output_names = [info.name for info in self._output_infos]
        self._input_name = self.hef.get_input_vstream_infos()[0].name

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _set_input_type(self, input_type: str) -> None:
        """Set input stream format type for all model inputs."""
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type_dict: Dict[str, str]) -> None:
        """Set output stream format type per named output layer."""
        for output_name, fmt in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, fmt)
            )

    # ------------------------------------------------------------------
    # Async callback
    # ------------------------------------------------------------------

    def callback(
        self,
        completion_info,
        bindings_list: list,
        input_batch: list,
    ) -> None:
        """
        Called by the Hailo runtime when an async inference job completes.

        Places (original_frame, result) tuples into output_queue.
        Single-output models return a raw numpy array; multi-output models
        return a dict of {output_name: np.ndarray}.

        Args:
            completion_info: Runtime completion metadata.
            bindings_list (list): Binding objects holding output buffers.
            input_batch (list): The original or preprocessed frames that
                correspond 1-to-1 with bindings_list.
        """
        if completion_info.exception:
            logger.error(f"Inference callback error: {completion_info.exception}")
            return

        for i, bindings in enumerate(bindings_list):
            if len(self._output_names) == 1:
                result = bindings.output(self._output_names[0]).get_buffer()
            else:
                result = {
                    name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                    for name in self._output_names
                }
            self.output_queue.put((input_batch[i], result))

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------

    def get_vstream_info(self) -> Tuple[list, list]:
        """Return (input_vstream_infos, output_vstream_infos) from the HEF."""
        return (
            self.hef.get_input_vstream_infos(),
            self.hef.get_output_vstream_infos(),
        )

    def get_hef(self) -> HEF:
        """Return the loaded HEF object."""
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the shape of the model's first input layer (H, W, C)."""
        return self.hef.get_input_vstream_infos()[0].shape

    # ------------------------------------------------------------------
    # Inference loop (run in a dedicated thread)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Blocking inference loop.  Reads from input_queue, submits async jobs,
        and forwards results via callback → output_queue.

        Send ``None`` to input_queue to stop the loop cleanly.
        """
        with self.infer_model.configure() as configured_infer_model:
            last_job = None
            while True:
                try:
                    batch_data = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if batch_data is None:
                    break  # Sentinel: graceful shutdown

                if self.send_original_frame:
                    original_batch, preprocessed_batch = batch_data
                else:
                    preprocessed_batch = batch_data
                    original_batch = preprocessed_batch

                try:
                    bindings_list = []
                    for frame in preprocessed_batch:
                        buf = np.ascontiguousarray(frame)
                        bindings = self._create_bindings(
                            configured_infer_model, input_buf=buf
                        )
                        bindings_list.append(bindings)

                    configured_infer_model.wait_for_async_ready(timeout_ms=10_000)
                    last_job = configured_infer_model.run_async(
                        bindings_list,
                        partial(
                            self.callback,
                            input_batch=original_batch,
                            bindings_list=bindings_list,
                        ),
                    )
                except Exception as exc:
                    logger.error(
                        f"Hailo inference step failed (frame skipped): {exc}"
                    )
                    # Do NOT re-raise — keep the thread alive for next frames

            if last_job is not None:
                last_job.wait(10_000)

    # ------------------------------------------------------------------
    # Internal binding factory
    # ------------------------------------------------------------------

    def _get_output_type_str(self, output_info) -> str:
        """Resolve numpy dtype string for an output layer."""
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        return self.output_type[output_info.name].lower()

    def _create_bindings(self, configured_infer_model, input_buf: np.ndarray) -> object:
        """
        Create a fully-populated bindings object (input + all outputs).

        The correct Hailo async API pattern is:
          1. ``create_bindings()``  — no pre-filled buffers
          2. ``bindings.input(name).set_buffer(input_np)``
          3. ``bindings.output(name).set_buffer(output_np)`` for each output

        Passing ``output_buffers=`` to ``create_bindings`` leaves the input
        binding slot uninitialized (size 0), which makes the subsequent
        ``set_buffer`` call fail with HAILO_INVALID_OPERATION.
        """
        bindings = configured_infer_model.create_bindings()

        # --- Input ---
        bindings.input(self._input_name).set_buffer(input_buf)

        # --- Outputs ---
        for info in self._output_infos:
            dtype_str = self._get_output_type_str(info)
            out_buf = np.empty(
                self.infer_model.output(info.name).shape,
                dtype=getattr(np, dtype_str),
            )
            bindings.output(info.name).set_buffer(out_buf)

        return bindings


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two 1-D feature vectors.

    Args:
        a (np.ndarray): First embedding vector (any finite length).
        b (np.ndarray): Second embedding vector (same length as ``a``).

    Returns:
        float: Similarity score in the range [-1.0, 1.0].
               Higher values indicate a closer match.
    """
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def preprocess_face_crop(
    crop: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Resize and normalise a face crop for ArcFace/MobileFaceNet input.

    Steps:
        1. Resize to ``target_size`` (width, height) using bicubic interpolation.
        2. Convert BGR → RGB.
        3. Normalise to [0, 1] and apply ImageNet-style mean/std.

    Args:
        crop (np.ndarray): BGR face crop array (H×W×3, uint8).
        target_size (Tuple[int, int]): (width, height) for the output.
            Defaults to (112, 112) as expected by ArcFace MobileFaceNet.

    Returns:
        np.ndarray: float32 array of shape (H, W, 3) ready for inference.
    """
    import cv2  # local import to avoid import-time dependency on CI

    if crop is None or crop.size == 0:
        raise ValueError("preprocess_face_crop received an empty crop.")

    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalise: (pixel / 255 - mean) / std  — ArcFace standard
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalised = (rgb.astype(np.float32) / 255.0 - mean) / std

    return normalised


# ---------------------------------------------------------------------------
# Image I/O utilities
# ---------------------------------------------------------------------------


def load_images_opencv(images_path: str) -> List[np.ndarray]:
    """
    Load image(s) from a file path or directory using OpenCV.

    Args:
        images_path (str): Path to a single image file or a directory.

    Returns:
        List[np.ndarray]: BGR images as NumPy arrays.
    """
    import cv2

    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [cv2.imread(str(path))]
    if path.is_dir():
        return [
            cv2.imread(str(img))
            for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []


def load_input_images(images_path: str) -> list:
    """
    Load image(s) from a file path or directory using PIL.

    Args:
        images_path (str): Path to a single image file or a directory.

    Returns:
        List[PIL.Image.Image]: PIL Image objects.
    """
    from PIL import Image

    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Image.open(path)]
    if path.is_dir():
        return [
            Image.open(img)
            for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []


def validate_images(images: List[np.ndarray], batch_size: int) -> None:
    """
    Validate that a list of images is non-empty and evenly divisible by batch_size.

    Args:
        images (List[np.ndarray]): List of loaded images.
        batch_size (int): Intended inference batch size.

    Raises:
        ValueError: If the list is empty or not divisible by batch_size.
    """
    if not images:
        raise ValueError("No valid images found in the specified path.")
    if len(images) % batch_size != 0:
        raise ValueError(
            f"Image count ({len(images)}) must be divisible by "
            f"batch_size ({batch_size}) without remainder."
        )


def divide_list_to_batches(
    images_list: List[np.ndarray],
    batch_size: int,
) -> Generator[List[np.ndarray], None, None]:
    """
    Yield successive batches from a flat list of images.

    Args:
        images_list (List[np.ndarray]): Flat list of images.
        batch_size (int): Number of images per batch.

    Yields:
        List[np.ndarray]: Successive slices of length ``batch_size``.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i: i + batch_size]

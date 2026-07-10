"""
models/model_config.py
======================
Centralized configuration for all model assets, inference parameters,
and SCRFD anchor definitions used by the Smart Locker System.

All paths are resolved relative to this file's directory so the project
can be run from any working directory.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base path resolution
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _MODELS_DIR.parent.resolve()


def get_model_path(filename: str) -> str:
    """
    Resolve a model asset filename to its absolute path.

    Args:
        filename (str): Bare filename (e.g. 'scrfd_500m.hef').

    Returns:
        str: Absolute path to the model file.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    path = _MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model asset not found: {path}. "
            f"Ensure the .hef file is present in the 'models/' directory."
        )
    return str(path)


# ---------------------------------------------------------------------------
# Model HEF Paths
# ---------------------------------------------------------------------------

DETECTOR_HEF = get_model_path("scrfd_500m.hef")
RECOGNIZER_HEF = get_model_path("arcface_mobilefacenet-2.hef")

# Shared-object postprocessing libraries (Hailo plugin .so files)
DETECTOR_SO = str(_MODELS_DIR / "libyolo_hailortpp_postprocess.so")
RECOGNIZER_SO = str(_MODELS_DIR / "libface_recognition_post.so")

# ---------------------------------------------------------------------------
# Face Database
# ---------------------------------------------------------------------------

FACE_DB_PATH = str(_ROOT_DIR / "face_db.npy")

# ---------------------------------------------------------------------------
# Camera Configuration
# ---------------------------------------------------------------------------

RTSP_URL = "rtsp://admin:Triton123@192.168.1.103/stream1"

# Target resolution for all frames entering the inference pipeline
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

# ---------------------------------------------------------------------------
# Detector (SCRFD 500M) Parameters
# ---------------------------------------------------------------------------

DETECTOR_SCORE_THRESHOLD = 0.60
DETECTOR_NMS_IOU_THRESHOLD = 0.40

# SCRFD multi-scale anchor configuration.
# 'steps'     : stride values for each branch (8, 16, 32)
# 'min_sizes' : anchor sizes per stride
SCRFD_ANCHORS = {
    "steps": [8, 16, 32],
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
}

# Output layer names used by rescale_network_outputs() in hailo_model.py
SCRFD_BOX_LAYERS = [
    "scrfd_500m/conv27",
    "scrfd_500m/conv33",
    "scrfd_500m/conv39",
]
SCRFD_CLASS_LAYERS = [
    "scrfd_500m/conv26",
    "scrfd_500m/conv32",
    "scrfd_500m/conv38",
]
SCRFD_LANDMARK_LAYERS = [
    "scrfd_500m/conv25",
    "scrfd_500m/conv34",
    "scrfd_500m/conv40",
]

# SCRFD fixed-point quantization parameters
SCRFD_BOX_DOWNSCALE = 32
SCRFD_LANDMARK_ZERO_POINT = 113
SCRFD_LANDMARK_SCALE = 29

# ---------------------------------------------------------------------------
# Recognizer (ArcFace MobileFaceNet) Parameters
# ---------------------------------------------------------------------------

# Output layer name → format type for HailoAsyncInference output_type dict
RECOGNIZER_OUTPUT_TYPE = {"arcface_mobilefacenet/fc1": "FLOAT32"}

# Input face crop size expected by ArcFace model
ARCFACE_INPUT_SIZE = (112, 112)

# Cosine similarity threshold: scores above this value are considered a match
RECOGNITION_COSINE_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Relay / Modbus TCP Configuration
# ---------------------------------------------------------------------------

RELAY_HOST = "192.168.1.200"
RELAY_PORT = 502
RELAY_CHANNELS = list(range(1, 9))   # Valid channels: 1–8
RELAY_DEFAULT_CHANNEL = 1            # Channel opened on successful recognition

# Minimum seconds between consecutive relay triggers (per-channel cooldown)
RELAY_COOLDOWN_SECONDS = 5.0

# Socket timeout for Modbus TCP connection
RELAY_SOCKET_TIMEOUT = 2.0

# ---------------------------------------------------------------------------
# Inference Queue Sizes
# ---------------------------------------------------------------------------

FRAME_QUEUE_MAXSIZE = 4       # Small → drops stale frames, keeps latency low
RELAY_QUEUE_MAXSIZE = 8       # Buffered relay commands

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = str(_ROOT_DIR / "smart_locker.log")
LOG_ROTATION = "10 MB"
LOG_RETENTION = "7 days"

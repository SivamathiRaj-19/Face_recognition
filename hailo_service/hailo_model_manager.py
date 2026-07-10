from __future__ import annotations

import threading
from typing import Optional

from loguru import logger

try:
    from hailo_platform import HailoSchedulingAlgorithm, VDevice
    _HAILO_AVAILABLE = True
except ImportError:
    _HAILO_AVAILABLE = False
    logger.warning(
        "hailo_platform not found — running in MOCK mode. "
        "All inference calls will return empty results."
    )


class HailoModelManager:
    _instance: Optional[object] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_device(cls) -> object:
        if cls._initialized:
            return cls._instance

        with cls._lock:
            # Double-checked locking
            if cls._initialized:
                return cls._instance

            if not _HAILO_AVAILABLE:
                raise RuntimeError(
                    "hailo_platform is not installed. "
                    "Install it with: pip install hailo-platform"
                )

            logger.info("Initializing Hailo VDevice (ROUND_ROBIN scheduler)…")
            try:
                params = VDevice.create_params()
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                cls._instance = VDevice(params)
                cls._initialized = True
                logger.success("Hailo VDevice initialized successfully.")
            except Exception as exc:
                logger.critical(
                    f"Failed to open Hailo device: {exc}. "
                    "Check that the Hailo hardware is connected and the "
                    "HailoRT driver is loaded."
                )
                raise RuntimeError(f"Hailo device initialization failed: {exc}") from exc

        return cls._instance

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance = None
                    cls._initialized = False
                    logger.info("Hailo VDevice released.")
                except Exception as exc:
                    logger.warning(f"Error releasing Hailo VDevice: {exc}")

    @classmethod
    def is_ready(cls) -> bool:
        """Return True if the VDevice has been successfully initialized."""
        return cls._initialized

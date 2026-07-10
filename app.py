from __future__ import annotations

import argparse
import math
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Path resolution — ensure repo root is on sys.path regardless of CWD
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from models.model_config import (
    DETECTOR_HEF,
    RECOGNIZER_HEF,
    RECOGNIZER_OUTPUT_TYPE,
    SCRFD_ANCHORS,
    DETECTOR_SCORE_THRESHOLD,
    DETECTOR_NMS_IOU_THRESHOLD,
    ARCFACE_INPUT_SIZE,
    RECOGNITION_COSINE_THRESHOLD,
    RELAY_DEFAULT_CHANNEL,
    RELAY_COOLDOWN_SECONDS,
    FRAME_SIZE,
    FRAME_QUEUE_MAXSIZE,
    RELAY_QUEUE_MAXSIZE,
    FACE_DB_PATH,
    RTSP_URL,
    LOG_LEVEL,
    LOG_FILE,
    LOG_ROTATION,
    LOG_RETENTION,
)
from cam_service.cam_service import RTSPCameraService
from face_db.face_db_manager import FaceDBManager
from hailo_service.hailo_model_manager import HailoModelManager
from hailo_service.hailo_model import HAILO, SCRFD_HAILO
from hailo_service.utils import preprocess_face_crop
from relay_service.relay_service import RelayService

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(level: str = LOG_LEVEL) -> None:
    """Configure loguru for console + rotating file output."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "— <level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        LOG_FILE,
        level="DEBUG",
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} — {message}",
    )


# ---------------------------------------------------------------------------
# Face alignment helper
# ---------------------------------------------------------------------------


def _rotate_image(image: np.ndarray, angle: float, center: Tuple[int, int]) -> np.ndarray:
    """Rotate image around a given centre point."""
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def align_and_crop(
    frame: np.ndarray,
    box: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:

    x1, y1, x2, y2 = map(int, box)
    left_eye_x, left_eye_y = int(landmarks[0]), int(landmarks[1])
    right_eye_x, right_eye_y = int(landmarks[2]), int(landmarks[3])

    angle = 0.0
    if left_eye_x != right_eye_x or left_eye_y != right_eye_y:
        angle_rad = math.atan2(
            right_eye_y - left_eye_y, right_eye_x - left_eye_x
        )
        angle = math.degrees(angle_rad)

    cx = x1 + abs(x1 - x2) // 2
    cy = y1 + abs(y1 - y2) // 2
    rotated = _rotate_image(frame, angle, (cx, cy))

    h, w = rotated.shape[:2]
    crop = rotated[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    return crop


# ---------------------------------------------------------------------------
# Thread 2 — AI Inference Thread
# ---------------------------------------------------------------------------


def ai_inference_thread(
    face_detector: SCRFD_HAILO,
    feature_model: HAILO,
    face_db: FaceDBManager,
    frame_queue: queue.Queue,
    relay_queue: queue.Queue,
    stop_event: threading.Event,
    display_queue: Optional[queue.Queue] = None,
) -> None:

    logger.info("AI inference thread started.")
    frames_processed = 0
    last_stats_time = time.monotonic()

    while not stop_event.is_set():
        # Pull next frame (timeout avoids blocking stop_event check)
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        display_frame = frame.copy()
        frames_processed += 1

        # ── Stage 1: Face Detection ─────────────────────────────────────
        try:
            detections = face_detector(frame)
        except Exception as exc:
            logger.error(f"Face detector error: {exc}")
            continue

        if detections is None or detections.get("num_detections", 0) == 0:
            if display_queue is not None:
                _push_display(display_queue, display_frame)
            continue

        boxes = detections["detection_boxes"]      # shape (N, 4) pixel coords
        scores = detections["detection_scores"]    # shape (N,)
        landmarks = detections["face_landmarks"]   # shape (N, 10)
        num_det = detections["num_detections"]

        # ── Stage 2: Recognition per detected face ──────────────────────
        for i in range(num_det):
            score = float(scores[i])
            if score < DETECTOR_SCORE_THRESHOLD:
                continue

            box = boxes[i]
            landm = landmarks[i]

            # Draw detection box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 50), 2)
            for j in range(5):
                lx, ly = int(landm[j * 2]), int(landm[j * 2 + 1])
                cv2.circle(display_frame, (lx, ly), 3, (0, 80, 255), -1)

            # Align & crop face
            face_crop = align_and_crop(frame, box, landm)
            if face_crop is None or face_crop.size == 0:
                logger.debug("Empty face crop — skipping recognition.")
                continue

            # Prepare ArcFace input
            try:
                face_input = preprocess_face_crop(face_crop, target_size=ARCFACE_INPUT_SIZE)
            except Exception as exc:
                logger.warning(f"Face crop preprocessing failed: {exc}")
                continue

            # Run embedding extraction
            try:
                embedding_result = feature_model([face_crop])
            except Exception as exc:
                logger.error(f"ArcFace inference error: {exc}")
                continue

            if embedding_result is None:
                logger.debug("No embedding returned — skipping.")
                continue

            embedding = np.array(embedding_result).flatten()

            # Match against database
            matched_name, sim_score = face_db.find_match(
                embedding, threshold=RECOGNITION_COSINE_THRESHOLD
            )

            if matched_name is not None:
                label = f"{matched_name} ({sim_score:.2f})"
                color = (0, 255, 120)
                logger.info(
                    f"✅ Recognised: '{matched_name}' "
                    f"(cosine={sim_score:.3f}) — triggering locker."
                )

                # Enqueue relay trigger (non-blocking)
                cmd = (RELAY_DEFAULT_CHANNEL, matched_name)
                try:
                    relay_queue.put_nowait(cmd)
                except queue.Full:
                    logger.warning("Relay queue full — trigger dropped.")
            else:
                label = f"Unknown ({sim_score:.2f})"
                color = (0, 80, 255)
                logger.debug(f"No match (best cosine={sim_score:.3f}).")

            # Annotate display frame
            cv2.putText(
                display_frame, label, (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        if display_queue is not None:
            _push_display(display_queue, display_frame)

        # Log throughput every 30 s
        now = time.monotonic()
        if now - last_stats_time >= 30.0:
            fps = frames_processed / (now - last_stats_time + 1e-9)
            logger.info(f"AI thread: {frames_processed} frames, ~{fps:.1f} fps")
            frames_processed = 0
            last_stats_time = now

    logger.info("AI inference thread exiting.")
    face_detector.stop()
    feature_model.stop()


# ---------------------------------------------------------------------------
# Thread 3 — Relay Control Thread
# ---------------------------------------------------------------------------


def relay_control_thread(
    relay_service: RelayService,
    relay_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:

    logger.info("Relay control thread started.")

    while not stop_event.is_set():
        try:
            item = relay_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        channel, person_name = item
        relay_service.trigger(channel=channel, person_name=person_name)
        relay_queue.task_done()

    logger.info("Relay control thread exiting.")


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------


def _push_display(display_queue: queue.Queue, frame: np.ndarray) -> None:
    """Non-blocking push to display queue, drops old frame if full."""
    if display_queue.full():
        try:
            display_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        display_queue.put_nowait(frame)
    except queue.Full:
        pass


# ---------------------------------------------------------------------------
# Registration mode
# ---------------------------------------------------------------------------


def registration_mode(
    face_detector: SCRFD_HAILO,
    feature_model: HAILO,
    face_db: FaceDBManager,
    cam: RTSPCameraService,
) -> None:

    logger.info("Entering REGISTRATION mode. Press 's' to save, 'q' to quit.")

    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        display_frame = frame.copy()
        detections = face_detector(frame)
        valid_faces = []

        if detections and detections.get("num_detections", 0) > 0:
            boxes = detections["detection_boxes"]
            scores = detections["detection_scores"]
            landmarks = detections["face_landmarks"]

            for i in range(detections["num_detections"]):
                if float(scores[i]) >= DETECTOR_SCORE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"{scores[i]:.2f}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
                    for j in range(5):
                        lx, ly = int(landmarks[i][j * 2]), int(landmarks[i][j * 2 + 1])
                        cv2.circle(display_frame, (lx, ly), 2, (0, 0, 255), -1)
                    valid_faces.append({"box": boxes[i], "landmarks": landmarks[i]})

        cv2.imshow("Smart Locker — Registration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            if not valid_faces:
                logger.warning("No face detected — cannot save.")
                continue

            # Pick largest face
            valid_faces.sort(
                key=lambda f: abs(f["box"][2] - f["box"][0]) * abs(f["box"][3] - f["box"][1]),
                reverse=True,
            )
            best = valid_faces[0]
            crop = align_and_crop(frame, best["box"], best["landmarks"])

            if crop.size == 0:
                logger.error("Crop failed — try again.")
                continue

            embedding = feature_model([crop])
            if embedding is None:
                logger.error("Embedding extraction failed.")
                continue

            name = input("\n[INPUT] Enter name for this face: ").strip()
            if not name:
                logger.warning("Name cannot be empty — registration aborted.")
                continue

            face_db.add(name, np.array(embedding).flatten(), overwrite=True)
            logger.success(f"Registered '{name}' successfully.")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _install_signal_handlers(stop_event: threading.Event) -> None:
    """Install SIGINT and SIGTERM handlers that set the stop_event."""

    def _handler(signum, frame):
        logger.warning(f"Signal {signum} received — initiating graceful shutdown…")
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Locker System — Hailo AI Face Recognition"
    )
    parser.add_argument(
        "--rtsp",
        default=RTSP_URL,
        help="RTSP camera URL (default: %(default)s)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live annotated video window (requires a display / desktop).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Run in face registration mode (interactive).",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RECOGNITION_COSINE_THRESHOLD,
        help="Cosine similarity threshold for face recognition (default: %(default)s).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("  Smart Locker System — Starting")
    logger.info("=" * 60)

    # ── Shared stop event ────────────────────────────────────────────────
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    # ── Load face database ───────────────────────────────────────────────
    face_db = FaceDBManager(FACE_DB_PATH)
    face_db.summary()

    # ── Initialize Hailo VDevice (singleton) ────────────────────────────
    logger.info("Initializing Hailo AI accelerator…")
    try:
        HailoModelManager.get_device()
    except RuntimeError as exc:
        logger.critical(f"Cannot start without Hailo device: {exc}")
        sys.exit(1)

    # ── Load AI models ───────────────────────────────────────────────────
    logger.info("Loading AI models…")
    try:
        face_detector = SCRFD_HAILO(
            DETECTOR_HEF,
            nms_iou_thresh=DETECTOR_NMS_IOU_THRESHOLD,
            score_threshold=DETECTOR_SCORE_THRESHOLD,
        )
        feature_model = HAILO(
            RECOGNIZER_HEF,
            task=1,
            output_type=RECOGNIZER_OUTPUT_TYPE,
        )
    except Exception as exc:
        logger.critical(f"Model loading failed: {exc}")
        HailoModelManager.release()
        sys.exit(1)

    # ── Initialize camera ────────────────────────────────────────────────
    logger.info(f"Connecting to camera: {args.rtsp}")
    cam = RTSPCameraService(args.rtsp, frame_size=FRAME_SIZE)
    cam.start()

    # ── Registration mode ────────────────────────────────────────────────
    if args.register:
        logger.info("Waiting for camera stream…")
        time.sleep(2.0)
        try:
            registration_mode(face_detector, feature_model, face_db, cam)
        finally:
            cam.stop()
            HailoModelManager.release()
        return

    # ── Thread-safe queues ───────────────────────────────────────────────
    frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    relay_queue: queue.Queue = queue.Queue(maxsize=RELAY_QUEUE_MAXSIZE)
    display_queue: Optional[queue.Queue] = queue.Queue(maxsize=2) if args.display else None

    # ── Initialize relay service ─────────────────────────────────────────
    relay_service = RelayService(cooldown_seconds=RELAY_COOLDOWN_SECONDS)
    relay_service.start()

    # ── Thread 1: Camera Capture ─────────────────────────────────────────
    def _camera_feeder():
        """Forward frames from RTSPCameraService into frame_queue."""
        logger.info("Camera feeder thread started.")
        while not stop_event.is_set():
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        logger.info("Camera feeder thread exiting.")

    t_camera = threading.Thread(
        target=_camera_feeder,
        name="CameraFeederThread",
        daemon=True,
    )

    # ── Thread 2: AI Inference ───────────────────────────────────────────
    t_ai = threading.Thread(
        target=ai_inference_thread,
        name="AIInferenceThread",
        daemon=True,
        kwargs=dict(
            face_detector=face_detector,
            feature_model=feature_model,
            face_db=face_db,          # FaceDBManager instance
            frame_queue=frame_queue,
            relay_queue=relay_queue,
            stop_event=stop_event,
            display_queue=display_queue,
        ),
    )

    # ── Thread 3: Relay Control ──────────────────────────────────────────
    t_relay = threading.Thread(
        target=relay_control_thread,
        name="RelayControlThread",
        daemon=True,
        kwargs=dict(
            relay_service=relay_service,
            relay_queue=relay_queue,
            stop_event=stop_event,
        ),
    )

    # ── Start all threads ────────────────────────────────────────────────
    t_camera.start()
    t_ai.start()
    t_relay.start()

    logger.success("All threads started. System is running. Press Ctrl+C to stop.")

    # ── Main thread: optional display loop ──────────────────────────────
    try:
        if args.display and display_queue is not None:
            while not stop_event.is_set():
                try:
                    annotated = display_queue.get(timeout=0.1)
                except queue.Empty:
                    annotated = None

                if annotated is not None:
                    cv2.imshow("Smart Locker System", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("'q' pressed — stopping.")
                    stop_event.set()
                    break
        else:
            # Headless: main thread waits for stop_event
            while not stop_event.is_set():
                stop_event.wait(timeout=1.0)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — shutting down.")
        stop_event.set()

    finally:
        # ── Graceful shutdown ────────────────────────────────────────────
        logger.info("Shutting down — waiting for threads…")
        stop_event.set()

        t_camera.join(timeout=5.0)
        t_ai.join(timeout=10.0)
        t_relay.join(timeout=5.0)

        relay_service.stop()
        cam.stop()
        HailoModelManager.release()

        if args.display:
            cv2.destroyAllWindows()

        logger.info("=" * 60)
        logger.info(f"  Relay stats: {relay_service.stats}")
        logger.info(f"  Camera frames captured: {cam.frames_captured}")
        logger.info("  Smart Locker System — Shutdown complete.")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()

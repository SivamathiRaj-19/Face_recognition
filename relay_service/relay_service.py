from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from loguru import logger

from relay_service.relay_control import control_relay, CHANNELS

# ---------------------------------------------------------------------------
# Internal constants (overridable via constructor)
# ---------------------------------------------------------------------------

_DEFAULT_COOLDOWN_SECONDS: float = 5.0
_DEFAULT_QUEUE_MAXSIZE: int = 8


class RelayService:


    def __init__(
        self,
        cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        self._cooldown = cooldown_seconds
        self._cmd_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)

        # Per-channel last-trigger timestamps (thread-safe via GIL on float reads)
        self._last_trigger: dict[int, float] = {ch: 0.0 for ch in CHANNELS}

        self._thread: Optional[threading.Thread] = None
        self._stopped: bool = False
        self._commands_sent: int = 0
        self._commands_failed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "RelayService":
        """
        Start the background relay worker thread.

        Returns:
            self — enables fluent chaining.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("RelayService is already running.")
            return self

        self._stopped = False
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="RelayWorkerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("RelayService started (Modbus TCP relay controller).")
        return self

    def stop(self, wait: bool = True) -> None:
        """
        Gracefully stop the relay worker thread.

        Args:
            wait (bool): If True (default), block until the worker exits.
        """
        if self._stopped:
            return

        self._stopped = True
        # Sentinel value signals the worker to exit cleanly
        try:
            self._cmd_queue.put(None, timeout=2.0)
        except queue.Full:
            logger.warning("Relay queue full during shutdown — forcing sentinel.")

        if wait and self._thread is not None:
            self._thread.join(timeout=5.0)

        logger.info(
            f"RelayService stopped. "
            f"Sent: {self._commands_sent}, "
            f"Failed: {self._commands_failed}."
        )

    def trigger(
        self,
        channel: int = 1,
        person_name: str = "unknown",
    ) -> bool:
        """
        Enqueue a relay trigger command (non-blocking).

        The command is silently dropped if:
          • The service is stopped.
          • The channel is still within the cooldown window.
          • The relay queue is full (another trigger is already pending).

        Args:
            channel (int): Relay channel number (1–8).
            person_name (str): Name of the recognised person for logging.

        Returns:
            bool: True if the command was successfully enqueued.
        """
        if self._stopped:
            logger.warning("trigger() called on a stopped RelayService.")
            return False

        if channel not in CHANNELS:
            logger.error(
                f"Invalid relay channel {channel}. Valid channels: {CHANNELS}"
            )
            return False

        now = time.monotonic()
        elapsed = now - self._last_trigger.get(channel, 0.0)
        if elapsed < self._cooldown:
            remaining = self._cooldown - elapsed
            logger.debug(
                f"Channel {channel} on cooldown for {remaining:.1f}s more — "
                f"trigger by '{person_name}' suppressed."
            )
            return False

        try:
            self._cmd_queue.put_nowait((channel, person_name))
            logger.info(
                f"🔓 Relay trigger enqueued → channel={channel}, "
                f"person='{person_name}'"
            )
            return True
        except queue.Full:
            logger.warning(
                f"Relay queue full — trigger for channel {channel} "
                f"by '{person_name}' dropped."
            )
            return False

    # ------------------------------------------------------------------
    # Worker loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """
        Background thread: dequeues relay commands and executes them.

        Reads ``(channel, person_name)`` tuples from ``_cmd_queue``.
        A sentinel ``None`` causes the loop to exit.
        """
        logger.debug("Relay worker thread started.")

        while True:
            try:
                item = self._cmd_queue.get(timeout=1.0)
            except queue.Empty:
                # Periodically check _stopped in case stop() is called
                # after the sentinel was somehow lost
                if self._stopped:
                    break
                continue

            if item is None:
                # Graceful shutdown sentinel
                break

            channel, person_name = item
            self._execute_relay(channel, person_name)
            self._cmd_queue.task_done()

        logger.debug("Relay worker thread exited.")

    def _execute_relay(self, channel: int, person_name: str) -> None:
        """
        Execute a single relay pulse and update the cooldown timestamp.

        Args:
            channel (int): Modbus coil channel to pulse.
            person_name (str): Person name for access-log entry.
        """
        logger.info(
            f"⚡ Executing relay pulse → channel={channel}, "
            f"person='{person_name}'"
        )
        try:
            control_relay(channel)
            self._last_trigger[channel] = time.monotonic()
            self._commands_sent += 1
            logger.success(
                f"✅ Locker {channel} opened for '{person_name}' at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as exc:
            self._commands_failed += 1
            logger.error(
                f"❌ Relay channel {channel} failed for '{person_name}': {exc}"
            )

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Return a snapshot of relay command statistics."""
        return {
            "commands_sent": self._commands_sent,
            "commands_failed": self._commands_failed,
            "queue_depth": self._cmd_queue.qsize(),
            "is_running": self._thread is not None and self._thread.is_alive(),
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "RelayService":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"RelayService("
            f"cooldown={self._cooldown}s, "
            f"sent={self._commands_sent}, "
            f"failed={self._commands_failed})"
        )

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat float32 vectors."""
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


class FaceDBManager:
    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._db_path.exists():
            logger.info(f"No face DB found at {self._db_path} — starting empty.")
            return
        try:
            data = np.load(str(self._db_path), allow_pickle=True).item()
            if not isinstance(data, dict):
                raise ValueError("DB file is not a dict.")
            with self._lock:
                self._db = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
            logger.info(f"Loaded {len(self._db)} face(s) from {self._db_path}.")
        except Exception as exc:
            logger.error(f"Failed to load face DB: {exc} — starting empty.")
            self._db = {}

    def save(self) -> None:
        with self._lock:
            snapshot = dict(self._db)
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(self._db_path), snapshot)
            logger.info(f"Face DB saved → {self._db_path} ({len(snapshot)} entries).")
        except Exception as exc:
            logger.error(f"Failed to save face DB: {exc}")

    def reload(self) -> None:
        with self._lock:
            self._db.clear()
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, name: str, embedding: np.ndarray, overwrite: bool = False) -> bool:
        name = name.strip()
        if not name:
            logger.warning("add() called with an empty name — ignored.")
            return False

        embedding = np.array(embedding, dtype=np.float32).flatten()

        with self._lock:
            if name in self._db and not overwrite:
                logger.warning(
                    f"'{name}' already exists in DB. Pass overwrite=True to replace."
                )
                return False
            self._db[name] = embedding

        self.save()
        action = "Updated" if (name in self._db and overwrite) else "Registered"
        logger.success(f"{action} face: '{name}'  (embedding dim={embedding.shape[0]})")
        return True

    def remove(self, name: str) -> bool:
        with self._lock:
            if name not in self._db:
                logger.warning(f"remove(): '{name}' not found in DB.")
                return False
            del self._db[name]

        self.save()
        logger.info(f"Removed '{name}' from face DB.")
        return True

    def rename(self, old_name: str, new_name: str) -> bool:
        new_name = new_name.strip()
        with self._lock:
            if old_name not in self._db:
                logger.warning(f"rename(): '{old_name}' not found in DB.")
                return False
            if new_name in self._db:
                logger.warning(f"rename(): '{new_name}' already exists.")
                return False
            self._db[new_name] = self._db.pop(old_name)

        self.save()
        logger.info(f"Renamed '{old_name}' → '{new_name}'.")
        return True

    def get_embedding(self, name: str) -> Optional[np.ndarray]:
        with self._lock:
            emb = self._db.get(name)
            return emb.copy() if emb is not None else None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.40,
    ) -> Tuple[Optional[str], float]:
        embedding = np.array(embedding, dtype=np.float32).flatten()

        with self._lock:
            if not self._db:
                return None, 0.0
            snapshot = dict(self._db)

        best_name: Optional[str] = None
        best_score: float = -1.0

        for name, db_emb in snapshot.items():
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= threshold:
            return best_name, best_score
        return None, best_score

    def ranked_matches(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        embedding = np.array(embedding, dtype=np.float32).flatten()

        with self._lock:
            snapshot = dict(self._db)

        scores = [
            (name, cosine_similarity(embedding, emb))
            for name, emb in snapshot.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_names(self) -> List[str]:
        with self._lock:
            return sorted(self._db.keys())

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._db

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._db)

    @property
    def db_path(self) -> str:
        return str(self._db_path)

    def summary(self) -> None:
        names = self.list_names()
        logger.info(f"FaceDBManager — {len(names)} registered face(s):")
        for i, name in enumerate(names, 1):
            dim = self._db[name].shape[0]
            logger.info(f"  {i:>3}. {name}  (dim={dim})")

    def __len__(self) -> int:
        return self.count

    def __contains__(self, name: str) -> bool:
        return self.exists(name)

    def __repr__(self) -> str:
        return f"FaceDBManager(path={self._db_path!r}, count={self.count})"

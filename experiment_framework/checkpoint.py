"""Atomic best-checkpoint promotion with checksums and a write lock."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator


class CheckpointError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class BestCheckpointStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.checkpoint_path = directory / "best_checkpoint.json"
        self.manifest_path = directory / "best_checkpoint.manifest.json"
        self.lock_path = directory / ".checkpoint.lock"

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as exc:
            raise CheckpointError("checkpoint store is locked") from exc
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            yield
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self.lock_path.unlink(missing_ok=True)

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            if self.checkpoint_path.exists():
                raise CheckpointError("checkpoint exists without a manifest")
            return None
        if not self.checkpoint_path.exists():
            raise CheckpointError("manifest exists without a checkpoint")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
        if manifest.get("sha256") != actual:
            raise CheckpointError("best checkpoint checksum mismatch")
        return manifest

    def consider(
        self,
        *,
        node_id: str,
        score: float,
        direction: str,
        checkpoint: dict[str, Any],
    ) -> bool:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be maximize or minimize")
        if not math.isfinite(score):
            raise ValueError("checkpoint score must be finite")
        checkpoint_bytes = _canonical_bytes(checkpoint)
        with self._exclusive_lock():
            current = self._read_manifest()
            if current is not None:
                current_score = float(current["score"])
                better = score > current_score if direction == "maximize" else score < current_score
                if not better:
                    return False
            manifest = {
                "schema_version": 1,
                "node_id": node_id,
                "score": score,
                "direction": direction,
                "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write(self.checkpoint_path, checkpoint_bytes)
            _atomic_write(self.manifest_path, _canonical_bytes(manifest))
            return True

    def validate(self) -> dict[str, Any] | None:
        return self._read_manifest()

"""Durable append-only JSONL event logging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlEventLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._sequence = 0
        self._lock = Lock()

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            record = {
                "sequence": self._sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **fields,
            }
            encoded = json.dumps(record, sort_keys=True, allow_nan=False)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return record

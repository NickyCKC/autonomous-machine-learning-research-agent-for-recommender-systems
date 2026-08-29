"""Deterministic selection policy for research experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DeterministicResearchPolicy:
    """Selects a stable, bounded queue without LLM calls or hidden-test data."""

    def __init__(self, priorities_path: Path) -> None:
        self.priorities_path = priorities_path
        self.document = json.loads(priorities_path.read_text(encoding="utf-8"))
        experiments = self.document.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("research priorities must contain experiments")

    def select(
        self,
        *,
        limit: int = 3,
        max_compute: int = 3,
        max_leakage_risk: int = 3,
    ) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        eligible = [
            experiment
            for experiment in self.document["experiments"]
            if int(experiment["compute"]) <= max_compute
            and int(experiment["leakage_risk"]) <= max_leakage_risk
        ]
        eligible.sort(key=lambda item: (int(item["rank"]), str(item["id"])))
        selected = [str(item["id"]) for item in eligible[:limit]]
        policy_inputs = {
            "limit": limit,
            "max_compute": max_compute,
            "max_leakage_risk": max_leakage_risk,
            "selected": selected,
        }
        fingerprint = hashlib.sha256(
            json.dumps(policy_inputs, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "policy": "rank_then_id_v1",
            "selected_experiment_ids": selected,
            "constraints": {
                "limit": limit,
                "max_compute": max_compute,
                "max_leakage_risk": max_leakage_risk,
                "hidden_test_access": False,
                "llm_calls": False,
            },
            "fingerprint_sha256": fingerprint,
        }

"""Real click-ranking templates for the Part 4 experiment runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiment_framework.registry import TemplateRegistry, TemplateResult
from experiment_framework.schema import ExperimentNode

from .data import Track2Dataset, load_track2_dataset
from .metrics import evaluate_full_catalog
from .models import FMRanker, train_bpr, train_pointwise


_DATA_CACHE: dict[str, Track2Dataset] = {}


def _dataset(path: str) -> Track2Dataset:
    resolved = str(Path(path).resolve())
    if resolved not in _DATA_CACHE:
        _DATA_CACHE[resolved] = load_track2_dataset(resolved)
    return _DATA_CACHE[resolved]


def click_fm_template(node: ExperimentNode, output_dir: Path) -> TemplateResult:
    allowed = {
        "data_dir",
        "loss",
        "factors",
        "learning_rate",
        "l2",
        "epochs",
        "batch_size",
        "evaluation_batch_size",
    }
    unknown = set(node.parameters) - allowed
    if unknown:
        raise ValueError(f"unknown click_fm parameters: {sorted(unknown)}")
    data = _dataset(str(node.parameters.get("data_dir", "KuaiRand-Pure/data")))
    loss = str(node.parameters["loss"])
    model = FMRanker(
        data.dimension,
        factors=int(node.parameters.get("factors", 16)),
        learning_rate=float(node.parameters.get("learning_rate", 0.002)),
        l2=float(node.parameters.get("l2", 1e-6)),
        seed=node.seed,
    )
    training_started = time.monotonic()
    arguments = {
        "epochs": int(node.parameters.get("epochs", 8)),
        "batch_size": int(node.parameters.get("batch_size", 8192)),
        "seed": node.seed,
    }
    if loss == "pointwise":
        history = train_pointwise(model, data.train_X, data.train_y, **arguments)
    elif loss == "bpr":
        history = train_bpr(
            model, data.train_X, data.train_y, data.candidate_X, **arguments
        )
    else:
        raise ValueError("loss must be pointwise or bpr")
    training_runtime = time.monotonic() - training_started
    evaluation_started = time.monotonic()
    metrics = evaluate_full_catalog(
        model,
        data,
        user_batch_size=int(node.parameters.get("evaluation_batch_size", 128)),
    )
    evaluation_runtime = time.monotonic() - evaluation_started

    artifact_path = output_dir / "model.npz"
    np.savez_compressed(artifact_path, **model.state())
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    training_record = {
        "loss": loss,
        "seed": node.seed,
        "fields": list(data.field_names),
        "train_rows": len(data.train_y),
        "train_click_rate": float(data.train_y.mean()),
        "catalogue_items": len(data.candidate_X),
        "validation_users": len(data.validation_user_features),
        "training_runtime_seconds": training_runtime,
        "evaluation_runtime_seconds": evaluation_runtime,
        "history": history,
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(training_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint = {
        "kind": "track2_click_fm",
        "node_id": node.node_id,
        "loss": loss,
        "seed": node.seed,
        "metrics": metrics,
        "artifact": "model.npz",
        "artifact_sha256": artifact_sha256,
        "protocol": "full_catalogue_v1",
    }
    return TemplateResult(
        metrics=metrics,
        checkpoint=checkpoint,
        notes="Click relevance; full 7,583-video catalogue; training-click filtering.",
    )


def build_track2_registry() -> TemplateRegistry:
    registry = TemplateRegistry()
    registry.register("click_fm", click_fm_template)
    return registry

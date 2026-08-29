"""Real click-ranking templates for the Part 4 experiment runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiment_framework.registry import TemplateRegistry, TemplateResult
from experiment_framework.schema import ExperimentNode

from .data import Track2Dataset, load_track2_dataset
from .metrics import evaluate_full_catalog
from .history import DINLiteRanker
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
    registry.register("din_lite", din_lite_template)
    registry.register("checkpoint_reference", checkpoint_reference_template)
    return registry


def _copy_verified_artifact(source: Path, destination: Path, expected_sha256: str) -> str:
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"source model checksum mismatch: {source}")
    shutil.copyfile(source, destination)
    copied = hashlib.sha256(destination.read_bytes()).hexdigest()
    if copied != expected_sha256:
        raise ValueError("copied model checksum mismatch")
    return copied


def din_lite_template(node: ExperimentNode, output_dir: Path) -> TemplateResult:
    allowed = {
        "data_dir",
        "base_model",
        "base_model_sha256",
        "alpha",
        "max_history",
        "recency_decay",
        "author_weight",
        "duration_weight",
        "evaluation_batch_size",
    }
    unknown = set(node.parameters) - allowed
    if unknown:
        raise ValueError(f"unknown DIN-lite parameters: {sorted(unknown)}")
    data = _dataset(str(node.parameters.get("data_dir", "KuaiRand-Pure/data")))
    source = Path(str(node.parameters["base_model"]))
    expected_sha256 = str(node.parameters["base_model_sha256"])
    copied_sha256 = _copy_verified_artifact(
        source, output_dir / "model.npz", expected_sha256
    )
    base_model = FMRanker.from_npz(str(output_dir / "model.npz"))
    model = DINLiteRanker(
        base_model,
        data,
        alpha=float(node.parameters["alpha"]),
        max_history=int(node.parameters.get("max_history", 50)),
        recency_decay=float(node.parameters.get("recency_decay", 0.05)),
        author_weight=float(node.parameters.get("author_weight", 0.5)),
        duration_weight=float(node.parameters.get("duration_weight", 0.1)),
    )
    started = time.monotonic()
    metrics = evaluate_full_catalog(
        model,
        data,
        user_batch_size=int(node.parameters.get("evaluation_batch_size", 128)),
    )
    evaluation_runtime = time.monotonic() - started
    record = {
        "seed": node.seed,
        "strict_history_source": "training clicks only",
        "alpha": model.alpha,
        "max_history": int(node.parameters.get("max_history", 50)),
        "recency_decay": float(node.parameters.get("recency_decay", 0.05)),
        "evaluation_runtime_seconds": evaluation_runtime,
        "metrics": metrics,
    }
    (output_dir / "history_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint = {
        "kind": "track2_din_lite",
        "node_id": node.node_id,
        "seed": node.seed,
        "metrics": metrics,
        "artifact": "model.npz",
        "artifact_sha256": copied_sha256,
        "history_config": record,
        "protocol": "full_catalogue_v1",
    }
    return TemplateResult(
        metrics=metrics,
        checkpoint=checkpoint,
        notes="DIN-lite uses only chronological training clicks; no validation history.",
    )


def checkpoint_reference_template(
    node: ExperimentNode, output_dir: Path
) -> TemplateResult:
    allowed = {
        "model",
        "model_sha256",
        "ndcg_at_10",
        "recall_at_50",
        "combined",
    }
    unknown = set(node.parameters) - allowed
    if unknown:
        raise ValueError(f"unknown checkpoint reference parameters: {sorted(unknown)}")
    source = Path(str(node.parameters["model"]))
    expected_sha256 = str(node.parameters["model_sha256"])
    copied_sha256 = _copy_verified_artifact(
        source, output_dir / "model.npz", expected_sha256
    )
    metrics = {
        "validation.ndcg_at_10": float(node.parameters["ndcg_at_10"]),
        "validation.recall_at_50": float(node.parameters["recall_at_50"]),
        "validation.combined": float(node.parameters["combined"]),
    }
    return TemplateResult(
        metrics=metrics,
        checkpoint={
            "kind": "verified_existing_checkpoint",
            "node_id": node.node_id,
            "metrics": metrics,
            "artifact": "model.npz",
            "artifact_sha256": copied_sha256,
            "protocol": "full_catalogue_v1",
        },
        notes="Verified reference protects the previous global best.",
    )

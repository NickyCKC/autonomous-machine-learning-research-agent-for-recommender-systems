"""Allowlisted experiment template registry and safe demo templates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Callable

from .schema import ExperimentNode


@dataclass(frozen=True)
class TemplateResult:
    metrics: dict[str, float]
    checkpoint: dict[str, Any]
    notes: str = ""


Template = Callable[[ExperimentNode, Path], TemplateResult]


class TemplateRegistry:
    """Maps explicit template names to functions; arbitrary commands are rejected."""

    def __init__(self) -> None:
        self._templates: dict[str, Template] = {}

    def register(self, name: str, template: Template) -> None:
        if not name or name in self._templates:
            raise ValueError(f"template name is empty or already registered: {name!r}")
        self._templates[name] = template

    def execute(self, node: ExperimentNode, output_dir: Path) -> TemplateResult:
        try:
            template = self._templates[node.template]
        except KeyError as exc:
            raise ValueError(f"unregistered template: {node.template}") from exc
        result = template(node, output_dir)
        for metric, value in result.metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"metric {metric} must be finite")
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))


def synthetic_ranker(node: ExperimentNode, output_dir: Path) -> TemplateResult:
    """Return a deterministic proxy score without training a competition model."""
    allowed = {"base_score", "uplift", "jitter"}
    unknown = set(node.parameters) - allowed
    if unknown:
        raise ValueError(f"unknown synthetic_ranker parameters: {sorted(unknown)}")
    base = float(node.parameters.get("base_score", 0.5))
    uplift = float(node.parameters.get("uplift", 0.0))
    jitter = float(node.parameters.get("jitter", 0.0001))
    score = base + uplift + random.Random(node.seed).uniform(-jitter, jitter)
    metric = node.objective.metric
    checkpoint = {
        "kind": "synthetic_demo_only",
        "node_id": node.node_id,
        "experiment_id": node.experiment_id,
        "seed": node.seed,
        "parameters": node.parameters,
        "score": score,
    }
    (output_dir / "template_output.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TemplateResult(
        metrics={metric: score},
        checkpoint=checkpoint,
        notes="Synthetic framework smoke test; not an official model result.",
    )


def intentional_failure(node: ExperimentNode, output_dir: Path) -> TemplateResult:
    """Create a partial artifact, then fail to exercise isolation and recovery."""
    (output_dir / "partial_artifact.txt").write_text(
        "This incomplete artifact must never become the best checkpoint.\n",
        encoding="utf-8",
    )
    raise RuntimeError("intentional Part 3 failure-safety demonstration")


def build_default_registry() -> TemplateRegistry:
    registry = TemplateRegistry()
    registry.register("synthetic_ranker", synthetic_ranker)
    registry.register("intentional_failure", intentional_failure)
    return registry

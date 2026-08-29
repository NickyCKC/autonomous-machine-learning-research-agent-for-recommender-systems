#!/usr/bin/env python3
"""Safe autonomous experiment controller for the official Track 2 evaluator.

The controller never edits code. A policy may only select a registered,
reviewed experiment. This makes LLM providers interchangeable while keeping
the evaluator, data splits, and experiment budget outside the model's control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from baseline import FM, run_fm_ensemble, run_hybrid_ensemble
from data import encode, load
from evaluate import evaluate


ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "evaluate.py"
EXPECTED_EVALUATOR_SHA256 = (
    "c76b598fa83fe79fe33aaf554e46807d25678c1ecf5ae7633761e911e7a6e24b"
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    family: str
    description: str
    seeds: tuple[int, ...]
    parameters: dict[str, int | float]


REGISTRY = {
    "fm_seed0_control": Experiment(
        "fm_seed0_control",
        "pointwise_fm",
        "Single-seed starter FM control.",
        (0,),
        {"k": 16, "lr": 0.001, "epochs": 40},
    ),
    "fm_ensemble_4": Experiment(
        "fm_ensemble_4",
        "pointwise_fm",
        "Average four independently seeded starter FMs.",
        (0, 1, 2, 3),
        {"k": 16, "lr": 0.001, "epochs": 40},
    ),
    "hybrid_bpr_ensemble_4": Experiment(
        "hybrid_bpr_ensemble_4",
        "hybrid_bpr",
        "Average four FMs continued on within-user training-only BPR pairs.",
        (0, 1, 2, 3),
        {
            "k": 16,
            "lr": 0.001,
            "pointwise_epochs": 40,
            "bpr_epochs": 6,
            "bpr_lr": 0.0002,
            "pairs_per_positive": 1,
        },
    ),
    "fm_k8_seed0": Experiment(
        "fm_k8_seed0",
        "pointwise_fm",
        "Fast screen of a smaller eight-dimensional FM.",
        (0,),
        {"k": 8, "lr": 0.001, "epochs": 40},
    ),
    "fm_k32_seed0": Experiment(
        "fm_k32_seed0",
        "pointwise_fm",
        "Fast screen of a larger 32-dimensional FM.",
        (0,),
        {"k": 32, "lr": 0.001, "epochs": 40},
    ),
    "hybrid_bpr_low_lr_seed0": Experiment(
        "hybrid_bpr_low_lr_seed0",
        "hybrid_bpr",
        "Fast screen of gentler BPR continuation.",
        (0,),
        {
            "k": 16, "lr": 0.001, "pointwise_epochs": 40,
            "bpr_epochs": 6, "bpr_lr": 0.0001, "pairs_per_positive": 1,
        },
    ),
    "hybrid_bpr_high_lr_seed0": Experiment(
        "hybrid_bpr_high_lr_seed0",
        "hybrid_bpr",
        "Fast screen of stronger BPR continuation.",
        (0,),
        {
            "k": 16, "lr": 0.001, "pointwise_epochs": 40,
            "bpr_epochs": 6, "bpr_lr": 0.0004, "pairs_per_positive": 1,
        },
    ),
    "hybrid_bpr_pairs2_seed0": Experiment(
        "hybrid_bpr_pairs2_seed0",
        "hybrid_bpr",
        "Fast screen using two sampled negatives per positive.",
        (0,),
        {
            "k": 16, "lr": 0.001, "pointwise_epochs": 40,
            "bpr_epochs": 6, "bpr_lr": 0.0001, "pairs_per_positive": 2,
        },
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """Recursively convert NumPy values to portable JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class DeterministicPolicy:
    """Reproducible no-cost policy used for testing and offline operation."""

    priority = (
        "fm_seed0_control",
        "fm_ensemble_4",
        "hybrid_bpr_ensemble_4",
        "fm_k8_seed0",
        "fm_k32_seed0",
        "hybrid_bpr_low_lr_seed0",
        "hybrid_bpr_high_lr_seed0",
        "hybrid_bpr_pairs2_seed0",
    )

    def choose(self, context: dict, remaining: list[Experiment]) -> tuple[str, dict]:
        available = {experiment.experiment_id for experiment in remaining}
        choice = next(item for item in self.priority if item in available)
        return choice, {"provider": "deterministic", "usage": None}


class CommandLLMPolicy:
    """Provider-neutral adapter around any JSON-in/JSON-out LLM command.

    The configured command receives one JSON document on stdin and must return
    ``{"experiment_id": "..."}`` on stdout. Optional provider/model/usage
    fields are copied into the audit log. The command is executed without a
    shell and cannot select an unregistered experiment.
    """

    def __init__(self, command: str, timeout_seconds: int = 120):
        self.command = shlex.split(command)
        if not self.command:
            raise ValueError("LLM policy command cannot be empty")
        self.timeout_seconds = timeout_seconds

    def choose(self, context: dict, remaining: list[Experiment]) -> tuple[str, dict]:
        request = {
            "task": (
                "Select one experiment_id. Optimize validation primary, which "
                "is mean(GAUC, nDCG@5). Do not request code or evaluator changes."
            ),
            "context": context,
            "allowed_experiments": [asdict(item) for item in remaining],
        }
        completed = subprocess.run(
            self.command,
            input=json.dumps(json_safe(request)),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "no stderr"
            raise RuntimeError(
                f"LLM policy command exited {completed.returncode}: {detail}"
            )
        response = json.loads(completed.stdout)
        return response["experiment_id"], {
            "provider": response.get("provider", "command"),
            "model": response.get("model"),
            "usage": response.get("usage"),
            "reason": response.get("reason"),
            "response_id": response.get("response_id"),
        }


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(event), sort_keys=True) + "\n")


def stronger_result(incumbent: dict | None, candidate: dict | None) -> dict | None:
    """Return the higher-validation result without ever demoting the incumbent."""
    if candidate is None:
        return incumbent
    if incumbent is None:
        return candidate
    if candidate["metrics"]["primary"] > incumbent["metrics"]["primary"]:
        return candidate
    return incumbent


def save_models(path: Path, models: list, experiment: Experiment) -> str:
    arrays = {"metadata": np.asarray(json.dumps(asdict(experiment)))}
    for index, model in enumerate(models):
        arrays[f"V_{index}"] = model.V
        arrays[f"W_{index}"] = model.W
        arrays[f"b_{index}"] = np.asarray(model.b)
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return sha256(path)


def recover_models(path: Path, experiment: Experiment) -> list[FM]:
    """Load a completed atomic artifact after an interrupted agent run."""
    models = []
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["experiment_id"] != experiment.experiment_id:
            raise ValueError(f"Artifact metadata mismatch: {path}")
        index = 0
        while f"V_{index}" in archive:
            V = archive[f"V_{index}"]
            model = FM(V.shape[0], k=V.shape[1], seed=0)
            model.V = V.copy()
            model.W = archive[f"W_{index}"].copy()
            model.b = np.float32(archive[f"b_{index}"])
            models.append(model)
            index += 1
    if not models:
        raise ValueError(f"Artifact contains no models: {path}")
    return models


def run_experiment(experiment: Experiment, splits: dict) -> tuple[dict, list]:
    common = dict(
        seeds=experiment.seeds,
        verbose=False,
        evaluate_test=False,
        return_models=True,
    )
    if experiment.family == "pointwise_fm":
        metrics, models, _ = run_fm_ensemble(
            splits, **common, **experiment.parameters
        )
    elif experiment.family == "hybrid_bpr":
        metrics, models, _ = run_hybrid_ensemble(
            splits, **common, **experiment.parameters
        )
    else:
        raise ValueError(f"Unsupported registered family: {experiment.family}")
    return metrics["valid"], models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "KuaiRand-Pure" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "official_agent")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument(
        "--experiments",
        help="Optional comma-separated subset of registered experiment IDs",
    )
    parser.add_argument("--policy", choices=("deterministic", "command"), default="deterministic")
    parser.add_argument("--policy-command")
    parser.add_argument("--policy-timeout", type=int, default=120)
    parser.add_argument(
        "--policy-retries", type=int, default=2,
        help="Number of retries after a failed LLM decision call",
    )
    parser.add_argument(
        "--recover-run",
        help="Recover completed atomic artifacts from a previously interrupted run ID",
    )
    parser.add_argument("--list", action="store_true", help="List safe experiments and exit")
    args = parser.parse_args()

    if args.list:
        print(json.dumps({key: asdict(value) for key, value in REGISTRY.items()}, indent=2))
        return 0
    if args.budget < 1:
        parser.error("--budget must be at least 1")
    if args.policy_retries < 0:
        parser.error("--policy-retries cannot be negative")
    if args.policy == "command" and not args.policy_command:
        parser.error("--policy-command is required for command policy")
    if sha256(EVALUATOR) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("evaluate.py differs from the preserved official evaluator")

    policy = (
        DeterministicPolicy()
        if args.policy == "deterministic"
        else CommandLLMPolicy(args.policy_command, args.policy_timeout)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = args.output_dir / "events.jsonl"
    manifest = args.output_dir / "best.json"
    global_best = (
        json.loads(manifest.read_text(encoding="utf-8"))
        if manifest.is_file() else None
    )
    run_id = args.recover_run or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    context = {
        "run_id": run_id,
        "completed": [],
        "best": None,
        "incumbent_at_start": global_best,
    }
    remaining = list(REGISTRY.values())
    if args.experiments:
        requested = [item.strip() for item in args.experiments.split(",") if item.strip()]
        unknown = set(requested) - set(REGISTRY)
        if unknown:
            parser.error(f"Unknown experiment IDs: {sorted(unknown)}")
        requested_set = set(requested)
        remaining = [item for item in remaining if item.experiment_id in requested_set]
    splits = load(str(args.data_dir))

    if args.recover_run:
        enc, _ = encode(splits)
        Xva, yva, uva = enc["valid"]
        for experiment in remaining[:min(args.budget, len(remaining))]:
            artifact = args.output_dir / f"{run_id}_{experiment.experiment_id}.npz"
            models = recover_models(artifact, experiment)
            scores = sum(
                (model.predict(Xva).astype(np.float64) for model in models),
                np.zeros(len(yva), dtype=np.float64),
            ) / len(models)
            metrics = evaluate(uva, yva, scores)
            result = {
                "experiment_id": experiment.experiment_id,
                "metrics": metrics,
                "runtime_seconds": 0.0,
                "artifact": artifact.name,
                "artifact_sha256": sha256(artifact),
                "recovered": True,
            }
            context["completed"].append(result)
            context["best"] = stronger_result(context["best"], result)
            append_event(events, {
                "event": "experiment_recovered", "run_id": run_id, **result,
            })
        remaining = []

    for iteration in range(1, min(args.budget, len(remaining)) + 1):
        for attempt in range(1, args.policy_retries + 2):
            try:
                choice, policy_metadata = policy.choose(context, remaining)
                break
            except Exception as error:
                append_event(events, {
                    "event": "policy_call_failed", "run_id": run_id,
                    "iteration": iteration, "attempt": attempt,
                    "will_retry": attempt <= args.policy_retries,
                    "error_type": type(error).__name__, "error": str(error),
                })
                if attempt > args.policy_retries:
                    raise
        allowed = {item.experiment_id: item for item in remaining}
        if choice not in allowed:
            raise ValueError(f"Policy selected unregistered experiment: {choice}")
        experiment = allowed[choice]
        append_event(events, {
            "event": "experiment_started", "run_id": run_id,
            "iteration": iteration, "experiment": asdict(experiment),
            "policy": policy_metadata,
        })
        started = time.perf_counter()
        try:
            metrics, models = run_experiment(experiment, splits)
            runtime = time.perf_counter() - started
            artifact = args.output_dir / f"{run_id}_{choice}.npz"
            artifact_hash = save_models(artifact, models, experiment)
            result = {
                "experiment_id": choice,
                "metrics": metrics,
                "runtime_seconds": runtime,
                "artifact": artifact.name,
                "artifact_sha256": artifact_hash,
            }
            context["completed"].append(result)
            context["best"] = stronger_result(context["best"], result)
            append_event(events, {
                "event": "experiment_succeeded", "run_id": run_id,
                "iteration": iteration, **result,
            })
        except Exception as error:
            append_event(events, {
                "event": "experiment_failed", "run_id": run_id,
                "iteration": iteration, "experiment_id": choice,
                "error_type": type(error).__name__, "error": str(error),
            })
            context["completed"].append({
                "experiment_id": choice,
                "status": "failed",
                "error_type": type(error).__name__,
            })
        remaining.remove(experiment)

    global_best = stronger_result(global_best, context["best"])
    context["global_best"] = global_best
    if global_best is not None:
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(json_safe(global_best), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest)

    if sha256(EVALUATOR) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("evaluate.py changed during the agent run")
    summary = args.output_dir / f"{run_id}_summary.json"
    summary.write_text(
        json.dumps(json_safe(context), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(context), indent=2))
    return 0 if context["best"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

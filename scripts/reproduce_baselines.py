#!/usr/bin/env python3
"""Reproduce the KuaiRand-Pure starter-kit baselines with auditable results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline import run_fm, run_pop, run_random  # noqa: E402
from data import load  # noqa: E402


EXPECTED_EVALUATOR_SHA256 = (
    "c76b598fa83fe79fe33aaf554e46807d25678c1ecf5ae7633761e911e7a6e24b"
)
EXPECTED_ROWS = {"train": 1_141_112, "valid": 124_909, "test": 170_588}
RANDOM_TEST_PRIMARY = 0.4753
FM_TEST_PRIMARY = 0.5946


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": __import__("numpy").__version__,
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in sorted({run["model"] for run in runs}):
        model_runs = [run for run in runs if run["model"] == model]
        output[model] = {
            "run_count": len(model_runs),
            "runtime_seconds": {
                "mean": statistics.mean(run["runtime_seconds"] for run in model_runs),
                "total": sum(run["runtime_seconds"] for run in model_runs),
            },
        }
        for split in ("valid", "test"):
            output[model][split] = {}
            keys = ("GAUC", "nDCG@5", "primary")
            for key in keys:
                values = [run["metrics"][split][key] for run in model_runs]
                output[model][split][key] = {
                    "mean": statistics.mean(values),
                    "population_std": statistics.pstdev(values),
                    "min": min(values),
                    "max": max(values),
                }
    return output


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars and nested containers to JSON-native values."""
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "KuaiRand-Pure.tar.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "baseline_results.json",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from the existing output JSON",
    )
    args = parser.parse_args()

    evaluator_path = ROOT / "evaluate.py"
    evaluator_hash = sha256(evaluator_path)
    if evaluator_hash != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            "evaluate.py changed before reproduction: "
            f"expected {EXPECTED_EVALUATOR_SHA256}, received {evaluator_hash}"
        )
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    started = datetime.now(timezone.utc)
    if args.resume and args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload["integrity"]["evaluate_py_sha256"] != evaluator_hash:
            raise RuntimeError("Cannot resume because the evaluator hash changed")
        payload["status"] = "running"
        payload["resumed_at_utc"] = started.isoformat()
        payload["resume_command"] = [sys.executable, *sys.argv]
    else:
        payload = {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": started.isoformat(),
            "command": [sys.executable, *sys.argv],
            "environment": environment(),
            "integrity": {
                "evaluate_py_sha256": evaluator_hash,
                "dataset_archive_sha256": sha256(args.archive) if args.archive.is_file() else None,
            },
            "expected": {
                "rows": EXPECTED_ROWS,
                "random_test_primary": RANDOM_TEST_PRIMARY,
                "fm_test_primary": FM_TEST_PRIMARY,
                "random_tolerance": 0.002,
                "fm_tolerance": 0.003,
            },
            "runs": [],
        }
    write_result(args.output, payload)

    print(f"Loading data from {args.data_dir} ...", flush=True)
    load_started = time.perf_counter()
    splits = load(str(args.data_dir))
    payload["dataset"] = {
        "rows": {name: len(rows) for name, rows in splits.items()},
        "load_seconds": time.perf_counter() - load_started,
    }
    print(f"Rows: {payload['dataset']['rows']}", flush=True)

    def execute(model: str, seed: int | None, function: Any) -> None:
        label = model if seed is None else f"{model} seed={seed}"
        print(f"\nRunning {label} ...", flush=True)
        run_started = time.perf_counter()
        metrics = json_safe(function())
        record = {
            "model": model,
            "seed": seed,
            "runtime_seconds": time.perf_counter() - run_started,
            "metrics": metrics,
        }
        payload["runs"].append(record)
        write_result(args.output, payload)
        print(json.dumps(record, indent=2), flush=True)

    completed = {(run["model"], run["seed"]) for run in payload["runs"]}

    for seed in args.seeds:
        if ("random", seed) not in completed:
            execute("random", seed, lambda seed=seed: run_random(splits, seed=seed))
    if ("popularity", None) not in completed:
        execute("popularity", None, lambda: run_pop(splits))
    for seed in args.seeds:
        if ("fm", seed) not in completed:
            execute(
                "fm",
                seed,
                lambda seed=seed: run_fm(splits, seed=seed, verbose=True),
            )

    payload["aggregates"] = aggregate(payload["runs"])
    rows_match = payload["dataset"]["rows"] == EXPECTED_ROWS
    random_mean = payload["aggregates"]["random"]["test"]["primary"]["mean"]
    fm_mean = payload["aggregates"]["fm"]["test"]["primary"]["mean"]
    payload["acceptance"] = {
        "dataset_rows_match": rows_match,
        "random_test_primary_delta": random_mean - RANDOM_TEST_PRIMARY,
        "random_test_primary_pass": abs(random_mean - RANDOM_TEST_PRIMARY) <= 0.002,
        "fm_test_primary_delta": fm_mean - FM_TEST_PRIMARY,
        "fm_test_primary_pass": abs(fm_mean - FM_TEST_PRIMARY) <= 0.003,
        "evaluator_unchanged": sha256(evaluator_path) == EXPECTED_EVALUATOR_SHA256,
    }
    required_checks = (
        "dataset_rows_match",
        "random_test_primary_pass",
        "fm_test_primary_pass",
        "evaluator_unchanged",
    )
    payload["status"] = (
        "passed"
        if all(payload["acceptance"][name] for name in required_checks)
        else "failed"
    )
    completed = datetime.now(timezone.utc)
    original_started = datetime.fromisoformat(payload["started_at_utc"])
    payload["completed_at_utc"] = completed.isoformat()
    payload["total_seconds"] = (completed - original_started).total_seconds()
    write_result(args.output, payload)

    print("\nAcceptance:", flush=True)
    print(json.dumps(payload["acceptance"], indent=2), flush=True)
    print(f"Result: {args.output}", flush=True)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

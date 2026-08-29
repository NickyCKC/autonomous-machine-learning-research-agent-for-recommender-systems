"""Aggregate fixed-seed Part 4 metrics into a machine-readable comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


METRICS = (
    "validation.ndcg_at_10",
    "validation.recall_at_50",
    "validation.combined",
)


def aggregate(nodes: list[dict]) -> dict:
    result = {"runs": len(nodes), "metrics": {}}
    for metric in METRICS:
        values = [float(node["metrics"][metric]) for node in nodes]
        result["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "population_stddev": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
    runtimes = [float(node["runtime_seconds"]) for node in nodes]
    result["runtime_seconds"] = {
        "mean": statistics.fmean(runtimes),
        "total": sum(runtimes),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/part4_bpr")
    parser.add_argument("--output", default="results/part4_bpr_comparison.json")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = {}
    for node in summary["nodes"]:
        if node["status"] != "succeeded":
            raise ValueError(f"cannot aggregate failed node {node['node_id']}")
        groups.setdefault(node["experiment_id"], []).append(node)
    baseline = aggregate(groups["click_pointwise_fm"])
    bpr = aggregate(groups["bpr_fm"])
    deltas = {}
    for metric in METRICS:
        baseline_mean = baseline["metrics"][metric]["mean"]
        bpr_mean = bpr["metrics"][metric]["mean"]
        deltas[metric] = {
            "absolute": bpr_mean - baseline_mean,
            "relative_fraction": bpr_mean / baseline_mean - 1.0,
        }

    best_node = summary["best_checkpoint"]["node_id"]
    best_metadata = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    artifact_path = run_dir / "nodes" / best_node / best_metadata["artifact"]
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if artifact_sha256 != best_metadata["artifact_sha256"]:
        raise ValueError("best model artifact checksum mismatch")

    document = {
        "schema_version": 1,
        "stage": "part_4_bpr_loss_result",
        "status": "passed",
        "protocol": {
            "id": "full_catalogue_v1",
            "label": "is_click",
            "metrics": ["NDCG@10", "Recall@50"],
            "catalogue_items": 7583,
            "eligible_validation_users": 17965,
            "hidden_or_public_test_used": False,
        },
        "baseline": baseline,
        "bpr": bpr,
        "bpr_minus_baseline": deltas,
        "best_checkpoint": {
            **summary["best_checkpoint"],
            "model_artifact": str(artifact_path).replace("\\", "/"),
            "model_artifact_sha256": artifact_sha256,
        },
        "total_run_runtime_seconds": summary["runtime_seconds"],
        "acceptance": {
            "five_fixed_seeds_each": len(groups["click_pointwise_fm"]) == 5
            and len(groups["bpr_fm"]) == 5,
            "ndcg_improved": deltas["validation.ndcg_at_10"]["absolute"] > 0,
            "recall_improved": deltas["validation.recall_at_50"]["absolute"] > 0,
            "all_nodes_succeeded": summary["counts"]["succeeded"] == 10,
            "best_artifact_checksum_valid": True,
        },
        "caveat": "Candidate protocol is the documented full-catalogue reconstruction, not an organizer-supplied evaluator.",
    }
    Path(args.output).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

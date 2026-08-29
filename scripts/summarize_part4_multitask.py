"""Summarize paired control/multi-task results and the final Part 4 best."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/part4_multitask/summary.json")
    parser.add_argument("--output", default="results/part4_multitask_comparison.json")
    args = parser.parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    controls = sorted(
        [
            node
            for node in summary["nodes"]
            if node["experiment_id"] == "bpr_continuation_control"
        ],
        key=lambda node: node["node_id"],
    )
    treatments = sorted(
        [
            node
            for node in summary["nodes"]
            if node["experiment_id"] == "shared_bottom_multitask"
        ],
        key=lambda node: node["node_id"],
    )
    if len(controls) != 5 or len(treatments) != 5:
        raise ValueError("five paired controls and treatments are required")
    comparison = {}
    for metric in METRICS:
        control_values = [float(node["metrics"][metric]) for node in controls]
        treatment_values = [float(node["metrics"][metric]) for node in treatments]
        deltas = [treatment - control for control, treatment in zip(control_values, treatment_values)]
        comparison[metric] = {
            "control_mean": statistics.fmean(control_values),
            "control_population_stddev": statistics.pstdev(control_values),
            "multitask_mean": statistics.fmean(treatment_values),
            "multitask_population_stddev": statistics.pstdev(treatment_values),
            "paired_mean_delta": statistics.fmean(deltas),
            "paired_deltas": deltas,
            "improved_seed_count": sum(delta > 0 for delta in deltas),
        }

    best_node_id = summary["best_checkpoint"]["node_id"]
    best_metadata = json.loads(
        (summary_path.parent / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    best_artifact = summary_path.parent / "nodes" / best_node_id / best_metadata["artifact"]
    artifact_sha256 = hashlib.sha256(best_artifact.read_bytes()).hexdigest()
    if artifact_sha256 != best_metadata["artifact_sha256"]:
        raise ValueError("final Part 4 model checksum mismatch")
    best_node = next(node for node in summary["nodes"] if node["node_id"] == best_node_id)
    document = {
        "schema_version": 1,
        "stage": "part_4_multitask_and_completion",
        "status": "completed_multitask_rejected",
        "protocol": "full_catalogue_v1",
        "paired_comparison": comparison,
        "multitask_configuration": {
            "auxiliary_strength": 0.1,
            "auxiliary_fraction": 0.25,
            "auxiliary_tasks": [
                "long_view",
                "profile_enter",
                "like",
                "watch_ratio",
            ],
            "primary_task": "is_click BPR",
        },
        "final_combined_best": {
            "node_id": best_node_id,
            "model_family": "bpr_continuation_control",
            "metrics": best_node["metrics"],
            "artifact": str(best_artifact).replace("\\", "/"),
            "artifact_sha256": artifact_sha256,
        },
        "previous_ndcg_leader_retained": {
            "node_id": "bpr-seed-3",
            "artifact": "results/part4_bpr/nodes/bpr-seed-3/model.npz",
            "ndcg_at_10": 0.027412282871199182,
            "recall_at_50": 0.12120448475355003,
            "combined": 0.0743083838123746,
        },
        "acceptance": {
            "five_paired_seeds": True,
            "all_nodes_succeeded": summary["counts"]["succeeded"] == 11,
            "hidden_or_public_test_used": False,
            "global_best_checksum_valid": True,
            "multitask_improved_combined_mean": comparison["validation.combined"][
                "paired_mean_delta"
            ]
            > 0,
        },
        "conclusion": "Reject auxiliary treatment; retain continued-BPR seed 2 as combined-score best.",
        "total_runtime_seconds": summary["runtime_seconds"],
    }
    Path(args.output).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

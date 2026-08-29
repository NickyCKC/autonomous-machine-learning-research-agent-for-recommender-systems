"""Create the paired BPR-versus-DIN-lite comparison."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--bpr-summary", default="results/part4_bpr/summary.json")
    parser.add_argument("--din-summary", default="results/part4_din/summary.json")
    parser.add_argument("--output", default="results/part4_din_comparison.json")
    args = parser.parse_args()
    bpr_summary = json.loads(Path(args.bpr_summary).read_text(encoding="utf-8"))
    din_summary = json.loads(Path(args.din_summary).read_text(encoding="utf-8"))
    bpr_nodes = sorted(
        [node for node in bpr_summary["nodes"] if node["experiment_id"] == "bpr_fm"],
        key=lambda node: node["node_id"],
    )
    din_nodes = sorted(
        [
            node
            for node in din_summary["nodes"]
            if node["experiment_id"] == "din_lite_history"
        ],
        key=lambda node: node["node_id"],
    )
    if len(bpr_nodes) != 5 or len(din_nodes) != 5:
        raise ValueError("paired comparison requires five BPR and five DIN-lite nodes")
    comparison = {}
    for metric in METRICS:
        bpr_values = [float(node["metrics"][metric]) for node in bpr_nodes]
        din_values = [float(node["metrics"][metric]) for node in din_nodes]
        paired_deltas = [din - bpr for bpr, din in zip(bpr_values, din_values)]
        comparison[metric] = {
            "bpr_mean": statistics.fmean(bpr_values),
            "din_lite_mean": statistics.fmean(din_values),
            "din_lite_population_stddev": statistics.pstdev(din_values),
            "paired_mean_delta": statistics.fmean(paired_deltas),
            "paired_deltas": paired_deltas,
            "improved_seed_count": sum(delta > 0 for delta in paired_deltas),
        }
    document = {
        "schema_version": 1,
        "stage": "part_4_din_lite_result",
        "status": "completed_hypothesis_rejected",
        "protocol": "full_catalogue_v1",
        "label": "is_click",
        "fixed_configuration": {
            "alpha": 0.025,
            "max_history": 50,
            "recency_decay": 0.05,
            "author_weight": 0.5,
            "duration_weight": 0.1,
            "history_source": "chronological training clicks only",
        },
        "comparison": comparison,
        "global_best": {
            "node_id": din_summary["best_checkpoint"]["node_id"],
            "combined": din_summary["best_checkpoint"]["score"],
            "model_family": "bpr_fm",
        },
        "acceptance": {
            "five_paired_seeds": True,
            "all_nodes_succeeded": din_summary["counts"]["succeeded"] == 6,
            "no_validation_or_test_history": True,
            "previous_global_best_protected": din_summary["best_checkpoint"]["node_id"]
            == "bpr-global-best-reference",
            "din_improved_combined_mean": comparison["validation.combined"][
                "paired_mean_delta"
            ]
            > 0,
        },
        "conclusion": "Reject DIN-lite configuration; retain BPR seed 3.",
        "total_runtime_seconds": din_summary["runtime_seconds"],
    }
    Path(args.output).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

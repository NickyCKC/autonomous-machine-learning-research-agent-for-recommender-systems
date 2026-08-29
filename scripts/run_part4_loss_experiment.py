"""Run the fixed-seed pointwise-versus-BPR Part 4 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_framework.policy import DeterministicResearchPolicy
from experiment_framework.runner import ExperimentRunner
from experiment_framework.schema import load_plan
from track2.templates import build_track2_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/part4_bpr_nodes.json")
    parser.add_argument("--output", default="results/part4_bpr")
    parser.add_argument("--priorities", default="results/research_priorities.json")
    args = parser.parse_args()
    nodes = load_plan(args.plan)
    policy = DeterministicResearchPolicy(Path(args.priorities)).select(limit=3)
    summary = ExperimentRunner(build_track2_registry()).run(
        nodes, Path(args.output), policy_decision=policy
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

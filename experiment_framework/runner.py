"""Sequential unattended runner with failure isolation and protected promotion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any

from .checkpoint import BestCheckpointStore
from .events import JsonlEventLogger
from .registry import TemplateRegistry, build_default_registry
from .schema import ExperimentNode, load_plan


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class ExperimentRunner:
    def __init__(self, registry: TemplateRegistry | None = None) -> None:
        self.registry = registry or build_default_registry()

    def run(
        self,
        nodes: list[ExperimentNode],
        output_dir: Path,
        *,
        policy_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing run directory: {output_dir}"
            )
        output_dir.mkdir(parents=True)
        logger = JsonlEventLogger(output_dir / "events.jsonl")
        checkpoint_store = BestCheckpointStore(output_dir)
        started = time.monotonic()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        statuses: dict[str, str] = {}
        records: list[dict[str, Any]] = []
        logger.log(
            "run_started",
            node_count=len(nodes),
            registered_templates=list(self.registry.names),
            policy_decision=policy_decision,
        )

        unexpected_failure = False
        for node in nodes:
            blocked_by = [
                parent for parent in node.depends_on if statuses.get(parent) != "succeeded"
            ]
            if blocked_by:
                statuses[node.node_id] = "skipped_dependency"
                record = {
                    "node_id": node.node_id,
                    "status": "skipped_dependency",
                    "blocked_by": blocked_by,
                }
                records.append(record)
                logger.log("node_skipped", **record)
                continue

            node_dir = output_dir / "nodes" / node.node_id
            node_dir.mkdir(parents=True)
            atomic_json(node_dir / "node.json", node.to_dict())
            logger.log(
                "node_started",
                node_id=node.node_id,
                experiment_id=node.experiment_id,
                template=node.template,
                seed=node.seed,
            )
            node_started = time.monotonic()
            try:
                result = self.registry.execute(node, node_dir)
                if node.objective.metric not in result.metrics:
                    raise ValueError(
                        f"template omitted objective metric: {node.objective.metric}"
                    )
                score = float(result.metrics[node.objective.metric])
                promoted = checkpoint_store.consider(
                    node_id=node.node_id,
                    score=score,
                    direction=node.objective.direction,
                    checkpoint=result.checkpoint,
                )
                record = {
                    "node_id": node.node_id,
                    "experiment_id": node.experiment_id,
                    "status": "succeeded",
                    "metrics": result.metrics,
                    "objective_score": score,
                    "promoted_to_best": promoted,
                    "runtime_seconds": time.monotonic() - node_started,
                    "notes": result.notes,
                }
                atomic_json(node_dir / "result.json", record)
                statuses[node.node_id] = "succeeded"
                records.append(record)
                logger.log("node_succeeded", **record)
            except Exception as exc:  # Node errors are isolated and recorded.
                status = "failed_allowed" if node.allow_failure else "failed_unexpected"
                failure = {
                    "node_id": node.node_id,
                    "experiment_id": node.experiment_id,
                    "status": status,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "runtime_seconds": time.monotonic() - node_started,
                    "traceback": traceback.format_exc(),
                }
                atomic_json(node_dir / "failure.json", failure)
                statuses[node.node_id] = status
                records.append(failure)
                logger.log("node_failed", **failure)
                if not node.allow_failure:
                    unexpected_failure = True
                    break

        manifest = checkpoint_store.validate()
        counts = {
            name: sum(status == name for status in statuses.values())
            for name in (
                "succeeded",
                "failed_allowed",
                "failed_unexpected",
                "skipped_dependency",
            )
        }
        run_status = "failed" if unexpected_failure else (
            "completed_with_allowed_failures"
            if counts["failed_allowed"]
            else "completed"
        )
        summary = {
            "schema_version": 1,
            "status": run_status,
            "started_at_utc": started_at_utc,
            "runtime_seconds": time.monotonic() - started,
            "counts": counts,
            "best_checkpoint": manifest,
            "policy_decision": policy_decision,
            "nodes": records,
        }
        atomic_json(output_dir / "summary.json", summary)
        logger.log(
            "run_finished",
            status=run_status,
            counts=counts,
            best_checkpoint=manifest,
        )
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--priorities", default="results/research_priorities.json")
    args = parser.parse_args()

    from .policy import DeterministicResearchPolicy

    nodes = load_plan(args.plan)
    policy = DeterministicResearchPolicy(Path(args.priorities))
    decision = policy.select(limit=3)
    summary = ExperimentRunner().run(nodes, Path(args.output), policy_decision=decision)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

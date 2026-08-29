from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiment_framework.checkpoint import BestCheckpointStore
from experiment_framework.policy import DeterministicResearchPolicy
from experiment_framework.runner import ExperimentRunner
from experiment_framework.schema import ExperimentNode, SchemaError, load_plan


ROOT = Path(__file__).resolve().parents[1]


class SchemaTests(unittest.TestCase):
    def test_demo_plan_is_strict_and_valid(self) -> None:
        nodes = load_plan(str(ROOT / "configs" / "part3_demo_nodes.json"))
        self.assertEqual(len(nodes), 4)
        self.assertEqual(sum(node.allow_failure for node in nodes), 1)

    def test_unknown_node_field_is_rejected(self) -> None:
        raw = {
            "node_id": "bad-node",
            "experiment_id": "bad",
            "template": "synthetic_ranker",
            "hypothesis": "test",
            "seed": 0,
            "objective": {"metric": "score"},
            "shell_command": "unsafe",
        }
        with self.assertRaises(SchemaError):
            ExperimentNode.from_mapping(raw)


class PolicyTests(unittest.TestCase):
    def test_policy_is_deterministic_and_selects_research_order(self) -> None:
        policy = DeterministicResearchPolicy(ROOT / "results" / "research_priorities.json")
        first = policy.select(limit=3)
        second = policy.select(limit=3)
        self.assertEqual(first, second)
        self.assertEqual(
            first["selected_experiment_ids"],
            ["bpr_fm", "din_lite_history", "shared_bottom_multitask"],
        )
        self.assertFalse(first["constraints"]["hidden_test_access"])
        self.assertFalse(first["constraints"]["llm_calls"])


class CheckpointTests(unittest.TestCase):
    def test_worse_checkpoint_cannot_replace_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = BestCheckpointStore(Path(temporary))
            self.assertTrue(
                store.consider(
                    node_id="first", score=0.7, direction="maximize", checkpoint={"v": 1}
                )
            )
            before = (Path(temporary) / "best_checkpoint.json").read_bytes()
            self.assertFalse(
                store.consider(
                    node_id="worse", score=0.6, direction="maximize", checkpoint={"v": 2}
                )
            )
            self.assertEqual(before, (Path(temporary) / "best_checkpoint.json").read_bytes())
            manifest = store.validate()
            self.assertEqual(manifest["node_id"], "first")


class RunnerTests(unittest.TestCase):
    def test_three_nodes_succeed_and_allowed_failure_is_isolated(self) -> None:
        nodes = load_plan(str(ROOT / "configs" / "part3_demo_nodes.json"))
        decision = DeterministicResearchPolicy(
            ROOT / "results" / "research_priorities.json"
        ).select(limit=3)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            summary = ExperimentRunner().run(nodes, output, policy_decision=decision)
            self.assertEqual(summary["status"], "completed_with_allowed_failures")
            self.assertEqual(summary["counts"]["succeeded"], 3)
            self.assertEqual(summary["counts"]["failed_allowed"], 1)
            self.assertEqual(summary["best_checkpoint"]["node_id"], "din-lite-smoke")
            self.assertTrue(
                (output / "nodes" / "failure-safety-smoke" / "failure.json").exists()
            )
            self.assertTrue(
                (output / "nodes" / "multitask-smoke" / "result.json").exists()
            )
            checkpoint = output / "best_checkpoint.json"
            self.assertEqual(
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                summary["best_checkpoint"]["sha256"],
            )
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["sequence"] for event in events],
                list(range(1, len(events) + 1)),
            )
            self.assertEqual(len(events), 10)
            self.assertEqual(events[-1]["event"], "run_finished")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research_agent import (
    CommandLLMPolicy,
    DeterministicPolicy,
    REGISTRY,
    append_event,
    json_safe,
    sha256,
    stronger_result,
)


class ResearchAgentTests(unittest.TestCase):
    def test_deterministic_policy_only_selects_registered_experiment(self):
        policy = DeterministicPolicy()
        choice, metadata = policy.choose({}, list(REGISTRY.values()))
        self.assertIn(choice, REGISTRY)
        self.assertEqual(metadata["provider"], "deterministic")

    def test_policy_progresses_when_completed_choice_is_removed(self):
        policy = DeterministicPolicy()
        remaining = [
            value for key, value in REGISTRY.items()
            if key != "fm_seed0_control"
        ]
        choice, _ = policy.choose({}, remaining)
        self.assertEqual(choice, "fm_ensemble_4")

    def test_jsonl_event_and_hash_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            append_event(path, {"event": "test", "value": 1})
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record, {"event": "test", "value": 1})
            self.assertEqual(sha256(path), sha256(path))

    def test_numpy_metrics_are_json_serializable(self):
        value = {"primary": np.float32(0.602295), "seeds": (0, 1)}
        encoded = json.dumps(json_safe(value))
        decoded = json.loads(encoded)
        self.assertAlmostEqual(decoded["primary"], 0.602295, places=6)
        self.assertEqual(decoded["seeds"], [0, 1])

    @patch("research_agent.subprocess.run")
    def test_command_policy_accepts_provider_neutral_json(self, run):
        run.return_value.stdout = json.dumps({
            "experiment_id": "fm_seed0_control",
            "provider": "test-provider",
            "model": "test-model",
            "usage": {"tokens": 12},
        })
        policy = CommandLLMPolicy("fake-wrapper")
        choice, metadata = policy.choose({}, list(REGISTRY.values()))
        self.assertEqual(choice, "fm_seed0_control")
        self.assertEqual(metadata["provider"], "test-provider")
        self.assertEqual(metadata["usage"], {"tokens": 12})
        self.assertFalse(run.call_args.kwargs["shell"])

    def test_weaker_run_cannot_replace_global_best(self):
        incumbent = {"metrics": {"primary": 0.602295}}
        weaker = {"metrics": {"primary": 0.600000}}
        stronger = {"metrics": {"primary": 0.603000}}
        self.assertIs(stronger_result(incumbent, weaker), incumbent)
        self.assertIs(stronger_result(incumbent, stronger), stronger)


if __name__ == "__main__":
    unittest.main()

# Safe Sakana-Style Agent Report

## Outcome

A constrained autonomous experiment controller and a ranking-aware FM candidate
have been implemented against the official GAUC/nDCG@5 evaluator. The best
validation primary is **0.602295**, improving on the reproduced five-seed FM
mean of **0.601572** by **0.000723**.

This is a valid but modest improvement. It should not be presented as a score
near 0.604 or as evidence that the search is complete.

## Official Results

| Candidate | GAUC | nDCG@5 | Primary | Delta vs FM mean |
|---|---:|---:|---:|---:|
| FM seed 0 control | 0.667133 | 0.535806 | 0.601470 | -0.000103 |
| Four-seed FM ensemble | 0.668318 | 0.536103 | 0.602211 | +0.000639 |
| **Four-seed hybrid BPR ensemble** | **0.668698** | **0.535892** | **0.602295** | **+0.000723** |

The selected candidate was evaluated once on the test split after selection:

| Split | GAUC | nDCG@5 | Primary | Reproduced FM primary | Delta |
|---|---:|---:|---:|---:|---:|
| Test | 0.663485 | 0.529629 | **0.596557** | 0.594607 | **+0.001950** |

## Model

Each of four seeds trains the supplied FM with pointwise binary cross-entropy
and restores its best validation checkpoint. Training then continues with BPR:
for each positive `long_view` training impression, one negative impression from
the same user is sampled. This optimizes within-user order while staying inside
the logged-impression task. Predictions from seeds 0-3 are averaged.

No validation or test labels are used to construct training pairs. Explicit
video/author cross features were also tested and rejected after reducing seed-0
validation primary to `0.598690`.

## Agent Architecture

The controller provides:

- A fixed registry of reviewed experiment templates.
- A reproducible deterministic policy requiring no API usage.
- A provider-neutral command policy using JSON over standard input/output.
- An unchanged-evaluator SHA-256 check before and after each run.
- Validation-only autonomous selection.
- JSONL event logs, atomic model artifacts, artifact hashes, and a best manifest.
- Recovery of completed checkpoints after an interrupted orchestration run.

An LLM is optional. When used, it acts only as the experiment selector and may
choose one registered `experiment_id`; it cannot rewrite files or invent a new
evaluation protocol.

## Failure and Recovery Evidence

The first full controller pass trained and atomically saved all three model
artifacts, then hit a JSON serialization error because an evaluator metric was
a NumPy `float32`. The event log recorded the failure. Serialization was fixed
with recursive conversion to JSON-native values and covered by a regression
test. Recovery then loaded the hashed artifacts, reproduced all three
validation scores, and promoted the best checkpoint without retraining.

## Commands

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python research_agent.py --policy deterministic --budget 3
.venv/bin/python research_agent.py --budget 3 \
  --recover-run 20260829T095135Z
```

The official evaluator SHA-256 remained:

```text
c76b598fa83fe79fe33aaf554e46807d25678c1ecf5ae7633761e911e7a6e24b
```


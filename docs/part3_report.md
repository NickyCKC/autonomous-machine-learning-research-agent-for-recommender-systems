# Part 3 Sakana Experiment Framework Report

## Outcome

Part 3 passed. A deterministic, standard-library-only experiment framework now
validates versioned nodes, restricts execution to registered templates, writes
durable JSONL events, isolates failures, and prevents worse or failed nodes from
replacing the best checkpoint.

The unattended demonstration ran four nodes: three succeeded and one failed as
intended. The failure was recorded, its partial file remained isolated, the
following node still ran, and the protected checkpoint remained valid.

All demonstration scores use `validation.synthetic_proxy`. They test framework
behavior only and must not be interpreted as NDCG@10, Recall@50, or recommender
performance.

## Components Added

| Component | Evidence |
|---|---|
| Experiment-node schema | Strict Python validation plus `configs/experiment-node.schema.json` |
| Template registry | Only `synthetic_ranker` and `intentional_failure` are allowlisted in the demo registry |
| Runner | Ordered unattended execution with dependency checks and per-node directories |
| Checkpoint protection | Exclusive lock, atomic writes, SHA-256 validation, and better-only promotion |
| JSONL logging | Ten sequence-numbered, flushed, and synchronized events |
| Deterministic policy | Stable `rank_then_id_v1` selection with a SHA-256 decision fingerprint |

## Commands Run

From WSL2 Ubuntu using Python 3.12.3:

```bash
.venv/bin/python -m compileall -q experiment_framework tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m experiment_framework.runner \
  --plan configs/part3_demo_nodes.json \
  --priorities results/research_priorities.json \
  --output results/part3_demo
```

## Test Results

Five unit/integration tests pass:

1. The demo plan satisfies the strict node schema.
2. Unknown fields, including an arbitrary shell command, are rejected.
3. The research policy produces the same top-three selection every time.
4. A worse checkpoint cannot replace the current best.
5. Three nodes succeed while an allowed failure is isolated and execution
   continues.

The first test run had one failed assertion: the test expected six JSONL events
instead of the correct ten. The assertion was fixed to cover all node-start and
node-result events, and the complete suite then passed.

## Demonstration Results

| Node | Outcome | Synthetic proxy | Promoted |
|---|---|---:|---|
| `bpr-fm-smoke` | Succeeded | 0.5200688844 | Yes, initially |
| `din-lite-smoke` | Succeeded | 0.5299268728 | Yes, final best |
| `failure-safety-smoke` | Allowed failure | - | No |
| `multitask-smoke` | Succeeded after failure | 0.5250912069 | No, worse than best |

Run status: `completed_with_allowed_failures`.

The final best manifest names `din-lite-smoke` and validates the checkpoint
with SHA-256:

```text
5d870c92199c0a30d8cc0afa0ca4abeffa77fbcc1b5c4ec25b087b4bb7082c4f
```

The deterministic policy selected:

1. `bpr_fm`
2. `din_lite_history`
3. `shared_bottom_multitask`

Policy fingerprint:

```text
2ae4512fc792edf9ec5bbe2374f5318f453c8d2b4ee3641d72ffce0fb0db5000
```

## Remaining Risks and Boundaries

- The demonstration templates are synthetic; Part 4 must add real recommender
  implementations incrementally.
- The official Recall@50 candidate protocol is still missing. This prevents
  competition-comparable model selection, but did not block framework tests.
- Timeouts, retries, resume, rollback, convergence detection, and submission
  validation are intentionally deferred to Part 6.
- Provider-neutral LLM decisions and comparisons are intentionally deferred to
  Part 5.

## Acceptance Decision

All Part 3 acceptance criteria are met. No Part 4 experiment was implemented or
run.

# Part 3 Experiment Framework

## Purpose

This framework runs a bounded queue of allowlisted experiment templates and
records enough evidence to reproduce, inspect, and recover each node. It uses
only the Python standard library and does not access hidden-test data.

The included demonstration uses synthetic proxy scores solely to test the
orchestrator. They are not NDCG@10, Recall@50, model results, or evidence that
one recommender is better than another.

## Components

- `schema.py`: strict versioned node validation, dependency validation, seeds,
  hypotheses, objectives, and failure policy.
- `registry.py`: allowlisted template registry. Plan files cannot execute an
  arbitrary shell command.
- `runner.py`: unattended sequential execution, dependency checks, isolated
  node directories, atomic summaries, and continued execution after explicitly
  allowed failures.
- `events.py`: append-only JSONL events, monotonic sequence numbers, immediate
  flush, and filesystem synchronization.
- `checkpoint.py`: exclusive lock, atomic replacement, SHA-256 manifest, and
  promotion only when the objective improves.
- `policy.py`: deterministic rank-then-ID selection with fixed compute and
  leakage limits. The policy explicitly makes zero LLM calls and forbids hidden
  test access.

## Run Commands

From WSL2 in the starter-kit root:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m experiment_framework.runner \
  --plan configs/part3_demo_nodes.json \
  --priorities results/research_priorities.json \
  --output results/part3_demo
```

The output directory is deliberately write-once. The runner refuses to
overwrite an existing run. Resume and rollback behavior belongs to Part 6.

## Demonstration Acceptance Criteria

- Three synthetic experiment nodes complete without interaction.
- One intentional failure is recorded as `failed_allowed`.
- Execution continues after that failure.
- The failed node's partial artifact is isolated in its own directory.
- The failed node cannot promote a checkpoint.
- A lower-scoring later node cannot overwrite the best checkpoint.
- The checkpoint checksum validates after the run.
- Policy output selects `bpr_fm`, `din_lite_history`, and
  `shared_bottom_multitask` in the Part 2 order.

## Boundaries

Timeouts, retries, resume, rollback, convergence detection, and final
submission validation remain Part 6 work. Actual recommender templates and
loss functions remain Part 4 work. Provider-neutral LLM control remains Part 5
work.

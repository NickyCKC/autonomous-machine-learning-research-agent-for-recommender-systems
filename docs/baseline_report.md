# Track 2 Baseline Reproduction Report

## Outcome

Part 1 passed. The downloaded KuaiRand-Pure data has the expected row counts,
the random and popularity baselines match the published reference values, and
the five-seed Factorization Machine mean is effectively identical to the
official baseline.

## Environment

| Item | Value |
|---|---|
| Runtime | WSL2 Ubuntu |
| Python | 3.12.3 in `.venv` |
| NumPy | 2.5.2 |
| CPU | 12th Gen Intel Core i9-12900H, 20 logical CPUs |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB |
| GPU driver | 576.52 |
| Platform | Linux 6.18.33.2 Microsoft WSL2, x86_64 |

The baseline is CPU-based. The GPU was recorded for reproducibility but was not
used by the NumPy implementation.

## Data Integrity

The archive was downloaded from the Zenodo URL provided in the starter kit.

```text
File: KuaiRand-Pure.tar.gz
Bytes: 47,432,272
SHA-256: c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0
```

The extracted official date splits contain:

| Split | Expected rows | Observed rows | Result |
|---|---:|---:|---|
| Train | 1,141,112 | 1,141,112 | Pass |
| Validation | 124,909 | 124,909 | Pass |
| Test | 170,588 | 170,588 | Pass |

## Evaluator Integrity

`evaluate.py` was not modified during Part 1.

```text
SHA-256: c76b598fa83fe79fe33aaf554e46807d25678c1ecf5ae7633761e911e7a6e24b
```

The hash was checked before and after reproduction. The English evaluator was
also compared against the original evaluator on three synthetic edge cases,
including tied scores and all-positive/all-negative users. Outputs were
identical.

## Commands

The main commands, executed from the English starter-kit directory in WSL2,
were:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.2

curl -fL --retry 3 \
  --output KuaiRand-Pure.tar.gz \
  https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
sha256sum KuaiRand-Pure.tar.gz
tar -xzf KuaiRand-Pure.tar.gz

.venv/bin/python scripts/reproduce_baselines.py
.venv/bin/python scripts/reproduce_baselines.py --resume
```

The second runner command resumed after a serialization problem described
under Recovery Evidence. Completed random and popularity runs were not
repeated.

## Results

### Aggregate metrics

| Model | Runs | Validation GAUC | Validation nDCG@5 | Validation primary | Test GAUC | Test nDCG@5 | Test primary |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 5 | 0.499325 | 0.467452 | 0.483388 | 0.499579 | 0.451066 | 0.475322 |
| Popularity | 1 | 0.638726 | 0.522718 | 0.580722 | 0.630801 | 0.512117 | 0.571459 |
| FM | 5 | 0.667400 | 0.535744 | 0.601572 | 0.661015 | 0.528198 | 0.594607 |

### FM seed results

| Seed | Validation primary | Test primary | Runtime (seconds) |
|---:|---:|---:|---:|
| 0 | 0.601470 | 0.595340 | 25.51 |
| 1 | 0.601761 | 0.594769 | 19.87 |
| 2 | 0.601090 | 0.594820 | 25.71 |
| 3 | 0.601503 | 0.594972 | 24.23 |
| 4 | 0.602037 | 0.593131 | 28.84 |
| **Mean** | **0.601572** | **0.594607** | **24.83** |
| **Population standard deviation** | **0.000316** | **0.000764** | - |

The five FM runs used 124.16 seconds of model runtime in total. The complete
process, including interruption, resume, data loading, and orchestration, took
229.76 seconds.

## Acceptance Checks

| Check | Observed | Required | Result |
|---|---:|---:|---|
| Random test primary | 0.475322 | 0.4753 +/- 0.002 | Pass |
| FM test primary | 0.594607 | 0.5946 +/- 0.003 | Pass |
| Dataset row counts | Exact match | Exact match | Pass |
| Evaluator unchanged | Same SHA-256 | Same SHA-256 | Pass |

FM differs from the published primary score by only `+0.0000065`, well within
the expected seed variation.

## Recovery Evidence

The first FM seed trained successfully, but the initial reproducibility harness
could not serialize NumPy `float32` metrics to JSON. The harness was corrected
to normalize NumPy scalars and was given explicit resume support. It then:

1. Loaded the six completed random/popularity records.
2. Verified the saved evaluator hash.
3. Re-ran only the uncommitted FM seed and remaining seeds.
4. Checkpointed every subsequent run successfully.

This was a harness issue, not a model or dataset failure.

## Official Metric Definition

The official Track 2 evaluation uses:

- Positive label: `long_view`.
- Metrics: GAUC and nDCG@5.
- Primary score: their arithmetic mean.

The supplied `evaluate.py` implements these metrics and was preserved unchanged.
All future experiments must be compared using this evaluator and the official
date splits.

## Artifacts

- `results/baseline_results.json`: machine-readable environment, integrity,
  per-run metrics, aggregates, and acceptance checks.
- `results/baseline_run.log`: complete console record, including the recovered
  serialization error and all training epochs.
- `scripts/reproduce_baselines.py`: resumable reproduction harness.
- `requirements-baseline.txt`: pinned initial dependency.

## Part 1 Decision

The baseline reproduction gate is complete and passed. These results provide
the verified reference point for future improvements.

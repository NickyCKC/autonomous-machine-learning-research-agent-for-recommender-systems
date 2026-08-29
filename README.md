# TikTok TechJam 2026 Track 2 — KuaiRand-Pure

English workspace for reproducing and improving the official recommender-system
baseline.

## Official Evaluation

| Setting | Definition |
|---|---|
| Task | Rank each user's logged evaluation impressions |
| Positive label | `long_view` |
| Metrics | GAUC and nDCG@5 |
| Primary score | `(GAUC + nDCG@5) / 2` |

The bundled `evaluate.py` is authoritative and remains unchanged. Model
development must use the official date splits and must not use test labels for
training or model selection.

## Requirements and Data

Python 3.12 and NumPy are sufficient for the supplied baselines.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-baseline.txt

wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

Expected archive SHA-256:

```text
c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0
```

## Reproduce the Baseline

Run one supplied model directly:

```bash
.venv/bin/python baseline.py --model fm
```

Run the complete recorded reproduction:

```bash
.venv/bin/python scripts/reproduce_baselines.py
```

The five-seed FM reproduction achieved:

| Split | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Validation | 0.667400 | 0.535744 | **0.601572** |
| Test | 0.661015 | 0.528198 | **0.594607** |

These results reproduce the published baselines of approximately `0.6016` on
validation and `0.5946` on test. They establish a verified starting point; they
are not an improvement over the baseline.

## Preserved Files

| File | Purpose |
|---|---|
| `baseline.py` | Supplied random, popularity, and FM models |
| `data.py` | Supplied data loading and official date splits |
| `evaluate.py` | Supplied GAUC/nDCG@5 evaluator; do not modify |
| `submit.py` | Supplied submission generator and validator |
| `baseline_scores.json` | Published baseline references |
| `scripts/reproduce_baselines.py` | Five-seed reproduction harness |
| `results/baseline_results.json` | Machine-readable reproduced results |
| `docs/baseline_report.md` | Full English reproduction report |


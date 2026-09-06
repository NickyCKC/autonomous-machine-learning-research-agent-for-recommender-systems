# TikTok TechJam 2026 Track 2 — KuaiRand-Pure

English workspace for reproducing and improving the official recommender-system
baseline.

> The separate Track 1 conversational shopping project is available at
> [tiktok-techjam-2026-track1](https://github.com/NickyCKC/tiktok-techjam-2026-track1).

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

## Current Improved Result

The best validated candidate is a four-seed ensemble of FMs trained first with
the starter pointwise loss and then continued with within-user BPR pairs. All
pairs come from logged training impressions; model selection uses validation
only and the official evaluator remains unchanged.

| Model | Validation GAUC | Validation nDCG@5 | Validation primary |
|---|---:|---:|---:|
| Reproduced FM mean | 0.667400 | 0.535744 | 0.601572 |
| Four-seed FM ensemble | 0.668318 | 0.536103 | 0.602211 |
| **Four-seed hybrid BPR ensemble** | **0.668698** | **0.535892** | **0.602295** |

The selected candidate's one-time test score is `0.596557`, compared with the
reproduced FM test mean of `0.594607`.

Run the selected model directly:

```bash
.venv/bin/python baseline.py --model hybrid-ensemble \
  --ensemble-seeds 0,1,2,3 --bpr-epochs 6 --bpr-lr 0.0002
```

## Safe Research Agent

`research_agent.py` implements a small Sakana-style experiment loop. It checks
the official evaluator hash, lets a policy select only reviewed experiment
templates, logs every decision and result, saves atomic hashed checkpoints, and
can recover completed artifacts after interruption. Its eight current choices
cover the baseline, ensembles, FM size, BPR strength, and negative sampling.

Offline deterministic run:

```bash
.venv/bin/python research_agent.py --policy deterministic --budget 3
```

Provider-neutral LLM run:

```bash
.venv/bin/python research_agent.py --policy command \
  --policy-command "your-json-llm-wrapper"
```

The wrapper receives experiment context as JSON on standard input and returns
an allowed `experiment_id` as JSON. It may call GPT, Claude, a local model, or
another provider. The LLM never receives permission to modify the evaluator or
execute arbitrary experiment code.

A live three-decision GPT-5.4 Mini comparison has been completed. The LLM and
deterministic policies both found the same best screen (`0.601499`) within equal
budgets, so the overall `0.602295` checkpoint remains selected. The successful
LLM run used 3,412 tokens at an estimated decision cost of about `$0.005`.

## Preserved Files

| File | Purpose |
|---|---|
| `baseline.py` | Supplied random, popularity, and FM models |
| `data.py` | Supplied data loading and official date splits |
| `evaluate.py` | Supplied GAUC/nDCG@5 evaluator; do not modify |
| `submit.py` | Supplied submission generator and validator |
| `baseline_scores.json` | Published baseline references |
| `scripts/reproduce_baselines.py` | Five-seed reproduction harness |
| `research_agent.py` | Safe deterministic/LLM research controller |
| `results/baseline_results.json` | Machine-readable reproduced results |
| `results/official_agent/best.json` | Selected improved checkpoint manifest |
| `results/official_agent/live_llm_comparison.json` | Equal-budget policy comparison |
| `docs/baseline_report.md` | Full English reproduction report |
| `docs/sakana_agent_report.md` | Improved-model and agent evidence |

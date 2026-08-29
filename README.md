# KuaiRand-Pure Starter Kit

> **Competition metric notice:** the official Track 2 PDF defines `is_click`,
> NDCG@10, and Recall@50 as authoritative. The bundled `long_view`, GAUC, and
> nDCG@5 evaluator below is preserved only to reproduce the legacy starter
> baseline. Do not use its primary score to select the final competition model.
> See `docs/recommender_research.md` for the metric decision and unresolved
> Recall@50 candidate-set requirement.

## Current Project Status

- Part 1: legacy starter baseline reproduced over five seeds.
- Part 2: recommender research and experiment priorities completed.
- Part 3: deterministic experiment framework completed and failure-tested.
- Part 4 result 1: catalogue-negative BPR improved both reconstructed official
  metrics over the matched pointwise baseline across seeds 0-4.
- Part 4 result 2: DIN-lite history reranking was tested over seeds 0-4 and
  rejected after a small negative paired result; BPR seed 3 remains best.

See `docs/official_evaluation_protocol.md` for the candidate reconstruction and
`docs/part4_bpr_report.md` and `docs/part4_din_report.md` for the completed
Part 4 family results. Work is currently stopped before multi-task learning at
the required result gate.

## Requirements

Python 3.9+ and NumPy. **Nothing else is required.** You do not need PyTorch,
pandas, or scikit-learn.

## Data

Download the dataset from [KuaiRand](https://kuairand.com) using the direct
Zenodo link; no registration is required:

```bash
# Run from this Starter Kit directory. Extraction creates ./KuaiRand-Pure/.
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running the Baselines

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`. Specify it explicitly if the
dataset is stored elsewhere.

`--model` accepts `fm` (legacy published baseline), `pop` (trivial baseline), or
`random` (lower bound used to verify the evaluator). The full FM run takes
approximately 40 seconds on one CPU core.

## Legacy Starter Evaluator Definition

| Setting | Definition |
|---|---|
| Task | **Within-user ranking** - rank only each user's logged evaluation impressions; do not perform full-catalog retrieval |
| Relevance label | `long_view` (native binary column, 0/1) |
| Metrics | `GAUC` and `nDCG@5`; **primary score = their mean** |
| Data splits | train `20220408-20220421` / valid `20220422-20220428` / test `20220429-20220508` |
| Users with zero positives | Include nDCG as 0.0 in the mean; GAUC includes only users with `0 < positives < impressions`, weighted by positive count |
| nDCG gain | `2^rel - 1`, equivalent to identity for a binary label |

See `evaluate.py` for the implementation. All evaluation conventions are
documented in its module header.

## Baseline Ladder

The scores below reproduce the legacy starter evaluator on its public test
split. **They are sanity-check values, not the official Track 2 metrics.**

| Model | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Random (lower-bound sanity check) | 0.4996 | 0.4511 | 0.4753 |
| Item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (legacy baseline)** | **0.6610** | **0.5282** | **0.5946** |

### Important: the actual nDCG@5 ceiling is 0.729, not 1.0

Among the 23,875 test users:

| User group | Percentage | Effect on metrics |
|---|---:|---|
| All negative (none of the user's impressions is a `long_view`) | **27.1%** | nDCG is always **0** and cannot be improved by any model; excluded from GAUC |
| All positive | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Discriminative users | **63.7%** | The effective population for GAUC |

Even an oracle that uses the true label as its prediction score can achieve
only the following:

| Metric | Random | FM baseline | **Oracle ceiling** | Fraction of usable range captured by FM |
|---|---:|---:|---:|---:|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **Primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

Assess progress relative to the oracle ceiling. Seeing a primary score of
0.5946 and assuming that it is far from a maximum of 1.0 is misleading. The
remaining headroom is approximately 0.27, not 0.41.

The FM standard deviation over five random seeds is **0.0008**. Therefore, the
convergence rule is **epsilon = 0.002 (about 2.5 standard deviations), N = 3**:
declare convergence when the validation primary score has not improved by more
than 0.002 for three consecutive iterations.

> Sanity check: if `--model random` does not produce a primary score near
> 0.475 (+/-0.001), the evaluation harness is incorrect and should be fixed
> before further experimentation.

## Submission Format

The submission is a CSV with a header and one row for every evaluation row:

```csv
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
```

| Field | Description |
|---|---|
| `row_id` | Contiguous zero-based index corresponding to the row order in `data.load()[split]`. Deterministic order: read `log_standard_4_08_to_4_21_pure.csv`, then `log_standard_4_22_to_5_08_pure.csv`, filter by date, and preserve source-file order. |
| `user_id` / `video_id` | Redundant fields used only to validate alignment. |
| `score` | Any finite real-valued score produced by your model. Only relative ordering matters; NaN and Inf are rejected. |

`row_id` is required because `(user_id, video_id)` is **not unique** in the
evaluation set. In the test set, 3.06% of pairs repeat, with some pairs
appearing up to 12 times, so the pair cannot serve as a key.

Generate and validate submissions with:

```bash
python3 submit.py --make  --split test  submission.csv
python3 submit.py --check --split test  submission.csv
python3 submit.py --score --split valid submission.csv
```

`--check` rejects an incorrect header, row-count mismatch, gaps in `row_id`,
misaligned `user_id`/`video_id`, non-numeric scores, NaN, and Inf. Always run
`--check` before submission.

## Where to Start Improving

The guidance below is based on organizer experiments rather than speculation.
Known unproductive directions are listed so participants do not waste time
repeating them.

### Tested directions with no measurable gain

| Experiment | Result |
|---|---|
| **Add static features** - add all 13 CWM fields, including `music_id`, `video_type`, `upload_type`, and six coarse user-side buckets | Primary **0.5940** versus **0.5950** with five fields: no meaningful difference and a slight decrease |
| **Increase model capacity** - embedding dimensions k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887: almost unchanged |

The likely reason is that the `user_id x video_id` interaction already captures
most of the learnable signal. Coarse buckets such as
`follow_user_num_range` are redundant beside `user_id`, and 1.14 million rows
may not support substantially greater capacity. **The bottleneck does not
appear to be static features or embedding capacity.**

Also note that a first-order term containing only user-side information always
contributes zero to within-user ranking because it is constant for all items
shown to that user. User-side features can help only through interactions with
item-side features.

### Unexplored directions where headroom may exist

The organizers list these in their estimated order of promise. They have not
tested these directions; they are intentionally left for participants.

1. **Change the loss function.** The baseline uses pointwise log loss, while
   NDCG@10 and Recall@50 are ranking metrics. Align training with evaluation through a
   pairwise loss such as BPR or a listwise loss such as a per-user softmax. The
   organizers consider this the most promising direction.

2. **Model user history sequences.** The current features ignore behavioral
   sequences. Each KuaiRand user has hundreds or thousands of training
   interactions, leaving approaches such as DIN or SIM unexplored.

3. **Use multiple objectives.** The logs include `is_click`, `is_like`,
   `is_follow`, `is_comment`, `is_forward`, and `play_time_ms`. These can serve
   as auxiliary tasks supporting the official `is_click` objective.

4. **Model watch duration.** CWM models watch duration using censored
   regression. When a video finishes, the true preferred viewing duration may
   be censored, so a one-sided loss may be more appropriate than squared error.
   This is a research-rich direction.

5. **Change the model.** Candidates include DeepFM, DCN, and xDeepFM. Since
   capacity did not appear to be the bottleneck, prioritize directions 1-4
   before changing model families.

6. **Model time and distribution shift.** Explore `hourmin`, `date`, and the
   change in data distribution between training and evaluation periods.

7. **Use unbiased validation (advanced).**
   `log_random_4_22_to_5_08_pure.csv` contains 1.18 million randomly exposed
   interactions. It can provide an additional unbiased validation set to test
   whether a model merely overfits biased production traffic.

## Using Your Own Model with the Legacy Evaluator

`evaluate.py` is fully decoupled from the model and requires only three arrays
of equal length:

```python
from evaluate import evaluate

print(evaluate(user_ids, labels, scores))
```

- `user_ids`: the `user_id` for every evaluation row.
- `labels`: the row's binary `long_view` value for legacy reproduction only.
- `scores`: any real-valued score produced by your model; only ordering matters.

You may use this interface to reproduce the old scores, but final Track 2
selection requires `is_click`, NDCG@10, Recall@50, and the organizer's
candidate-set protocol. The supplied `evaluate.py` does not define that
official protocol.

CWM requires `torch==1.6.0`, an old 2020 release that may not install on modern
GPUs. Its loss optimizes counterfactual watch time, and it evaluates against a
reconstructed `long_view2` label. Treat it as an advanced research reference,
not as the starting point.

## Files

| File | Purpose |
|---|---|
| `evaluate.py` | Preserved legacy metric implementation. **Do not modify.** |
| `data.py` | Legacy data loading, date splits, and feature encoding. |
| `baseline.py` | Random, popularity, and FM legacy baselines. |
| `baseline_scores.json` | Legacy scores, seed variance, and convergence parameters. |
| `submit.py` | Generate and validate submission files. |
| `ablation_features.py` | Reproduce the experiment showing that additional static features did not help. |

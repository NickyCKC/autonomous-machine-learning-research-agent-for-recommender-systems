# Part 4 Result 1: Pairwise BPR Loss

## Outcome

The first Part 4 loss-family experiment passed. Pairwise BPR with full-catalogue
negative sampling improved both required click-ranking metrics over a matched
pointwise Factorization Machine baseline across fixed seeds 0-4.

This result uses the documented `full_catalogue_v1` reconstruction: 17,965
validation users with clicks rank all 7,583 known videos, with prior training
clicks filtered unless they are validation positives. Public-test rows were
discarded without being retained or scored, and hidden-test data was not
accessed.

## Matched Models

Both models use the same 16-factor FM architecture and the same fields:

- `user_id`
- `video_id`
- `author_id`
- catalogue-derived duration bucket

The pointwise baseline minimizes binary click log loss over logged impressions.
BPR pairs every training click with a randomly sampled catalogue item that the
same user did not click in training. The seed-0 pilot froze four pointwise
epochs because performance degraded by epoch eight, and eight BPR epochs
because Recall@50 continued improving. Those budgets were then held constant
for seeds 0-4.

## Five-Seed Results

| Model | NDCG@10 mean | NDCG std | Recall@50 mean | Recall std | Combined mean | Combined std |
|---|---:|---:|---:|---:|---:|---:|
| Pointwise FM | 0.00015972 | 0.00001459 | 0.00187760 | 0.00019052 | 0.00101866 | 0.00009918 |
| BPR-FM | 0.02615600 | 0.00097600 | 0.12031057 | 0.00128915 | 0.07323329 | 0.00089996 |
| Absolute improvement | +0.02599629 | - | +0.11843297 | - | +0.07221463 | - |

Population standard deviation is reported. BPR improved both metrics in every
seed, not only in the aggregate.

The gain is unusually large because logged-impression pointwise training does
not constrain most of the 7,583-video retrieval catalogue. BPR's catalogue
negative sampling directly teaches the model to separate clicked videos from
unseen catalogue items. This is a strong internal result under the reconstructed
protocol, but it is not presented as a hidden-leaderboard score.

## Best Checkpoint

Seed 3 produced the best combined validation score:

- NDCG@10: `0.0274122829`
- Recall@50: `0.1212044848`
- Combined: `0.0743083838`
- Node: `bpr-seed-3`

The framework preserved its compressed model artifact and verified its
SHA-256 against the checkpoint metadata. Later lower-scoring nodes did not
replace it.

## Commands

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.run_part4_loss_experiment \
  --plan configs/part4_bpr_nodes.json \
  --output results/part4_bpr
.venv/bin/python -m scripts.summarize_part4 \
  --run-dir results/part4_bpr \
  --output results/part4_bpr_comparison.json
```

The initial direct-file launch failed before execution because the repository
root was absent from Python's import path. Relaunching the same entry point as a
module fixed the import without creating or overwriting run artifacts.

## Remaining Risks

- The organizers have not supplied their candidate list, so leaderboard scores
  may use a different candidate universe.
- The pointwise baseline is weak in full-catalogue retrieval. A sampled-softmax
  baseline would be a useful future diagnostic, but it is a different loss
  family and does not change the BPR-versus-starter comparison.
- Seed 0 influenced the frozen epoch budgets. No further hyperparameter search
  should use the same validation result without recording the additional
  selection pressure.
- DIN-lite and multi-task learning have not started.

## Stage Gate

The BPR loss result meets the first Part 4 acceptance gate. Work is stopped
before the DIN-lite model family, as required by the incremental notification
rule.

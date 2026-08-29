# Part 4 Result 2: DIN-Lite History Reranking

## Outcome

The DIN-lite family was implemented and evaluated successfully, but its
hypothesis was rejected. A small recency-weighted training-click history signal
did not improve BPR under the same `full_catalogue_v1` click protocol.

The existing BPR seed-3 checkpoint remains the global best. The protected
reference was loaded and checksum-verified before DIN-lite ran, and none of the
five history nodes replaced it.

## Model

DIN-lite starts from each seed's trained BPR-FM and uses only clicks dated in
the training split. Histories are sorted by timestamp, repeated videos are
deduplicated, and the 50 most recent unique videos are retained.

The candidate-conditioned history score combines:

- cosine similarity between the candidate embedding and a recency-weighted
  click-interest vector;
- recency-weighted candidate-author matches;
- recency-weighted duration-bucket matches.

This is a lightweight DIN-inspired reranker, not a full deep DIN network. It
was chosen to fit the 72-hour and 6 GB GPU constraints while providing a
strictly leakage-controlled test of the history hypothesis.

## Pilot Decision

Seed 0 was used to freeze one global weight. We tested `0.25`, `0.5`, `1.0`,
and `2.0`; all degraded monotonically. We then bounded the search with `0.025`,
`0.05`, and `0.1`, plus an embedding-only `0.1` variant. None beat the seed-0
BPR score. Weight `0.025` was least harmful and was frozen for seeds 0-4.

No test data was used during this selection.

## Five-Seed Paired Results

| Metric | BPR mean | DIN-lite mean | DIN std | Paired mean delta | Seeds improved |
|---|---:|---:|---:|---:|---:|
| NDCG@10 | 0.02615600 | 0.02612065 | 0.00099412 | -0.00003535 | 1/5 |
| Recall@50 | 0.12031057 | 0.12023179 | 0.00125287 | -0.00007878 | 1/5 |
| Combined | 0.07323329 | 0.07317622 | 0.00089016 | -0.00005707 | 1/5 |

The effect is small relative to seed variation, but it is negative on both
metrics and not sufficiently consistent to keep. The correct decision is to
retain BPR rather than add complexity with no demonstrated gain.

## Likely Explanation

BPR already learns user-item affinity from the same training clicks. The
history reranker therefore adds mostly redundant information. In addition,
full-catalogue evaluation filters previously clicked videos, so emphasizing
near-duplicates of recent behavior can reduce discovery of held-out clicked
items.

## Commands and Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.run_part4_din_experiment \
  --plan configs/part4_din_nodes.json \
  --output results/part4_din
.venv/bin/python -m scripts.summarize_part4_din \
  --bpr-summary results/part4_bpr/summary.json \
  --din-summary results/part4_din/summary.json \
  --output results/part4_din_comparison.json
```

- Five DIN-lite seeds succeeded.
- One verified BPR reference succeeded.
- No node failed or was skipped.
- Histories contain training clicks only.
- The global best remained BPR seed 3 at combined `0.0743083838`.

## Stage Gate

The DIN-lite family result is complete. The configuration is rejected and no
multi-task model has been started.

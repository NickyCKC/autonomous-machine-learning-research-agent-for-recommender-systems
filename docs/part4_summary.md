# Part 4 Pipeline Improvement Summary

## Completed Families

| Family | Decision | Main evidence |
|---|---|---|
| Catalogue-negative BPR-FM | Keep | Improved both required metrics over pointwise FM across all five seeds |
| DIN-lite history reranker | Reject | Combined paired delta `-0.00005707`; only 1/5 seeds improved |
| Shared-bottom multi-task | Reject | Combined paired delta `-0.00001837`; only 1/5 seeds improved |
| One-epoch BPR continuation | Keep | Produced new best combined score `0.07506735` |

## Selected Pipeline

The Part 4 selected model is the catalogue-negative BPR-FM with one additional
continuation epoch, seed 2. It uses click as the positive signal and scores the
full 7,583-video catalogue.

| Metric | Selected checkpoint |
|---|---:|
| NDCG@10 | 0.02699737 |
| Recall@50 | 0.12313733 |
| Combined | 0.07506735 |

Original BPR seed 3 remains the NDCG-leading alternative at `0.02741228`.

## Part 4 Gate

The three experiments prioritized in Part 2 have been implemented and compared
with fixed seeds. The best validation checkpoint is preserved, negative results
are documented, and no Part 5 LLM integration work has started.

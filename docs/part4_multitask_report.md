# Part 4 Result 3: Shared-Bottom Multi-Task Learning

## Outcome

The shared-bottom multi-task experiment completed successfully, but the
auxiliary-treatment hypothesis was rejected. Against a matched BPR-only
continuation, multi-task learning was neutral on Recall@50 and slightly worse
on NDCG@10 and the equal-weight combined score.

The matched control also found a new combined-score best: continued BPR seed 2
at `0.0750673492`.

## Design

Every pair starts from the same seed's eight-epoch BPR checkpoint. Both members
receive one identical catalogue-negative BPR continuation epoch. The treatment
also updates shared FM embeddings through four task-specific auxiliary heads:

- long-view;
- profile entry;
- like;
- normalized watch ratio, clipped to `[0, 1]`.

Click BPR remains the primary objective and the only inference head. Auxiliary
labels come exclusively from training rows. The control's auxiliary strength is
zero; the treatment uses the frozen pilot strength `0.1` on 25% of training
rows.

## Pilot

Seed 0 tested strengths `1`, `10`, and `50`; all caused clear degradation. A
bounded follow-up tested `0.1`, `0.25`, and `0.5`. None beat the identical
strength-zero control, so `0.1` was frozen as the least invasive treatment.

## Five-Seed Paired Results

| Metric | Control mean | Multi-task mean | Treatment std | Paired mean delta | Seeds improved |
|---|---:|---:|---:|---:|---:|
| NDCG@10 | 0.02683278 | 0.02679543 | 0.00074527 | -0.00003735 | 1/5 |
| Recall@50 | 0.12016679 | 0.12016740 | 0.00162726 | +0.00000061 | 3/5 |
| Combined | 0.07349979 | 0.07348142 | 0.00104426 | -0.00001837 | 1/5 |

The Recall change is effectively zero. The NDCG decrease makes the combined
effect negative, so adding the auxiliary heads is not justified.

## Final Combined Best

The BPR-only seed-2 continuation produced:

- NDCG@10: `0.0269973702`
- Recall@50: `0.1231373282`
- Combined: `0.0750673492`

It improves the previous best combined score by `+0.0007589654`, driven by
higher Recall. Its model artifact is checksum-protected in
`results/part4_multitask/nodes/control-seed-2/model.npz`.

The previous BPR seed-3 checkpoint is retained separately because it has the
higher NDCG@10 (`0.0274122829`). These two checkpoints form the useful
validation trade-off for later robustness testing.

## Verification

- Eleven nodes succeeded: five controls, five treatments, and one protected
  reference.
- No node failed or was skipped.
- Tests cover auxiliary updates to shared embeddings and task heads.
- No public-test or hidden-test data was used.
- The final combined-best model checksum was validated.

## Decision

Reject the current shared-bottom auxiliary configuration. Adopt continued BPR
seed 2 as the combined-score checkpoint and retain original BPR seed 3 as the
NDCG-leading checkpoint.

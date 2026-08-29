# Team Progress Update

Part 1 reproduced the supplied starter baseline successfully. The five-seed FM
mean was `0.594607` on the starter kit's old primary metric. This number is only
a reproducibility sanity check because that evaluator uses `long_view`, GAUC,
and nDCG@5.

The official Track 2 task is click ranking (`is_click`) evaluated with NDCG@10
and Recall@50. We will use those official definitions for all improvements.

Research is complete, and the recommended order is:

1. Pairwise BPR-FM to align training with ranking.
2. DIN-lite using strictly chronological, candidate-aware user history.
3. Shared-bottom multi-task learning with click as primary and long-view,
   profile-entry, like, and normalized watch time as auxiliaries.

The immediate blocker is the candidate-set definition for Recall@50 and an
updated official evaluator. In the provided public validation data, 22,362 of
22,377 users have 50 or fewer logged rows, so Recall@50 is nearly trivial if it
is calculated only over those rows. We need the organizer's evaluator or exact
candidate-generation protocol before comparing model scores.

No improvement implementation has started. If anyone has a newer Track 2 kit
or evaluator, please share it. Once the metric protocol is locked, Nicky's
recommended first experiment is BPR-FM because it is the fastest, lowest-risk
test of objective alignment.

# Track 2 Recommender Research Brief

## Decision Summary

The competition task is click ranking on KuaiRand-Pure. The authoritative
metrics are NDCG@10 and Recall@50. The recommended experiment order is:

1. Pairwise BPR Factorization Machine (FM), followed by an optional
   NDCG-weighted pair variant.
2. DIN-lite candidate-aware chronological history.
3. Shared-bottom multi-task learning, with click as the primary task.

This sequence begins with the smallest objective-alignment change, then adds
personalized behavior history, and only then adds auxiliary feedback. It fits a
72-hour hackathon and the available 6 GB GPU better than beginning with a large
Transformer or an elaborate agent system.

No improvement experiment should be scored until the official candidate-set
and Recall@50 protocol is available or reproduced faithfully. The current
starter evaluator measures `long_view`, GAUC, and nDCG@5 and is only a legacy
sanity harness.

## Official Task Interpretation

- Dataset: KuaiRand-Pure.
- Primary relevance label: `is_click` (`click = positive`).
- Metrics: NDCG@10 and Recall@50.
- Final score: equal-weight average of the absolute improvements in the two
  metrics over the organizer baseline.
- Development data: train and validation only. Hidden test data must not be
  inspected or used for model selection.
- Required evidence: hypotheses, code changes, metrics, failures, recovery,
  runtime, token use, and GPU use.

This interpretation follows the official Track 2 PDF and supersedes the metric
wording in the meeting transcript and the legacy starter evaluator.

## Dataset Audit Relevant to Modeling

| Split | Rows | Users | Items | Click rate | Rows/user mean | Rows/user median | Maximum rows/user |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,141,112 | 26,210 | 7,538 | 0.463447 | 43.537 | 31 | 809 |
| Validation | 124,909 | 22,377 | 5,951 | 0.443827 | 5.582 | 4 | 74 |
| Public test | 170,588 | 23,875 | 5,982 | 0.445864 | 7.145 | 5 | 109 |

Useful training columns include click, like, follow, comment, forward, hate,
long-view, play duration, video duration, profile/comment dwell time, profile
entry, random-exposure status, timestamp fields, user ID, and video ID.

The train split is dense enough for compact ID embeddings, but the public
validation slice is shallow per user. In validation, 22,362 of 22,377 users
have at most 50 logged rows; in the public test, 23,807 of 23,875 do. Therefore,
Recall@50 would be nearly trivial if candidates were restricted to each user's
logged rows. This is strong evidence that an additional candidate-generation or
full-catalog protocol is necessary. It is an inference from the supplied data,
not a replacement for organizer instructions.

## Concepts and Research Findings

### 1. Ranking loss must match ranking evaluation

The legacy FM minimizes pointwise binary cross-entropy: it asks whether each
row is positive in isolation. NDCG and Recall instead care about the order of
items for each user. Bayesian Personalized Ranking (BPR) trains on a positive
item versus a sampled negative item and directly encourages the positive to
rank higher. LambdaRank-style weighting can give more weight to swaps that
would change NDCG near the top of the list.

Practical implication: retain the simple FM scorer at first, but replace its
loss and sampling scheme. This isolates the benefit of objective alignment
without confounding it with a larger architecture.

### 2. Sequential behavior provides intent

User and item IDs capture long-term affinity but not the user's current intent.
DIN uses candidate-conditioned attention over behavior history: the same
history is summarized differently for each candidate item. SASRec uses
self-attention to learn both short- and long-range sequential patterns.

Practical implication: start with a small DIN-like history encoder because it
is cheaper and easier to audit than a full sequential Transformer. Every
history must be constructed strictly from events earlier than the prediction
timestamp.

### 3. Auxiliary feedback can improve click representations

KuaiRand-Pure contains multiple feedback signals. Shared-bottom multi-task
learning can learn a common representation and separate output heads for
click, long-view, profile entry, like, and normalized watch time. MMoE is a
follow-up only if a shared bottom exhibits negative transfer; its task-specific
gates can model differing task relationships. ESMM demonstrates how multi-task
learning can address sparse post-click outcomes, although its exact conversion
setup should not be copied blindly here.

Practical implication: click remains the only primary prediction. Auxiliary
heads regularize the representation; they do not redefine relevance or the
competition score.

### 4. Watch duration is biased by video duration

Raw play time is truncated by content duration and should not be treated as a
clean preference label. Counterfactual Watch Model (CWM) explicitly addresses
this duration bias and reports experiments on KuaiRand.

Practical implication: use normalized or counterfactually adjusted watch time
only as an auxiliary target. Do not replace click with watch time.

### 5. Logged recommendations contain exposure bias

KuaiRand includes random-exposure data, making it useful for diagnosing or
correcting policy bias. Counterfactual learning-to-rank methods use propensity
weighting to estimate less-biased ranking risk. However, unstable inverse
weights and unclear exposure policies can make a rushed implementation worse.

Practical implication: treat debiasing as a high-novelty follow-up. Clip
weights, compare random versus standard traffic, and never mix future dates
into training features.

## Ranked Experiment Portfolio

Scores below are planning judgments on a 1-5 scale. Expected improvement is
higher-is-better; effort, compute, and leakage risk are lower-is-better.

| Rank | Experiment | Expected improvement | Effort | Compute | Leakage risk | Estimated implementation |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Pairwise BPR-FM | 4 | 2 | 1 | 1 | 4-6 hours |
| 2 | DIN-lite chronological history | 4 | 3 | 2 | 3 | 8-12 hours |
| 3 | Shared-bottom multi-task model | 3 | 3 | 2 | 2 | 6-10 hours |
| 4 | Random-exposure debiasing | 3 | 4 | 2 | 4 | 8-12 hours |
| 5 | CWM watch-time auxiliary | 3 | 4 | 3 | 3 | 12-18 hours |
| 6 | DeepFM capacity increase | 2 | 3 | 2 | 1 | 4-8 hours |

### Experiment 1: Pairwise BPR-FM

- Hypothesis: pairwise click ranking will improve top-of-list ordering over the
  pointwise legacy loss.
- Treatment: for each clicked item, sample unclicked items for the same user and
  timestamp/date context; optimize the positive-negative score difference.
- Optional child: weight pairs by their estimated NDCG@10 swap impact.
- Controls: identical ID mapping, feature availability, data split, seeds, and
  embedding dimension.
- Acceptance: positive official combined delta over the official baseline
  across at least three fixed seeds, with per-metric results reported.

### Experiment 2: DIN-lite Chronological History

- Hypothesis: candidate-aware recent behavior will capture intent missing from
  static user embeddings.
- Treatment: attend over a bounded sequence of prior clicked or engaged video
  embeddings conditioned on the candidate.
- Controls: compare against BPR-FM with the same candidate protocol and seeds.
- Leakage protection: sort by event time and exclude the current and all future
  interactions from its history; validation histories may use only data
  available before the validation event.

### Experiment 3: Shared-Bottom Multi-Task Model

- Hypothesis: dense related feedback improves embeddings and click ranking.
- Treatment: shared embeddings/MLP with a primary click head and auxiliary
  long-view, profile-entry, like, and normalized-watch-time heads.
- Controls: tune auxiliary loss weights only on validation; report click-only
  ranking metrics.
- Failure rule: if auxiliary tasks reduce both official click metrics, remove
  weak heads or test MMoE before increasing model size.

### Later Experiments

- Random-exposure debiasing: compare traffic subsets, estimate/clamp propensity
  weights, and document variance.
- CWM auxiliary: debias watch duration before adding it as a secondary task.
- DeepFM: test only after loss, history, and auxiliary-signal changes because a
  larger interaction network is less likely to be the first bottleneck.

## Evaluation Protocol Required for Part 4

Before model improvements are accepted, implement or obtain one evaluator with:

1. A documented candidate set for every user/query.
2. `is_click` as binary relevance.
3. Deterministic tie handling.
4. NDCG@10 with an explicit treatment of users with no positives.
5. Recall@50 with an explicit denominator and candidate universe.
6. A frozen train/validation split and seeds 0-4 where compute permits.
7. Per-seed metrics, mean, population standard deviation, runtime, and hardware.
8. No hidden-test access or tuning.

Until that exists, code can be unit-tested on synthetic rankings, but model
scores cannot be claimed as official-comparable.

## 72-Hour Execution Recommendation

- Hours 0-6: lock evaluator/candidate protocol and implement BPR-FM.
- Hours 6-18: run fixed-seed BPR and NDCG-weighted variants.
- Hours 18-34: add DIN-lite and strict chronological-history tests.
- Hours 34-46: add the shared-bottom multi-task model.
- Hours 46-58: select and ablate the best model; attempt debiasing only if the
  core pipeline is stable.
- Hours 58-72: robustness, reproducibility, final retraining, documentation, and
  demo evidence.

## Primary Research Sources

- Rendle et al., [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618).
- Burges et al., [Learning to Rank with Non-Smooth Cost Functions](https://www.microsoft.com/en-us/research/publication/learning-to-rank-with-non-smooth-cost-functions/).
- Zhou et al., [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978).
- Kang and McAuley, [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781).
- Ma et al., [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/).
- Ma et al., [Entire Space Multi-Task Model](https://arxiv.org/abs/1804.07931).
- Gao et al., [KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](https://arxiv.org/abs/2208.08696).
- Zheng et al., [Counterfactual Watch Model for Audience Retention](https://arxiv.org/abs/2406.07932).
- Joachims et al., [Unbiased Learning-to-Rank with Biased Feedback](https://arxiv.org/abs/1608.04468).

## Part 2 Gate Result

The research and prioritization criteria are complete. No model improvement or
Part 3 framework work was started. The remaining prerequisite is the official
candidate-set/evaluator protocol for click NDCG@10 and Recall@50.

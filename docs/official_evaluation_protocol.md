# Reconstructed Official Evaluation Protocol

## Fixed Requirements

- Relevance is `is_click == 1`.
- Both NDCG@10 and Recall@50 are reported.
- Only the training dates, 8-21 April 2022, train the model.
- Validation dates are 22-28 April 2022.
- Public-test rows are discarded and never retained, scored, or used; hidden
  test data is not accessed.

## Candidate Reconstruction

The supplied files do not include a non-trivial per-user Recall@50 candidate
list. Version `full_catalogue_v1` therefore uses all 7,583 videos from
`video_features_basic_pure.csv` as the candidate universe for every eligible
validation user.

Videos clicked by a user during training are filtered before ranking, unless
the same video is a validation positive. This avoids recommending already
consumed positives while ensuring every validation positive remains eligible.

Users without a validation click are excluded because Recall has a zero
denominator. The remaining 17,965 users contain 54,800 unique clicked
user-video pairs, averaging 3.05 positives per eligible user.

Exact score ties are resolved by ascending video ID. Metrics are macro-averaged
over eligible users:

- NDCG@10 uses binary relevance and logarithmic discount.
- Recall@50 is clicked positives retrieved in the top 50 divided by all of that
  user's validation clicked positives.
- `validation.combined` is the arithmetic mean of the two metrics. Against a
  fixed baseline, maximizing this is equivalent to maximizing the equal-weight
  mean absolute improvement requested in the problem statement.

## Status

This is the team's documented, non-trivial reconstruction of the official
requirements. If organizers supply a candidate file or evaluator, only the
candidate provider should be replaced; click relevance and the two metric
implementations remain fixed.

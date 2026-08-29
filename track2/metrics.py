"""Click NDCG@10 and Recall@50 over a deterministic full catalogue."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .data import Track2Dataset


class CatalogueScorer(Protocol):
    def score_catalogue(
        self, user_features: np.ndarray, candidate_X: np.ndarray
    ) -> np.ndarray: ...


def metrics_from_rankings(
    rankings: np.ndarray, positives: tuple[np.ndarray, ...]
) -> dict[str, float]:
    if rankings.ndim != 2 or rankings.shape[0] != len(positives):
        raise ValueError("rankings must have one row per positive-item set")
    if rankings.shape[1] < 50:
        raise ValueError("rankings must contain at least 50 candidates")
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    ndcg_values = np.empty(len(positives), dtype=np.float64)
    recall_values = np.empty(len(positives), dtype=np.float64)
    for index, relevant in enumerate(positives):
        if len(relevant) == 0:
            raise ValueError("zero-positive users must be excluded before evaluation")
        top_10_relevance = np.isin(rankings[index, :10], relevant, assume_unique=True)
        dcg = float(np.dot(top_10_relevance, discounts))
        ideal = float(discounts[: min(len(relevant), 10)].sum())
        ndcg_values[index] = dcg / ideal
        recall_values[index] = np.isin(
            rankings[index, :50], relevant, assume_unique=True
        ).sum() / len(relevant)
    ndcg = float(ndcg_values.mean())
    recall = float(recall_values.mean())
    return {
        "validation.ndcg_at_10": ndcg,
        "validation.recall_at_50": recall,
        "validation.combined": (ndcg + recall) / 2.0,
        "validation.eligible_users": float(len(positives)),
    }


def evaluate_full_catalog(
    model: CatalogueScorer,
    dataset: Track2Dataset,
    *,
    user_batch_size: int = 128,
) -> dict[str, float]:
    """Rank all catalogue videos, filtering past clicks except current positives."""
    if user_batch_size < 1:
        raise ValueError("user_batch_size must be positive")
    all_rankings = np.empty(
        (len(dataset.validation_user_features), 50), dtype=np.int32
    )
    for start in range(0, len(all_rankings), user_batch_size):
        stop = min(start + user_batch_size, len(all_rankings))
        scores = model.score_catalogue(
            dataset.validation_user_features[start:stop], dataset.candidate_X
        )
        if scores.shape != (stop - start, len(dataset.candidate_X)):
            raise ValueError("model returned an invalid catalogue-score shape")
        if not np.isfinite(scores).all():
            raise ValueError("catalogue scores must be finite before filtering")
        for local_index, global_index in enumerate(range(start, stop)):
            seen = dataset.training_clicked_items[global_index]
            relevant = dataset.validation_positive_items[global_index]
            filtered = np.setdiff1d(seen, relevant, assume_unique=True)
            scores[local_index, filtered] = -np.inf
        # Catalogue rows are sorted by video ID; stable sorting therefore breaks
        # exact score ties by ascending video ID.
        all_rankings[start:stop] = np.argsort(
            -scores, axis=1, kind="stable"
        )[:, :50]
    return metrics_from_rankings(all_rankings, dataset.validation_positive_items)

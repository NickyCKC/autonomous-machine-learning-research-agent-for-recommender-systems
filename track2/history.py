"""A lightweight DIN-inspired scorer using strictly prior click history."""

from __future__ import annotations

import numpy as np

from .data import Track2Dataset
from .models import FMRanker


class DINLiteRanker:
    """Rerank FM scores with candidate-conditioned recency-weighted interests.

    The history representation combines cosine similarity to a recency-weighted
    interest vector with sparse author and duration matches. It is intentionally
    small enough for the 72-hour setting and contains no validation events.
    """

    def __init__(
        self,
        base_model: FMRanker,
        dataset: Track2Dataset,
        *,
        alpha: float,
        max_history: int = 50,
        recency_decay: float = 0.05,
        author_weight: float = 0.5,
        duration_weight: float = 0.1,
    ) -> None:
        if alpha < 0 or max_history < 1 or recency_decay < 0:
            raise ValueError("alpha/recency must be non-negative and history positive")
        self.base_model = base_model
        self.alpha = float(alpha)
        self.candidate_X = dataset.candidate_X
        item_embeddings = base_model.V[dataset.candidate_X].sum(axis=1)
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        self.normalized_items = item_embeddings / np.maximum(norms, 1e-8)
        self.candidate_authors = dataset.candidate_X[:, 1]
        self.candidate_durations = dataset.candidate_X[:, 2]
        self.profiles: dict[int, tuple[np.ndarray, dict[int, float], dict[int, float]]] = {}

        for user_feature, full_history in zip(
            dataset.validation_user_features, dataset.training_click_history
        ):
            history = full_history[-max_history:]
            if len(history) == 0:
                continue
            ages = np.arange(len(history) - 1, -1, -1, dtype=np.float32)
            weights = np.exp(-recency_decay * ages)
            weights /= weights.sum()
            interest = weights @ self.normalized_items[history]
            interest /= max(float(np.linalg.norm(interest)), 1e-8)
            author_profile: dict[int, float] = {}
            duration_profile: dict[int, float] = {}
            for item_index, weight in zip(history, weights):
                author = int(self.candidate_authors[item_index])
                duration = int(self.candidate_durations[item_index])
                author_profile[author] = author_profile.get(author, 0.0) + float(weight)
                duration_profile[duration] = duration_profile.get(duration, 0.0) + float(weight)
            self.profiles[int(user_feature)] = (
                interest.astype(np.float32),
                {key: author_weight * value for key, value in author_profile.items()},
                {key: duration_weight * value for key, value in duration_profile.items()},
            )

    def score_catalogue(
        self, user_features: np.ndarray, candidate_X: np.ndarray
    ) -> np.ndarray:
        if not np.array_equal(candidate_X, self.candidate_X):
            raise ValueError("DIN-lite candidate catalogue differs from its dataset")
        scores = self.base_model.score_catalogue(user_features, candidate_X)
        if self.alpha == 0:
            return scores
        for row, user_feature in enumerate(user_features):
            profile = self.profiles.get(int(user_feature))
            if profile is None:
                continue
            interest, author_profile, duration_profile = profile
            history_score = self.normalized_items @ interest
            for author, weight in author_profile.items():
                history_score[self.candidate_authors == author] += weight
            for duration, weight in duration_profile.items():
                history_score[self.candidate_durations == duration] += weight
            scores[row] += self.alpha * history_score
        return scores

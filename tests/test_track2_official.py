from __future__ import annotations

import numpy as np
import unittest

from track2.data import Track2Dataset
from track2.metrics import evaluate_full_catalog, metrics_from_rankings
from track2.models import FMRanker


class OfficialMetricTests(unittest.TestCase):
    def test_ndcg_and_recall_use_click_positions(self) -> None:
        ranking = np.arange(60, dtype=np.int32)[None, :]
        metrics = metrics_from_rankings(
            ranking, (np.asarray([0, 1, 55], dtype=np.int32),)
        )
        discounts = 1.0 / np.log2(np.arange(2, 5, dtype=np.float64))
        expected_ndcg = discounts[:2].sum() / discounts.sum()
        self.assertAlmostEqual(metrics["validation.ndcg_at_10"], expected_ndcg)
        self.assertAlmostEqual(metrics["validation.recall_at_50"], 2 / 3)
        self.assertAlmostEqual(
            metrics["validation.combined"], (expected_ndcg + 2 / 3) / 2
        )

    def test_zero_positive_users_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics_from_rankings(
                np.arange(50, dtype=np.int32)[None, :],
                (np.asarray([], dtype=np.int32),),
            )

    def test_training_click_is_filtered_without_removing_validation_positive(self) -> None:
        class FixedScorer:
            def score_catalogue(self, user_features, candidate_X):
                return np.tile(np.arange(60, 0, -1, dtype=np.float32), (len(user_features), 1))

        candidate_X = np.column_stack(
            [np.arange(60), np.arange(60), np.arange(60)]
        ).astype(np.int32)
        dataset = Track2Dataset(
            train_X=np.empty((0, 4), dtype=np.int32),
            train_y=np.empty(0, dtype=np.float32),
            candidate_X=candidate_X,
            candidate_video_ids=np.arange(60, dtype=np.int32),
            validation_user_features=np.asarray([0], dtype=np.int32),
            validation_positive_items=(np.asarray([1], dtype=np.int32),),
            training_clicked_items=(np.asarray([0, 1], dtype=np.int32),),
            unknown_user_feature=1,
            dimension=180,
            field_names=("user_id", "video_id", "author_id", "duration_bucket"),
        )
        metrics = evaluate_full_catalog(FixedScorer(), dataset)
        self.assertEqual(metrics["validation.ndcg_at_10"], 1.0)
        self.assertEqual(metrics["validation.recall_at_50"], 1.0)


class BprTests(unittest.TestCase):
    def test_bpr_step_increases_positive_negative_difference(self) -> None:
        model = FMRanker(8, factors=4, learning_rate=0.02, seed=0)
        positive = np.asarray([[0, 2, 4, 6]], dtype=np.int32)
        negative = np.asarray([[0, 3, 5, 7]], dtype=np.int32)
        before = float(model.logits(positive)[0][0] - model.logits(negative)[0][0])
        for _ in range(20):
            model.bpr_step(positive, negative)
        after = float(model.logits(positive)[0][0] - model.logits(negative)[0][0])
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()

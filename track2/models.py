"""Matching pointwise and pairwise Factorization Machine training."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


class FMRanker:
    def __init__(
        self,
        dimension: int,
        *,
        factors: int = 16,
        learning_rate: float = 0.002,
        l2: float = 1e-6,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dimension, factors)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.step_number = 0

    def logits(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interaction = 0.5 * (
            (summed * summed).sum(axis=1)
            - (embeddings * embeddings).sum(axis=(1, 2))
        )
        return self.bias + self.W[X].sum(axis=1) + interaction, embeddings, summed

    def _gradients(
        self,
        X: np.ndarray,
        embeddings: np.ndarray,
        summed: np.ndarray,
        coefficient: np.ndarray,
        grad_V: np.ndarray,
        grad_W: np.ndarray,
    ) -> None:
        np.add.at(grad_W, X, coefficient[:, None])
        np.add.at(
            grad_V,
            X,
            coefficient[:, None, None] * (summed[:, None, :] - embeddings),
        )

    def _adam(self, grad_V: np.ndarray, grad_W: np.ndarray) -> None:
        grad_V += self.l2 * self.V
        grad_W += self.l2 * self.W
        self.step_number += 1
        beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, moment, variance in (
            (self.V, grad_V, self.mV, self.vV),
            (self.W, grad_W, self.mW, self.vW),
        ):
            moment *= beta_1
            moment += (1 - beta_1) * gradient
            variance *= beta_2
            variance += (1 - beta_2) * gradient * gradient
            corrected_moment = moment / (1 - beta_1**self.step_number)
            corrected_variance = variance / (1 - beta_2**self.step_number)
            parameter -= self.learning_rate * corrected_moment / (
                np.sqrt(corrected_variance) + epsilon
            )

    def pointwise_step(self, X: np.ndarray, y: np.ndarray) -> float:
        logits, embeddings, summed = self.logits(X)
        probabilities = sigmoid(logits)
        coefficient = ((probabilities - y) / len(y)).astype(np.float32)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        self._gradients(X, embeddings, summed, coefficient, grad_V, grad_W)
        self._adam(grad_V, grad_W)
        self.bias -= self.learning_rate * coefficient.sum()
        return float(
            -np.mean(
                y * np.log(probabilities + 1e-9)
                + (1 - y) * np.log(1 - probabilities + 1e-9)
            )
        )

    def bpr_step(self, positive_X: np.ndarray, negative_X: np.ndarray) -> float:
        positive_logits, positive_embeddings, positive_summed = self.logits(positive_X)
        negative_logits, negative_embeddings, negative_summed = self.logits(negative_X)
        difference = positive_logits - negative_logits
        coefficient = ((sigmoid(difference) - 1.0) / len(difference)).astype(np.float32)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        self._gradients(
            positive_X,
            positive_embeddings,
            positive_summed,
            coefficient,
            grad_V,
            grad_W,
        )
        self._gradients(
            negative_X,
            negative_embeddings,
            negative_summed,
            -coefficient,
            grad_V,
            grad_W,
        )
        self._adam(grad_V, grad_W)
        return float(-np.mean(np.log(sigmoid(difference) + 1e-9)))

    def score_catalogue(
        self, user_features: np.ndarray, candidate_X: np.ndarray
    ) -> np.ndarray:
        item_embeddings = self.V[candidate_X]
        item_sum = item_embeddings.sum(axis=1)
        item_base = self.W[candidate_X].sum(axis=1) + 0.5 * (
            (item_sum * item_sum).sum(axis=1)
            - (item_embeddings * item_embeddings).sum(axis=(1, 2))
        )
        return self.V[user_features] @ item_sum.T + item_base[None, :]

    def state(self) -> dict[str, np.ndarray]:
        return {"V": self.V, "W": self.W, "bias": np.asarray(self.bias)}

    @classmethod
    def from_npz(cls, path: str, *, learning_rate: float = 0.002) -> "FMRanker":
        with np.load(path) as state:
            factors = int(state["V"].shape[1])
            model = cls(
                int(state["V"].shape[0]),
                factors=factors,
                learning_rate=learning_rate,
                seed=0,
            )
            model.V[:] = state["V"]
            model.W[:] = state["W"]
            model.bias = np.float32(state["bias"])
        return model


def train_pointwise(
    model: FMRanker,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        order = rng.permutation(len(y))
        losses = [
            model.pointwise_step(X[order[start : start + batch_size]], y[order[start : start + batch_size]])
            for start in range(0, len(order), batch_size)
        ]
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "runtime_seconds": time.monotonic() - started,
            }
        )
    return history


def _catalogue_pairs(
    X: np.ndarray,
    y: np.ndarray,
    candidate_X: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair every click with a catalogue item not clicked by that user."""
    positive_rows = np.flatnonzero(y > 0.5)
    if len(positive_rows) == 0:
        raise ValueError("BPR training requires positive rows")
    users = X[positive_rows, 0].astype(np.int64)
    positive_item_features = X[positive_rows, 1].astype(np.int64)
    key_base = int(candidate_X[:, 0].max()) + 1
    positive_keys = np.unique(users * key_base + positive_item_features)
    sampled_indices = rng.integers(0, len(candidate_X), size=len(positive_rows))
    while True:
        sampled_features = candidate_X[sampled_indices, 0].astype(np.int64)
        sampled_keys = users * key_base + sampled_features
        conflicts = np.isin(sampled_keys, positive_keys, assume_unique=False)
        if not conflicts.any():
            break
        sampled_indices[conflicts] = rng.integers(
            0, len(candidate_X), size=int(conflicts.sum())
        )
    negative_X = np.empty_like(X[positive_rows])
    negative_X[:, 0] = X[positive_rows, 0]
    negative_X[:, 1:] = candidate_X[sampled_indices]
    return positive_rows, negative_X


def train_bpr(
    model: FMRanker,
    X: np.ndarray,
    y: np.ndarray,
    candidate_X: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        positives, negative_X = _catalogue_pairs(X, y, candidate_X, rng)
        order = rng.permutation(len(positives))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            losses.append(model.bpr_step(X[positives[batch]], negative_X[batch]))
        history.append(
            {
                "epoch": epoch,
                "pairs": len(positives),
                "loss": float(np.mean(losses)),
                "runtime_seconds": time.monotonic() - started,
            }
        )
    return history


class AuxiliaryHeads:
    """Task-specific linear heads over shared FM interaction embeddings."""

    def __init__(self, dimension: int, targets: dict[str, np.ndarray]) -> None:
        self.weights = {
            name: np.zeros(dimension, dtype=np.float32) for name in targets
        }
        self.biases = {}
        for name, values in targets.items():
            mean = float(np.clip(values.mean(), 1e-4, 1 - 1e-4))
            self.biases[name] = np.float32(np.log(mean / (1 - mean)))

    def state(self) -> dict[str, np.ndarray]:
        state = {}
        for name in sorted(self.weights):
            state[f"aux_{name}_W"] = self.weights[name]
            state[f"aux_{name}_bias"] = np.asarray(self.biases[name])
        return state


def auxiliary_shared_step(
    model: FMRanker,
    heads: AuxiliaryHeads,
    X: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    learning_rate: float,
    strength: float,
    task_weights: dict[str, float],
) -> dict[str, float]:
    embeddings = model.V[X]
    summed = embeddings.sum(axis=1)
    interaction = 0.5 * (
        (summed * summed).sum(axis=1)
        - (embeddings * embeddings).sum(axis=(1, 2))
    )
    shared_gradient = np.zeros_like(model.V)
    losses = {}
    for name, values in targets.items():
        weight = float(task_weights.get(name, 1.0))
        logits = heads.biases[name] + heads.weights[name][X].sum(axis=1) + interaction
        probabilities = sigmoid(logits)
        coefficient = (
            strength * weight * (probabilities - values) / len(values)
        ).astype(np.float32)
        task_gradient = np.zeros_like(heads.weights[name])
        np.add.at(task_gradient, X, coefficient[:, None])
        np.add.at(
            shared_gradient,
            X,
            coefficient[:, None, None] * (summed[:, None, :] - embeddings),
        )
        heads.weights[name] -= learning_rate * task_gradient
        heads.biases[name] -= learning_rate * coefficient.sum()
        losses[name] = float(
            -np.mean(
                values * np.log(probabilities + 1e-9)
                + (1 - values) * np.log(1 - probabilities + 1e-9)
            )
        )
    model.V -= learning_rate * shared_gradient
    return losses


def continue_bpr_with_auxiliary(
    model: FMRanker,
    X: np.ndarray,
    click_y: np.ndarray,
    candidate_X: np.ndarray,
    auxiliary_targets: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    auxiliary_strength: float,
    auxiliary_fraction: float = 0.25,
    auxiliary_learning_rate: float = 0.001,
    task_weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], AuxiliaryHeads]:
    """Continue primary BPR, optionally alternating shared auxiliary updates."""
    if not 0 <= auxiliary_fraction <= 1 or auxiliary_strength < 0:
        raise ValueError("invalid auxiliary fraction or strength")
    task_weights = task_weights or {}
    heads = AuxiliaryHeads(model.V.shape[0], auxiliary_targets)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        positive_rows, negative_X = _catalogue_pairs(X, click_y, candidate_X, rng)
        pair_order = rng.permutation(len(positive_rows))
        bpr_losses = []
        for start in range(0, len(pair_order), batch_size):
            batch = pair_order[start : start + batch_size]
            bpr_losses.append(
                model.bpr_step(X[positive_rows[batch]], negative_X[batch])
            )

        auxiliary_losses: dict[str, list[float]] = {
            name: [] for name in auxiliary_targets
        }
        if auxiliary_strength > 0 and auxiliary_fraction > 0:
            sample_count = int(len(X) * auxiliary_fraction)
            auxiliary_order = rng.permutation(len(X))[:sample_count]
            for start in range(0, len(auxiliary_order), batch_size):
                batch = auxiliary_order[start : start + batch_size]
                losses = auxiliary_shared_step(
                    model,
                    heads,
                    X[batch],
                    {name: values[batch] for name, values in auxiliary_targets.items()},
                    learning_rate=auxiliary_learning_rate,
                    strength=auxiliary_strength,
                    task_weights=task_weights,
                )
                for name, loss in losses.items():
                    auxiliary_losses[name].append(loss)
        history.append(
            {
                "epoch": epoch,
                "bpr_loss": float(np.mean(bpr_losses)),
                "auxiliary_losses": {
                    name: float(np.mean(values)) if values else None
                    for name, values in auxiliary_losses.items()
                },
                "runtime_seconds": time.monotonic() - started,
            }
        )
    return history, heads

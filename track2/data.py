"""Leakage-controlled click data for full-catalogue Track 2 evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRAIN_DATES = (20220408, 20220421)
VALID_DATES = (20220422, 20220428)


@dataclass(frozen=True)
class Track2Dataset:
    train_X: np.ndarray
    train_y: np.ndarray
    candidate_X: np.ndarray
    candidate_video_ids: np.ndarray
    validation_user_features: np.ndarray
    validation_positive_items: tuple[np.ndarray, ...]
    training_clicked_items: tuple[np.ndarray, ...]
    training_click_history: tuple[np.ndarray, ...]
    unknown_user_feature: int
    dimension: int
    field_names: tuple[str, ...]


def _read_video_catalog(data_dir: Path) -> tuple[list[int], dict[int, tuple[int, float]]]:
    metadata: dict[int, tuple[int, float]] = {}
    with (data_dir / "video_features_basic_pure.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            video_id = int(row["video_id"])
            author_id = int(row["author_id"])
            duration = float(row["video_duration"] or 0.0)
            metadata[video_id] = (author_id, duration)
    video_ids = sorted(metadata)
    if not video_ids:
        raise ValueError("video catalogue is empty")
    return video_ids, metadata


def load_track2_dataset(data_dir: str | Path) -> Track2Dataset:
    """Load train/validation clicks while discarding all public-test dates."""
    data_dir = Path(data_dir)
    video_ids, metadata = _read_video_catalog(data_dir)
    item_to_index = {video_id: index for index, video_id in enumerate(video_ids)}
    authors = sorted({metadata[video_id][0] for video_id in video_ids})
    author_to_index = {author: index for index, author in enumerate(authors)}
    durations = np.asarray([metadata[video_id][1] for video_id in video_ids])
    duration_edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])

    train_rows: list[tuple[int, int, int, int]] = []
    train_users: dict[int, int] = {}
    train_path = data_dir / "log_standard_4_08_to_4_21_pure.csv"
    with train_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            date = int(row["date"])
            if not (TRAIN_DATES[0] <= date <= TRAIN_DATES[1]):
                continue
            user_id = int(row["user_id"])
            video_id = int(row["video_id"])
            if video_id not in item_to_index:
                continue
            if user_id not in train_users:
                train_users[user_id] = len(train_users)
            train_rows.append(
                (
                    user_id,
                    video_id,
                    int(row["is_click"] != "0"),
                    int(row["time_ms"]),
                )
            )

    user_count = len(train_users)
    unknown_user_feature = user_count
    user_dim = user_count + 1
    item_offset = user_dim
    author_offset = item_offset + len(video_ids)
    duration_offset = author_offset + len(authors)
    dimension = duration_offset + 10

    candidate_X = np.empty((len(video_ids), 3), dtype=np.int32)
    for item_index, video_id in enumerate(video_ids):
        author_id, duration = metadata[video_id]
        duration_bucket = int(np.searchsorted(duration_edges, duration))
        candidate_X[item_index] = (
            item_offset + item_index,
            author_offset + author_to_index[author_id],
            duration_offset + duration_bucket,
        )

    train_X = np.empty((len(train_rows), 4), dtype=np.int32)
    train_y = np.empty(len(train_rows), dtype=np.float32)
    clicked_by_user: dict[int, set[int]] = {}
    click_events_by_user: dict[int, list[tuple[int, int]]] = {}
    for index, (user_id, video_id, click, time_ms) in enumerate(train_rows):
        item_index = item_to_index[video_id]
        train_X[index, 0] = train_users[user_id]
        train_X[index, 1:] = candidate_X[item_index]
        train_y[index] = click
        if click:
            clicked_by_user.setdefault(user_id, set()).add(item_index)
            click_events_by_user.setdefault(user_id, []).append((time_ms, item_index))

    positives_by_user: dict[int, set[int]] = {}
    valid_path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    with valid_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            date = int(row["date"])
            if not (VALID_DATES[0] <= date <= VALID_DATES[1]):
                continue
            if row["is_click"] == "0":
                continue
            video_id = int(row["video_id"])
            if video_id in item_to_index:
                positives_by_user.setdefault(int(row["user_id"]), set()).add(
                    item_to_index[video_id]
                )

    eligible_users = sorted(positives_by_user)
    validation_user_features = np.asarray(
        [train_users.get(user_id, unknown_user_feature) for user_id in eligible_users],
        dtype=np.int32,
    )
    validation_positive_items = tuple(
        np.asarray(sorted(positives_by_user[user_id]), dtype=np.int32)
        for user_id in eligible_users
    )
    training_clicked_items = tuple(
        np.asarray(sorted(clicked_by_user.get(user_id, set())), dtype=np.int32)
        for user_id in eligible_users
    )
    training_click_history_list = []
    for user_id in eligible_users:
        events = sorted(click_events_by_user.get(user_id, []))
        # Preserve chronological order while removing repeated video clicks.
        seen: set[int] = set()
        ordered_unique = []
        for _, item_index in reversed(events):
            if item_index not in seen:
                seen.add(item_index)
                ordered_unique.append(item_index)
        training_click_history_list.append(
            np.asarray(list(reversed(ordered_unique)), dtype=np.int32)
        )
    training_click_history = tuple(training_click_history_list)

    return Track2Dataset(
        train_X=train_X,
        train_y=train_y,
        candidate_X=candidate_X,
        candidate_video_ids=np.asarray(video_ids, dtype=np.int32),
        validation_user_features=validation_user_features,
        validation_positive_items=validation_positive_items,
        training_clicked_items=training_clicked_items,
        training_click_history=training_click_history,
        unknown_user_feature=unknown_user_feature,
        dimension=dimension,
        field_names=("user_id", "video_id", "author_id", "duration_bucket"),
    )

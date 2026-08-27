"""Purged walk-forward evaluation utilities.

Designed for trading research: training data always precedes validation data,
and an embargo prevents overlapping label leakage around fold boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    embargo_days: int


@dataclass(frozen=True)
class WalkForwardResult:
    fold: int
    metrics: dict[str, float]


def make_folds(
    timestamps: Iterable[pd.Timestamp],
    min_train_days: int = 252,
    validation_days: int = 63,
    step_days: int = 63,
    embargo_days: int = 5,
) -> list[WalkForwardFold]:
    ts = pd.DatetimeIndex(sorted(pd.to_datetime(list(timestamps)).unique()))
    if len(ts) == 0:
        return []
    folds: list[WalkForwardFold] = []
    cursor = min_train_days
    fold = 1
    while cursor + validation_days <= len(ts):
        train_start = ts[0]
        train_end = ts[cursor - 1]
        validation_start = ts[cursor]
        validation_end = ts[cursor + validation_days - 1]
        folds.append(WalkForwardFold(fold, train_start, train_end, validation_start, validation_end, embargo_days))
        cursor += step_days
        fold += 1
    return folds


def evaluate_walk_forward(
    frame: pd.DataFrame,
    prediction_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    outcome_col: str = "outcome",
    min_train_days: int = 252,
    validation_days: int = 63,
    step_days: int = 63,
    embargo_days: int = 5,
) -> list[WalkForwardResult]:
    """Run a strictly chronological evaluation and return fold-level metrics."""
    if "timestamp" not in frame.columns or outcome_col not in frame.columns:
        raise ValueError("frame requires timestamp and outcome columns")
    df = frame.copy().sort_values("timestamp")
    folds = make_folds(df["timestamp"], min_train_days, validation_days, step_days, embargo_days)
    results: list[WalkForwardResult] = []
    for f in folds:
        train = df[(df.timestamp >= f.train_start) & (df.timestamp <= f.train_end)].copy()
        validation = df[(df.timestamp >= f.validation_start + timedelta(days=f.embargo_days)) & (df.timestamp <= f.validation_end)].copy()
        if train.empty or validation.empty:
            continue
        predictions = prediction_fn(train, validation)
        if len(predictions) != len(validation):
            raise ValueError("prediction_fn returned the wrong number of observations")
        actual = pd.to_numeric(validation[outcome_col], errors="coerce")
        pred = pd.to_numeric(predictions, errors="coerce")
        mask = actual.notna() & pred.notna()
        actual, pred = actual[mask], pred[mask]
        if len(actual) == 0:
            continue
        hit_rate = float((np_sign(actual) == np_sign(pred)).mean())
        mean_outcome = float(actual.mean())
        results.append(WalkForwardResult(f.fold, {"hit_rate": hit_rate, "mean_outcome": mean_outcome, "n": float(len(actual))}))
    return results


def np_sign(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

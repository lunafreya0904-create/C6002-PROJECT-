from __future__ import annotations

import numpy as np

from hdb_price.evaluation import (
    conformal_adjustment,
    ensemble_crps,
    interval_coverage,
    point_metrics,
)


def test_point_metrics_are_zero_for_perfect_prediction() -> None:
    values = np.array([1.0, 2.0, 3.0])
    metrics = point_metrics(values, values)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_probabilistic_metrics_are_finite() -> None:
    actual = np.array([10.0, 20.0])
    samples = np.array([[8.0, 10.0, 12.0], [18.0, 20.0, 22.0]])
    assert ensemble_crps(actual, samples) >= 0
    coverage = interval_coverage(actual, samples[:, 0], samples[:, -1])
    assert coverage == 1.0
    assert conformal_adjustment(actual, samples[:, 0], samples[:, -1], alpha=0.2) == 0.0


def test_interval_coverage_detects_misses() -> None:
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([9.0, 21.0, 29.0, 41.0])
    upper = np.array([11.0, 22.0, 31.0, 42.0])
    assert interval_coverage(actual, lower, upper) == 0.5

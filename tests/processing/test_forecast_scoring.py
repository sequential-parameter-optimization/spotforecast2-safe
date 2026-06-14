# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for processing.forecast_scoring.score_forecasts (pure comparison)."""

import math

import pandas as pd
import pytest

from spotforecast2_safe.processing.forecast_scoring import (
    SUPPORTED_METRICS,
    score_forecasts,
)

IDX = pd.date_range("2026-06-13 00:00", periods=24, freq="h", tz="UTC")
ACTUAL = pd.Series([43_858.0] * 24, index=IDX)


class TestScoreForecasts:
    def test_ranks_lower_mae_first(self):
        forecasts = {
            "four_zone_sum": ACTUAL + 1_780.0,  # flat over-prediction
            "combined": ACTUAL + 300.0,
        }
        table = score_forecasts(forecasts, ACTUAL)
        assert list(table.index) == ["combined", "four_zone_sum"]

    def test_metric_values_correct(self):
        forecasts = {"high": ACTUAL + 1_000.0}
        table = score_forecasts(forecasts, ACTUAL)
        row = table.loc["high"]
        assert row["mae"] == pytest.approx(1_000.0)
        assert row["rmse"] == pytest.approx(1_000.0)
        assert row["bias"] == pytest.approx(1_000.0)  # all-positive error
        assert row["mape"] == pytest.approx(1_000.0 / 43_858.0 * 100.0)
        assert row["n"] == 24

    def test_metric_subset_and_order(self):
        table = score_forecasts({"a": ACTUAL + 1.0}, ACTUAL, metrics=("bias", "mae"))
        assert list(table.columns) == ["bias", "mae", "n"]

    def test_n_reflects_overlap(self):
        partial = (ACTUAL + 5.0).iloc[:10]
        table = score_forecasts({"p": partial}, ACTUAL)
        assert table.loc["p", "n"] == 10

    def test_supported_metrics_constant(self):
        assert set(SUPPORTED_METRICS) == {"mae", "rmse", "bias", "mape"}

    def test_no_overlap_yields_nan_metrics(self):
        other = pd.Series(
            [1.0, 2.0],
            index=pd.date_range("2027-01-01", periods=2, freq="h", tz="UTC"),
        )
        table = score_forecasts({"x": other}, ACTUAL)
        assert table.loc["x", "n"] == 0
        assert math.isnan(table.loc["x", "mae"])

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="unsupported metric"):
            score_forecasts({"a": ACTUAL}, ACTUAL, metrics=("mae", "smape"))

    def test_empty_metrics_raises(self):
        with pytest.raises(ValueError, match="at least one metric"):
            score_forecasts({"a": ACTUAL}, ACTUAL, metrics=())

    def test_empty_forecasts_raises(self):
        with pytest.raises(ValueError, match="nothing to score"):
            score_forecasts({}, ACTUAL)

    def test_empty_actual_raises(self):
        with pytest.raises(ValueError, match="empty"):
            score_forecasts({"a": ACTUAL}, pd.Series(dtype=float))

    def test_non_series_actual_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            score_forecasts({"a": ACTUAL}, [1, 2, 3])  # type: ignore[arg-type]

    def test_non_series_forecast_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            score_forecasts({"a": [1, 2, 3]}, ACTUAL)  # type: ignore[arg-type]

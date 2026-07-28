# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for per-period forecast scoring and leaderboard aggregation."""

import math

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.processing.forecast_scoring import (
    aggregate_period_scores,
    mase_scaling_factors,
    score_forecasts_by_period,
)

IDX = pd.date_range("2026-06-10 00:00", periods=72, freq="h", tz="UTC")
ACTUAL = pd.Series([40_000.0] * 72, index=IDX)


def _offset_forecast(offsets_per_day):
    """Forecast with a constant offset per calendar day."""
    values = np.repeat(np.asarray(offsets_per_day, dtype=float), 24)
    return ACTUAL + pd.Series(values, index=IDX)


class TestScoreForecastsByPeriod:
    def test_exact_daily_metrics_from_constant_offsets(self):
        daily = score_forecasts_by_period(
            {"m": _offset_forecast([100.0, -200.0, 50.0])},
            ACTUAL,
            metrics=("mae", "bias", "upr"),
        )
        assert list(daily["mae"]) == [100.0, 200.0, 50.0]
        assert list(daily["bias"]) == [100.0, -200.0, 50.0]
        assert list(daily["upr"]) == [0.0, 100.0, 0.0]
        assert list(daily["n"]) == [24, 24, 24]

    def test_period_column_holds_day_starts(self):
        daily = score_forecasts_by_period({"m": ACTUAL + 1.0}, ACTUAL)
        expected = pd.date_range("2026-06-10", periods=3, freq="D", tz="UTC")
        assert list(daily["period"]) == list(expected)

    def test_min_obs_drops_incomplete_period(self):
        partial = (ACTUAL + 5.0).iloc[:52]  # third day has only 4 hours
        daily = score_forecasts_by_period({"m": partial}, ACTUAL, min_obs=24)
        assert len(daily) == 2

    def test_without_min_obs_incomplete_period_kept(self):
        partial = (ACTUAL + 5.0).iloc[:52]
        daily = score_forecasts_by_period({"m": partial}, ACTUAL)
        assert list(daily["n"]) == [24, 24, 4]

    def test_gap_days_produce_no_rows(self):
        with_gap = pd.concat([(ACTUAL + 5.0).iloc[:24], (ACTUAL + 5.0).iloc[48:]])
        daily = score_forecasts_by_period({"m": with_gap}, ACTUAL)
        assert len(daily) == 2  # the empty middle bin is skipped entirely

    def test_sorted_by_entry_then_period(self):
        daily = score_forecasts_by_period(
            {"z_entry": ACTUAL + 1.0, "a_entry": ACTUAL + 2.0}, ACTUAL
        )
        assert list(daily["entry"].unique()) == ["a_entry", "z_entry"]
        per_entry = daily.groupby("entry")["period"].apply(list)
        for periods in per_entry:
            assert periods == sorted(periods)

    def test_matches_manuscript_day_metrics(self):
        """Pin equality with the inline day_metrics of the bart26o manuscript."""
        rng = np.random.default_rng(42)
        forecast = ACTUAL + pd.Series(rng.normal(0, 300, 72), index=IDX)
        daily = score_forecasts_by_period(
            {"m": forecast}, ACTUAL, metrics=("mae", "rmse", "mape", "bias", "upr")
        )
        for day, row in zip(
            pd.date_range("2026-06-10", periods=3, freq="D", tz="UTC"),
            daily.itertuples(),
        ):
            mask = (IDX >= day) & (IDX < day + pd.Timedelta(days=1))
            err = (forecast - ACTUAL)[mask].to_numpy(float)
            act = ACTUAL[mask].to_numpy(float)
            nz = act != 0
            assert row.mae == float(np.mean(np.abs(err)))
            assert row.rmse == float(np.sqrt(np.mean(err**2)))
            assert row.mape == float(np.mean(np.abs(err[nz] / act[nz])) * 100)
            assert row.bias == float(np.mean(err))
            assert row.upr == float(np.mean(err < 0) * 100)

    def test_custom_column_names(self):
        daily = score_forecasts_by_period(
            {"m": ACTUAL + 1.0},
            ACTUAL,
            metrics=("mae",),
            entry_col="team_id",
            period_col="target_date",
        )
        assert list(daily.columns) == ["team_id", "target_date", "mae", "n"]

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="unsupported metric"):
            score_forecasts_by_period({"m": ACTUAL}, ACTUAL, metrics=("smape",))

    def test_empty_forecasts_raises(self):
        with pytest.raises(ValueError, match="nothing to score"):
            score_forecasts_by_period({}, ACTUAL)

    def test_non_datetime_index_raises(self):
        plain = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            score_forecasts_by_period({"m": plain}, plain)

    def test_non_series_forecast_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            score_forecasts_by_period({"m": [1.0]}, ACTUAL)  # type: ignore[dict-item]


class TestMaseScalingFactors:
    def test_ramp_gives_exactly_one(self):
        ramp = pd.Series(np.arange(72, dtype=float), index=IDX)
        factors = mase_scaling_factors(ramp, ["2026-06-11", "2026-06-12"])
        assert list(factors) == [1.0, 1.0]

    def test_alternator_gives_exactly_one(self):
        alternator = pd.Series([0.0, 1.0] * 36, index=IDX)
        factors = mase_scaling_factors(alternator, ["2026-06-12"])
        assert list(factors) == [1.0]

    def test_strict_inequality_excludes_boundary_observation(self):
        ramp = pd.Series(np.arange(72, dtype=float), index=IDX)
        ramp.iloc[23] = 100.0  # spike at 23:00 of day one
        # Day-two boundary is 00:00, which must EXCLUDE the 00:00 observation
        # itself but include the 23:00 spike diff.
        boundary = mase_scaling_factors(ramp, [pd.Timestamp("2026-06-11", tz="UTC")])
        diffs = ramp.diff().abs()
        expected = float(
            diffs[diffs.index < pd.Timestamp("2026-06-11", tz="UTC")].mean()
        )
        assert float(boundary.iloc[0]) == expected

    def test_matches_manuscript_denominator(self):
        rng = np.random.default_rng(7)
        load = pd.Series(rng.normal(40_000, 2_000, 72), index=IDX)
        days = ["2026-06-11", "2026-06-12"]
        factors = mase_scaling_factors(load, days)
        diffs = load.diff().abs()
        expected = {
            d: float(diffs[diffs.index < pd.Timestamp(d, tz="UTC")].mean())
            for d in days
        }
        assert dict(zip(factors.index, factors)) == expected

    def test_index_preserves_input_labels(self):
        ramp = pd.Series(np.arange(72, dtype=float), index=IDX)
        factors = mase_scaling_factors(ramp, ["2026-06-11"])
        assert list(factors.index) == ["2026-06-11"]

    def test_no_history_gives_nan(self):
        ramp = pd.Series(np.arange(72, dtype=float), index=IDX)
        factors = mase_scaling_factors(ramp, ["2026-06-10 00:00"])
        assert math.isnan(factors.iloc[0])

    def test_seasonality_changes_step(self):
        alternator = pd.Series([0.0, 1.0] * 36, index=IDX)
        factors = mase_scaling_factors(alternator, ["2026-06-12"], seasonality=2)
        assert list(factors) == [0.0]  # two-step difference of an alternator

    def test_bad_seasonality_raises(self):
        ramp = pd.Series(np.arange(72, dtype=float), index=IDX)
        with pytest.raises(ValueError, match="positive integer"):
            mase_scaling_factors(ramp, ["2026-06-11"], seasonality=0)

    def test_non_datetime_index_raises(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            mase_scaling_factors(pd.Series([1.0, 2.0]), ["2026-06-11"])


class TestAggregatePeriodScores:
    def _daily(self):
        return score_forecasts_by_period(
            {"good": ACTUAL + 50.0, "bad": ACTUAL + 900.0},
            ACTUAL,
            metrics=("mae", "bias"),
        )

    def test_means_and_ranks(self):
        board = aggregate_period_scores(self._daily(), rank_by="mae")
        assert list(board.index) == ["good", "bad"]
        assert list(board["rank"]) == [1, 2]
        assert board.loc["good", "mae"] == 50.0
        assert board.loc["good", "n_periods"] == 3

    def test_n_periods_breaks_ties(self):
        daily = pd.DataFrame(
            {
                "entry": ["few", "few", "many", "many", "many"],
                "period": pd.date_range("2026-06-10", periods=5, freq="D"),
                "mae": [100.0] * 5,
            }
        )
        board = aggregate_period_scores(daily, rank_by="mae")
        assert list(board.index) == ["many", "few"]

    def test_default_metrics_exclude_n(self):
        board = aggregate_period_scores(self._daily())
        assert list(board.columns) == ["mae", "bias", "n_periods", "rank"]

    def test_descending_ranking(self):
        daily = self._daily().rename(columns={"mae": "score"})
        board = aggregate_period_scores(daily, rank_by="score", ascending=False)
        assert list(board.index) == ["bad", "good"]

    def test_explicit_metrics_subset(self):
        board = aggregate_period_scores(self._daily(), metrics=["mae"], rank_by="mae")
        assert list(board.columns) == ["mae", "n_periods", "rank"]

    def test_missing_metric_raises(self):
        with pytest.raises(ValueError, match="not in period_scores"):
            aggregate_period_scores(self._daily(), metrics=["mase"])

    def test_rank_by_not_in_metrics_raises(self):
        with pytest.raises(ValueError, match="not among the metrics"):
            aggregate_period_scores(self._daily(), metrics=["bias"], rank_by="mae")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            aggregate_period_scores(pd.DataFrame())

    def test_matches_manuscript_aggregation(self):
        """Pin equality with the inline groupby-mean + sort of the manuscript."""
        rng = np.random.default_rng(3)
        daily = pd.DataFrame(
            {
                "entry": np.repeat(["a", "b", "c"], 4),
                "period": list(pd.date_range("2026-06-10", periods=4, freq="D")) * 3,
                "mae": rng.uniform(400, 900, 12),
            }
        )
        board = aggregate_period_scores(daily, rank_by="mae")
        g = daily.groupby("entry")
        manual = pd.DataFrame({"mean_mae": g["mae"].mean(), "n_days": g.size()})
        manual = manual.sort_values(["mean_mae", "n_days"], ascending=[True, False])
        assert list(board.index) == list(manual.index)
        assert list(board["mae"]) == list(manual["mean_mae"])

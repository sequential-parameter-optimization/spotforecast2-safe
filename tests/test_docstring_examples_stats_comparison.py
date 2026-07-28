# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Smoke tests mirroring the live ``{python}`` docstring examples of
``stats.comparison``, ``stats.errors``, and the per-period scoring
helpers in ``processing.forecast_scoring``.

Keep these in sync with the ``Examples:`` cells they duplicate — the
cells execute at ``quarto render`` time, these at pytest time.
"""

import numpy as np
import pandas as pd


def test_score_forecasts_by_period_example():
    from spotforecast2_safe.processing.forecast_scoring import score_forecasts_by_period

    idx = pd.date_range("2026-06-10", periods=48, freq="h", tz="UTC")
    actual = pd.Series([40_000.0] * 48, index=idx)
    forecasts = {"over": actual + 100.0, "under": actual - 200.0}
    daily = score_forecasts_by_period(
        forecasts, actual, metrics=("mae", "bias", "upr"), min_obs=24
    )
    assert set(daily["entry"]) == {"over", "under"}
    assert (daily.loc[daily["entry"] == "under", "upr"] == 100.0).all()


def test_mase_scaling_factors_example():
    from spotforecast2_safe.processing.forecast_scoring import mase_scaling_factors

    idx = pd.date_range("2026-06-01", periods=96, freq="h", tz="UTC")
    ramp = pd.Series(np.arange(96, dtype=float), index=idx)
    factors = mase_scaling_factors(ramp, ["2026-06-03", "2026-06-04"])
    assert (factors == 1.0).all()


def test_aggregate_period_scores_example():
    from spotforecast2_safe.processing.forecast_scoring import (
        aggregate_period_scores,
        score_forecasts_by_period,
    )

    idx = pd.date_range("2026-06-10", periods=72, freq="h", tz="UTC")
    actual = pd.Series([40_000.0] * 72, index=idx)
    daily = score_forecasts_by_period(
        {"good": actual + 50.0, "bad": actual + 900.0},
        actual,
        metrics=("mae", "bias"),
    )
    board = aggregate_period_scores(daily, rank_by="mae")
    assert list(board.index) == ["good", "bad"]
    assert list(board["rank"]) == [1, 2]


def test_paired_comparison_example():
    from spotforecast2_safe.stats.comparison import paired_comparison

    days = pd.date_range("2026-06-10", periods=6, freq="D")
    a = pd.Series([500.0, 520.0, 480.0, 510.0, 490.0, 505.0], index=days)
    result = paired_comparison(a, a + 40.0)
    assert result.wins_a == 6


def test_holm_adjust_example():
    from spotforecast2_safe.stats.comparison import holm_adjust

    raw = pd.Series([0.01, 0.02, 0.03, 0.04], index=list("abcd"))
    assert list(holm_adjust(raw)) == [0.04, 0.06, 0.06, 0.06]


def test_pairwise_paired_t_example():
    from spotforecast2_safe.stats.comparison import pairwise_paired_t

    days = pd.date_range("2026-06-10", periods=8, freq="D")
    panel = pd.DataFrame(
        {
            "alpha": [500.0, 520, 480, 510, 490, 505, 515, 495],
            "beta": [540.0, 565, 515, 555, 530, 545, 560, 535],
            "gamma": [502.0, 519, 483, 508, 493, 504, 517, 496],
        },
        index=days,
    )
    p = pairwise_paired_t(panel)
    assert (p.values == p.values.T).all()


def test_repeated_measures_anova_example():
    from spotforecast2_safe.stats.comparison import repeated_measures_anova

    days = pd.date_range("2026-06-10", periods=5, freq="D")
    panel = pd.DataFrame(
        {
            "alpha": [500.0, 700, 400, 650, 550],
            "beta": [540.0, 745, 435, 690, 595],
            "gamma": [505.0, 702, 405, 653, 552],
        },
        index=days,
    )
    anova = repeated_measures_anova(panel)
    assert anova.n_blocks == 5 and anova.n_treatments == 3


def test_rank_table_example():
    from spotforecast2_safe.stats.comparison import rank_table

    scores = pd.DataFrame(
        {"mae": [510.0, 462.0, 891.0], "rmse": [640.0, 655.0, 1024.0]},
        index=["alpha", "beta", "gamma"],
    )
    assert list(rank_table(scores)["mae"]) == [2, 1, 3]


def test_rank_concordance_example():
    from spotforecast2_safe.stats.comparison import rank_concordance

    ranks = pd.DataFrame(
        {"mae": [1, 2, 3, 4], "rmse": [1, 2, 3, 4], "mape": [4, 3, 2, 1]},
        index=["a", "b", "c", "d"],
    )
    tau = rank_concordance(ranks, reference="mae")
    assert tau["rmse"] == 1.0 and tau["mape"] == -1.0


def test_error_summary_example():
    from spotforecast2_safe.stats.errors import error_summary

    table = error_summary({"model": pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])})
    assert table.loc["mean", "model"] == 0.0
    assert table.loc["min", "model"] == -2.0


def test_error_profile_example():
    from spotforecast2_safe.stats.errors import error_profile

    idx = pd.date_range("2026-06-10", periods=48, freq="h", tz="UTC")
    errors = {"model": pd.Series(np.tile(np.arange(24.0), 2), index=idx)}
    profile = error_profile(errors, by="hour")
    assert list(profile.index) == list(range(24))
    assert (profile["model"] == np.arange(24.0)).all()

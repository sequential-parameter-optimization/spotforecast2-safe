# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``metrics_per_fold`` and ``TimeSeriesFold.n_folds``."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from spotforecast2_safe.backtesting import backtesting_forecaster, metrics_per_fold
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.splitter import TimeSeriesFold


@pytest.fixture
def backtest_result():
    """A small deterministic backtest: returns (y, cv, metrics_df, predictions_df)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    y = pd.Series(
        50 + 10 * np.sin(np.arange(400) / 12) + rng.normal(0, 1, 400),
        index=idx,
        name="load",
    )
    forecaster = ForecasterRecursive(
        estimator=LGBMRegressor(
            n_estimators=20,
            n_jobs=1,
            verbose=-1,
            random_state=0,
            deterministic=True,
            force_col_wise=True,
        ),
        lags=24,
    )
    forecaster.fit(y=y.iloc[:200])
    cv = TimeSeriesFold(steps=24, initial_train_size=200, refit=False, verbose=False)
    metrics_df, predictions_df = backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
        verbose=False,
    )
    return y, cv, metrics_df, predictions_df


def test_row_count_equals_fold_count(backtest_result):
    y, cv, _, predictions_df = backtest_result
    per_fold = metrics_per_fold(predictions_df, y, metric="mean_absolute_error")
    assert len(per_fold) == cv.n_folds(y)
    assert len(per_fold) == predictions_df["fold"].nunique()
    assert len(per_fold) > 1  # the whole point: a real distribution, not one row
    assert per_fold.index.name == "fold"


def test_columns_named_after_metric(backtest_result):
    y, _, _, predictions_df = backtest_result
    per_fold = metrics_per_fold(
        predictions_df, y, metric=["mean_absolute_error", "mean_squared_error"]
    )
    assert per_fold.columns.tolist() == ["mean_absolute_error", "mean_squared_error"]
    assert np.isfinite(per_fold.to_numpy()).all()


def test_aggregate_lies_within_per_fold_range(backtest_result):
    """The pooled metric (a size-weighted mean of folds) lies within the spread."""
    y, _, metrics_df, predictions_df = backtest_result
    per_fold = metrics_per_fold(predictions_df, y, metric="mean_absolute_error")
    aggregated = metrics_df["mean_absolute_error"].iloc[0]
    col = per_fold["mean_absolute_error"]
    assert col.min() <= aggregated <= col.max()


def test_missing_fold_column_raises(backtest_result):
    y, _, _, predictions_df = backtest_result
    bad = predictions_df.drop(columns=["fold"])
    with pytest.raises(ValueError, match="fold"):
        metrics_per_fold(bad, y)


def test_uncovered_index_raises(backtest_result):
    y, _, _, predictions_df = backtest_result
    with pytest.raises(ValueError, match="cover"):
        metrics_per_fold(predictions_df, y.iloc[:10])


def test_not_a_dataframe_raises():
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        metrics_per_fold([1, 2, 3], pd.Series([1.0, 2.0, 3.0]))


def test_missing_pred_column_raises(backtest_result):
    y, _, _, predictions_df = backtest_result
    bad = predictions_df.drop(columns=["pred"])
    with pytest.raises(ValueError, match="pred"):
        metrics_per_fold(bad, y)


def test_empty_predictions_raises():
    empty = pd.DataFrame({"fold": [], "pred": []})
    with pytest.raises(ValueError, match="empty"):
        metrics_per_fold(empty, pd.Series([1.0], index=pd.RangeIndex(1)))

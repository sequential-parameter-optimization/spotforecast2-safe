# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the per-fold progress-bar description (``progress_desc``).

``_backtesting_forecaster`` accepts an optional ``progress_desc`` that is used
as the prefix of the per-fold ``tqdm`` bar. The SpotOptim search uses it to show
a coarse "config k/N" label alongside the fine-grained fold progress. These
tests patch ``tqdm`` with a pass-through so the real fold loop still runs while
the call is recorded.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from spotforecast2_safe.backtesting.validation import _backtesting_forecaster
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.splitter import TimeSeriesFold


def _small_setup():
    """A small, deterministic multi-fold backtest setup."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=300, freq="h", tz="UTC")
    y = pd.Series(
        50 + 10 * np.sin(np.arange(300) / 12) + rng.normal(0, 1, 300),
        index=idx,
        name="load",
    )
    forecaster = ForecasterRecursive(
        estimator=LGBMRegressor(
            n_estimators=10,
            n_jobs=1,
            verbose=-1,
            random_state=0,
            deterministic=True,
            force_col_wise=True,
        ),
        lags=24,
    )
    cv = TimeSeriesFold(steps=24, initial_train_size=200, refit=False, verbose=False)
    return forecaster, y, cv


def test_progress_desc_forwarded_to_tqdm():
    """A non-None progress_desc reaches the fold bar as ``desc`` when shown."""
    forecaster, y, cv = _small_setup()
    with patch(
        "spotforecast2_safe.backtesting.validation.tqdm",
        side_effect=lambda iterable, **kwargs: iterable,
    ) as mock_tqdm:
        _backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            n_jobs=1,
            verbose=False,
            show_progress=True,
            progress_desc="config 42/250",
        )
    assert mock_tqdm.called
    assert mock_tqdm.call_args.kwargs.get("desc") == "config 42/250"


def test_progress_desc_defaults_to_none():
    """Default behaviour is preserved: the bar is shown but undescribed."""
    forecaster, y, cv = _small_setup()
    with patch(
        "spotforecast2_safe.backtesting.validation.tqdm",
        side_effect=lambda iterable, **kwargs: iterable,
    ) as mock_tqdm:
        _backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            n_jobs=1,
            verbose=False,
            show_progress=True,
        )
    assert mock_tqdm.called
    assert mock_tqdm.call_args.kwargs.get("desc") is None


def test_progress_desc_ignored_when_progress_disabled():
    """show_progress=False suppresses the bar entirely, even with a label."""
    forecaster, y, cv = _small_setup()
    with patch(
        "spotforecast2_safe.backtesting.validation.tqdm",
        side_effect=lambda iterable, **kwargs: iterable,
    ) as mock_tqdm:
        _backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            n_jobs=1,
            verbose=False,
            show_progress=False,
            progress_desc="config 42/250",
        )
    mock_tqdm.assert_not_called()

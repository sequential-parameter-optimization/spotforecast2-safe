# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the prepare_data window-geometry clamp.

Incident 2026-06-05/06: trailing rows that carry only exogenous columns
(future day-ahead ``Forecasted Load`` kept by ``keep_forecast_future=True``)
pushed ``data_end`` to ``index.max()``, shifting ``cov_end`` and thereby
``exo_pred`` by ~36 h. The forecaster consumes exog positionally, so the live
forecast came out phase-rolled by ~+9 h. ``prepare_data`` now clamps the
pipeline frame to the last index where at least one configured target is
observed.
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.multitask import LazyTask

N_HOURS = 24 * 28  # 4 weeks, hourly
N_FUTURE_ONLY = 36  # trailing rows where only the exog-like column is observed
PREDICT_SIZE = 6


@pytest.fixture(scope="module")
def frontier_df() -> pd.DataFrame:
    """Hourly frame mimicking the ENTSO-E live shape: the target column ends
    36 h before the exog-like column (future day-ahead rows)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=N_HOURS, freq="h", tz="UTC")
    idx.name = "DateTime"
    actual = rng.normal(100.0, 10.0, N_HOURS)
    actual[-N_FUTURE_ONLY:] = np.nan  # target unobserved over the frontier
    forecast = rng.normal(100.0, 10.0, N_HOURS)  # exog-like column, complete
    return pd.DataFrame({"Actual Load": actual, "Forecasted Load": forecast}, index=idx)


def _cfg(**kwargs) -> ConfigMulti:
    defaults = dict(
        predict_size=PREDICT_SIZE,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        verbose=False,
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


class TestPrepareDataClamp:
    """data_end/cov_end must be anchored on the last observed target."""

    def test_trailing_exog_only_rows_are_dropped(self, frontier_df, tmp_path):
        task = LazyTask(
            _cfg(cache_home=tmp_path, targets=["Actual Load"]),
            dataframe=frontier_df,
        )
        task.prepare_data()
        last_target = frontier_df["Actual Load"].dropna().index.max()
        assert task.df_pipeline.index.max() == last_target

    def test_data_end_anchored_on_last_observed_target(self, frontier_df, tmp_path):
        task = LazyTask(
            _cfg(cache_home=tmp_path, targets=["Actual Load"]),
            dataframe=frontier_df,
        )
        task.prepare_data()
        last_target = frontier_df["Actual Load"].dropna().index.max()
        assert pd.to_datetime(task.run_state.data_end, utc=True) == last_target

    def test_cov_end_is_data_end_plus_predict_size(self, frontier_df, tmp_path):
        task = LazyTask(
            _cfg(cache_home=tmp_path, targets=["Actual Load"]),
            dataframe=frontier_df,
        )
        task.prepare_data()
        data_end = pd.to_datetime(task.run_state.data_end, utc=True)
        cov_end = pd.to_datetime(task.run_state.cov_end, utc=True)
        assert cov_end == data_end + pd.Timedelta(hours=PREDICT_SIZE)

    def test_clamp_emits_warning_log(self, frontier_df, tmp_path, caplog):
        task = LazyTask(
            _cfg(cache_home=tmp_path, targets=["Actual Load"]),
            dataframe=frontier_df,
        )
        with caplog.at_level("WARNING"):
            task.prepare_data()
        assert any(
            "trailing rows without any observed target" in m for m in caplog.messages
        )

    def test_aligned_targets_are_not_clamped(self, frontier_df, tmp_path):
        """A frame whose targets reach index.max() keeps the full extent."""
        aligned = frontier_df.copy()
        aligned["Actual Load"] = aligned["Forecasted Load"]  # complete target
        task = LazyTask(
            _cfg(cache_home=tmp_path, targets=["Actual Load"]),
            dataframe=aligned,
        )
        task.prepare_data()
        assert task.df_pipeline.index.max() == aligned.index.max()
        assert pd.to_datetime(task.run_state.data_end, utc=True) == aligned.index.max()

    def test_all_columns_as_targets_keeps_extent(self, frontier_df, tmp_path):
        """With targets=None every column is a target; the exog-like column is
        observed through index.max(), so dropna(how='all') keeps the frontier
        rows. Documented behaviour: the predictor-side alignment guard covers
        this configuration."""
        task = LazyTask(_cfg(cache_home=tmp_path, targets=None), dataframe=frontier_df)
        task.prepare_data()
        assert task.df_pipeline.index.max() == frontier_df.index.max()

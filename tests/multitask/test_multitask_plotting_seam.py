# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the plotting seam in spotforecast2-safe.

Verifies that:
- plot_with_outliers() raises NotImplementedError mentioning spotforecast2
- _show_prediction_figure() is a no-op returning None
- _show_prediction_figure_agg() is a no-op returning None
- run(show=True) does not raise (hooks are no-ops)
- runner.run(plot_with_outliers=True) raises NotImplementedError
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.multitask import BaseTask, CleanTask, DefaultsTask, LazyTask
from spotforecast2_safe.multitask.runner import run
from spotforecast2_safe.preprocessing import RollingFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_df(n_weeks: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 24 * 7 * n_weeks
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    return pd.DataFrame({"a": rng.normal(100, 10, n)}, index=idx)


def _fast_factory(config, *, weight_func=None, target=None):
    from lightgbm import LGBMRegressor

    return ForecasterRecursive(
        estimator=LGBMRegressor(n_estimators=5, random_state=42, verbose=-1),
        lags=6,
        window_features=RollingFeatures(stats=["mean"], window_sizes=6),
        weight_func=weight_func,
    )


def _minimal_cfg(tmp_path: Path, **kwargs) -> ConfigMulti:
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        forecaster_factory=_fast_factory,
        cache_home=tmp_path,
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


# ---------------------------------------------------------------------------
# plot_with_outliers()
# ---------------------------------------------------------------------------


class TestPlotWithOutliers:
    def test_raises_not_implemented_after_detect_outliers(self, tmp_path):
        df = _synth_df()
        task = LazyTask(_minimal_cfg(tmp_path), dataframe=df)
        task.prepare_data().detect_outliers()
        with pytest.raises(NotImplementedError):
            task.plot_with_outliers()

    def test_error_message_mentions_spotforecast2(self, tmp_path):
        df = _synth_df()
        task = LazyTask(_minimal_cfg(tmp_path), dataframe=df)
        task.prepare_data().detect_outliers()
        with pytest.raises(NotImplementedError, match="spotforecast2"):
            task.plot_with_outliers()

    def test_raises_runtime_error_before_detect_outliers(self, tmp_path):
        df = _synth_df()
        task = LazyTask(_minimal_cfg(tmp_path), dataframe=df)
        task.prepare_data()
        with pytest.raises(RuntimeError, match="detect_outliers"):
            task.plot_with_outliers()

    def test_default_task_also_raises_not_implemented(self, tmp_path):
        df = _synth_df()
        task = DefaultsTask(_minimal_cfg(tmp_path), dataframe=df)
        task.prepare_data().detect_outliers()
        with pytest.raises(NotImplementedError, match="spotforecast2"):
            task.plot_with_outliers()

    def test_clean_task_raises_not_implemented(self, tmp_path):
        """CleanTask inherits plot_with_outliers from BaseTask."""
        cfg = ConfigMulti(cache_home=tmp_path)
        task = CleanTask(cfg)
        # Simulate detect_outliers having run by setting df_pipeline_original
        task.df_pipeline_original = pd.DataFrame()
        with pytest.raises(NotImplementedError, match="spotforecast2"):
            task.plot_with_outliers()


# ---------------------------------------------------------------------------
# _show_prediction_figure (no-op hook)
# ---------------------------------------------------------------------------


class TestShowPredictionFigureHook:
    def test_exists_on_base_task(self):
        assert hasattr(BaseTask, "_show_prediction_figure")
        assert callable(BaseTask._show_prediction_figure)

    def test_returns_none(self, tmp_path):
        task = LazyTask(_minimal_cfg(tmp_path))
        pkg = {"future_pred": pd.Series([1.0, 2.0])}
        result = task._show_prediction_figure(pkg, "target_a", "test task label")
        assert result is None

    def test_does_not_raise(self, tmp_path):
        task = LazyTask(_minimal_cfg(tmp_path))
        pkg = {"future_pred": pd.Series([1.0])}
        task._show_prediction_figure(pkg, "any_target", "any task")  # must not raise

    def test_called_with_show_true_during_strategy_run(self, tmp_path):
        """show=True must not raise (hooks are no-ops in this package)."""
        df = _synth_df()
        task = LazyTask(_minimal_cfg(tmp_path), dataframe=df)
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        task.run(show=True)  # must not raise


# ---------------------------------------------------------------------------
# _show_prediction_figure_agg (no-op hook)
# ---------------------------------------------------------------------------


class TestShowPredictionFigureAggHook:
    def test_exists_on_base_task(self):
        assert hasattr(BaseTask, "_show_prediction_figure_agg")
        assert callable(BaseTask._show_prediction_figure_agg)

    def test_returns_none(self, tmp_path):
        task = LazyTask(_minimal_cfg(tmp_path))
        pkg = {"future_pred": pd.Series([1.0, 2.0])}
        result = task._show_prediction_figure_agg(pkg, "aggregated task label")
        assert result is None

    def test_does_not_raise(self, tmp_path):
        task = LazyTask(_minimal_cfg(tmp_path))
        pkg = {"future_pred": pd.Series([1.0])}
        task._show_prediction_figure_agg(pkg, "any task")  # must not raise


# ---------------------------------------------------------------------------
# runner.run with plot_with_outliers=True
# ---------------------------------------------------------------------------


class TestRunnerPlotWithOutliers:
    def test_raises_not_implemented(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(tmp_path)
        with pytest.raises(NotImplementedError):
            run(
                config=cfg,
                task="defaults",
                dataframe=df,
                cache_home=tmp_path,
                plot_with_outliers=True,
            )

    def test_error_message_mentions_spotforecast2(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(tmp_path)
        with pytest.raises(NotImplementedError, match="spotforecast2"):
            run(
                config=cfg,
                task="defaults",
                dataframe=df,
                cache_home=tmp_path,
                plot_with_outliers=True,
            )

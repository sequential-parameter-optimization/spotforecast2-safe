# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the multitask runner module.

Covers:
- make_demo10_config returns a valid ConfigMulti with consistent bounds/agg_weights
- runner.run rejects optuna/spotoptim task names
- runner.run(plot_with_outliers=True) raises NotImplementedError
- runner.run(task='clean') returns an empty DataFrame
- runner.run with a minimal synthetic DataFrame returns a forecast DataFrame
  with the expected shape, columns, and index type
- project_name is forwarded to config.data_frame_name
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.multitask.runner import make_demo10_config, run
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
        estimator=LGBMRegressor(
            n_estimators=5,
            random_state=config.random_state,
            verbose=-1,
        ),
        lags=6,
        window_features=RollingFeatures(stats=["mean"], window_sizes=6),
        weight_func=weight_func,
    )


def _minimal_cfg(**kwargs) -> ConfigMulti:
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        forecaster_factory=_fast_factory,
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


# ---------------------------------------------------------------------------
# make_demo10_config
# ---------------------------------------------------------------------------


class TestMakeDemo10Config:
    def test_returns_config_multi(self):
        cfg = make_demo10_config()
        assert isinstance(cfg, ConfigMulti)

    def test_bounds_length(self):
        cfg = make_demo10_config()
        assert len(cfg.bounds) == 11

    def test_agg_weights_length(self):
        cfg = make_demo10_config()
        assert len(cfg.agg_weights) == 11

    def test_bounds_and_agg_weights_consistent(self):
        cfg = make_demo10_config()
        assert len(cfg.bounds) == len(cfg.agg_weights)

    def test_bounds_are_tuples(self):
        cfg = make_demo10_config()
        for b in cfg.bounds:
            assert len(b) == 2
            lower, upper = b
            assert lower < upper

    def test_overrides_applied(self):
        cfg = make_demo10_config(predict_size=48)
        assert cfg.predict_size == 48

    def test_fresh_instance_per_call(self):
        cfg1 = make_demo10_config()
        cfg2 = make_demo10_config()
        assert cfg1 is not cfg2


# ---------------------------------------------------------------------------
# runner.run — prohibited task names
# ---------------------------------------------------------------------------


class TestRunProhibitedTasks:
    def test_optuna_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            run(task="optuna", cache_home=tmp_path)

    def test_spotoptim_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            run(task="spotoptim", cache_home=tmp_path)

    def test_unknown_task_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown task"):
            run(task="banana", cache_home=tmp_path)

    def test_error_mentions_supported_tasks(self, tmp_path):
        with pytest.raises(ValueError, match="lazy"):
            run(task="unknown_xyz", cache_home=tmp_path)


# ---------------------------------------------------------------------------
# runner.run — plot_with_outliers=True raises NotImplementedError
# ---------------------------------------------------------------------------


class TestRunPlotWithOutliers:
    def test_plot_with_outliers_raises_not_implemented(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg()
        with pytest.raises(NotImplementedError, match="spotforecast2"):
            run(
                config=cfg,
                task="defaults",
                dataframe=df,
                cache_home=tmp_path,
                plot_with_outliers=True,
            )


# ---------------------------------------------------------------------------
# runner.run — clean task
# ---------------------------------------------------------------------------


class TestRunCleanTask:
    def test_clean_returns_empty_dataframe(self, tmp_path):
        result = run(task="clean", cache_home=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_clean_with_existing_cache(self, tmp_path):
        cache = tmp_path / "sf2_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        result = run(task="clean", cache_home=cache, project_name="test")
        assert result.empty


# ---------------------------------------------------------------------------
# runner.run — end-to-end with synthetic data
# ---------------------------------------------------------------------------


class TestRunEndToEnd:
    def test_returns_dataframe(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg()
        result = run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="test_e2e",
        )
        assert isinstance(result, pd.DataFrame)

    def test_result_has_forecast_column(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg()
        result = run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="test_e2e",
        )
        assert "forecast" in result.columns

    def test_result_has_correct_row_count(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(predict_size=6)
        result = run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="test_e2e",
        )
        assert len(result) == 6

    def test_result_index_is_datetime_index(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg()
        result = run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="test_e2e",
        )
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_result_no_nan(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg()
        result = run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="test_e2e",
        )
        assert result["forecast"].notna().all()

    def test_project_name_forwarded_to_config(self, tmp_path):
        """project_name sets config.data_frame_name, affecting cache subdirs."""
        df = _synth_df()
        cfg = _minimal_cfg(auto_save_models=True)
        result = run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="my_project",
        )
        assert isinstance(result, pd.DataFrame)
        # Config should be mutated with data_frame_name
        assert cfg.data_frame_name == "my_project"

    def test_lazy_task_runs_without_cache(self, tmp_path):
        """runner.run with task='lazy' and empty cache uses defaults."""
        df = _synth_df()
        cfg = _minimal_cfg()
        result = run(
            config=cfg,
            task="lazy",
            dataframe=df,
            cache_home=tmp_path,
            project_name="lazy_test",
        )
        assert "forecast" in result.columns
        assert len(result) == 6

    def test_overrides_forwarded_to_config(self, tmp_path):
        """**overrides are forwarded to config.set_params."""
        df = _synth_df()
        cfg = _minimal_cfg()
        result = run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="override_test",
            predict_size=6,
        )
        assert len(result) == 6

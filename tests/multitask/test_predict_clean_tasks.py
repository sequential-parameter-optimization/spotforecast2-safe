# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for PredictTask and CleanTask.

Covers:
- PredictTask.run() raises RuntimeError when no saved models
- PredictTask loads and predicts after a DefaultsTask run with auto_save=True
- PredictTask prediction package has correct keys
- CleanTask dry_run lists without deleting
- CleanTask real run deletes the cache
- CleanTask on a non-existent directory returns status='empty'
- CleanTask can target a custom cache_home
- execute_clean and CleanTask.run() both work
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.multitask import CleanTask, DefaultsTask, PredictTask
from spotforecast2_safe.multitask.clean import execute_clean
from spotforecast2_safe.preprocessing import RollingFeatures

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
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


def _minimal_cfg(tmp_path: Path, auto_save: bool = False, **kwargs) -> ConfigMulti:
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=auto_save,
        number_folds=2,
        forecaster_factory=_fast_factory,
        cache_home=tmp_path,
        data_frame_name="test_proj",
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


def _run_defaults_pipeline(df: pd.DataFrame, cfg: ConfigMulti) -> DefaultsTask:
    """Run the full DefaultsTask pipeline and return the task object."""
    task = DefaultsTask(cfg, dataframe=df.copy())
    task.prepare_data().detect_outliers().impute().build_exogenous_features()
    task.run()
    return task


def _prepare_pipeline(task):
    """Run prepare + outliers + impute + exog on a task instance."""
    task.prepare_data().detect_outliers().impute().build_exogenous_features()
    return task


# ---------------------------------------------------------------------------
# PredictTask — error when no models
# ---------------------------------------------------------------------------


class TestPredictTaskNoModels:
    def test_raises_runtime_error_when_no_models(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(tmp_path)
        predict_task = PredictTask(cfg, dataframe=df)
        _prepare_pipeline(predict_task)
        with pytest.raises(RuntimeError, match="No saved models"):
            predict_task.run()

    def test_error_message_mentions_task_names(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(tmp_path)
        predict_task = PredictTask(cfg, dataframe=df)
        _prepare_pipeline(predict_task)
        with pytest.raises(RuntimeError, match="LazyTask"):
            predict_task.run()


# ---------------------------------------------------------------------------
# PredictTask — load and predict after DefaultsTask
# ---------------------------------------------------------------------------


class TestPredictTaskAfterDefaults:
    def test_predicts_after_training_run(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(tmp_path, auto_save=True)

        # Train and auto-save
        _run_defaults_pipeline(df, cfg)

        # Predict from saved models
        predict_task = PredictTask(cfg, dataframe=df.copy())
        _prepare_pipeline(predict_task)
        result = predict_task.run()
        assert "future_pred" in result
        assert len(result["future_pred"]) == 6

    def test_predict_result_not_all_nan(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(tmp_path, auto_save=True)
        _run_defaults_pipeline(df, cfg)

        predict_task = PredictTask(cfg, dataframe=df.copy())
        _prepare_pipeline(predict_task)
        result = predict_task.run()
        assert result["future_pred"].notna().all()

    def test_predict_result_stored_in_results(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(tmp_path, auto_save=True)
        _run_defaults_pipeline(df, cfg)

        predict_task = PredictTask(cfg, dataframe=df.copy())
        _prepare_pipeline(predict_task)
        predict_task.run()
        assert "predict" in predict_task.results
        assert "a" in predict_task.results["predict"]

    def test_predict_package_has_required_keys(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(tmp_path, auto_save=True)
        _run_defaults_pipeline(df, cfg)

        predict_task = PredictTask(cfg, dataframe=df.copy())
        _prepare_pipeline(predict_task)
        result = predict_task.run()
        for key in ("future_pred", "train_pred", "train_actual"):
            assert key in result

    def test_predict_task_name_attribute(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        task = PredictTask(cfg)
        assert task.TASK == "predict"
        assert task._task_name == "predict"


# ---------------------------------------------------------------------------
# CleanTask — dry_run
# ---------------------------------------------------------------------------


class TestCleanTaskDryRun:
    def test_dry_run_returns_dry_run_status(self, tmp_path):
        cache = tmp_path / "sf2_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        result = task.run(dry_run=True)
        assert result["status"] == "dry_run"

    def test_dry_run_lists_items(self, tmp_path):
        cache = tmp_path / "sf2_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        (cache / "tuning_results").mkdir()
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        result = task.run(dry_run=True)
        assert "models" in result["deleted_items"]
        assert "tuning_results" in result["deleted_items"]

    def test_dry_run_does_not_delete(self, tmp_path):
        cache = tmp_path / "sf2_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        task.run(dry_run=True)
        assert cache.exists()
        assert (cache / "models").exists()

    def test_dry_run_cache_dir_in_result(self, tmp_path):
        cache = tmp_path / "sf2_cache"
        cache.mkdir()
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        result = task.run(dry_run=True)
        assert result["cache_dir"] == cache


# ---------------------------------------------------------------------------
# CleanTask — real clean
# ---------------------------------------------------------------------------


class TestCleanTaskRealRun:
    def test_real_run_deletes_cache(self, tmp_path):
        cache = tmp_path / "sf2_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        result = task.run()
        assert result["status"] == "success"
        assert not cache.exists()

    def test_real_run_returns_deleted_items(self, tmp_path):
        cache = tmp_path / "sf2_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        (cache / "tuning_results").mkdir()
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        result = task.run()
        assert "models" in result["deleted_items"]

    def test_nonexistent_dir_clean_succeeds(self, tmp_path):
        """CleanTask creates the logging subdir on init, so run() cleans it.

        The cache dir is created by BaseTask._attach_file_handler before
        execute_clean is called; the result is 'success' (not 'empty') because
        the directory exists by the time run() runs.
        """
        cache = tmp_path / "nonexistent_cache"
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        # After init the logging/ subdir has been created
        assert cache.exists()
        result = task.run()
        assert result["status"] == "success"
        assert not cache.exists()

    def test_clean_after_pipeline_run(self, tmp_path):
        """Full pipeline: train with auto_save, then clean, verify cache gone."""
        df = _synth_df()
        cache = tmp_path / "pipeline_cache"
        cfg = _minimal_cfg(cache, auto_save=True)
        _run_defaults_pipeline(df, cfg)
        assert cache.exists()

        clean_task = CleanTask(cfg)
        result = clean_task.run()
        assert result["status"] == "success"
        assert not cache.exists()

    def test_clean_with_custom_cache_home_kwarg(self, tmp_path):
        """cache_home kwarg to run() overrides config.cache_home."""
        custom_cache = tmp_path / "custom"
        custom_cache.mkdir()
        (custom_cache / "models").mkdir()

        cfg = ConfigMulti(cache_home=tmp_path / "other", data_frame_name="proj")
        task = CleanTask(cfg)
        result = task.run(cache_home=custom_cache)
        assert result["status"] == "success"
        assert not custom_cache.exists()


# ---------------------------------------------------------------------------
# execute_clean function
# ---------------------------------------------------------------------------


class TestExecuteClean:
    def test_empty_status_via_execute_clean_directly(self, tmp_path):
        """execute_clean returns 'empty' when the target dir does not exist.

        This case only arises when execute_clean is called directly with a
        cache_home that was never created — bypassing the logging/ creation
        that happens in BaseTask.__init__.
        """
        nonexistent = tmp_path / "truly_nonexistent"
        cfg = ConfigMulti(cache_home=tmp_path, data_frame_name="proj")
        task = CleanTask(cfg)
        result = execute_clean(task, cache_home=nonexistent)
        assert result["status"] == "empty"
        assert result["deleted_items"] == []

    def test_execute_clean_dry_run(self, tmp_path):
        cache = tmp_path / "ec_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        result = execute_clean(task, dry_run=True)
        assert result["status"] == "dry_run"
        assert cache.exists()

    def test_execute_clean_real(self, tmp_path):
        cache = tmp_path / "ec_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        cfg = ConfigMulti(cache_home=cache, data_frame_name="proj")
        task = CleanTask(cfg)
        result = execute_clean(task)
        assert result["status"] == "success"
        assert not cache.exists()

    def test_execute_clean_cache_home_override(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "models").mkdir()
        cfg = ConfigMulti(cache_home=tmp_path / "other", data_frame_name="proj")
        task = CleanTask(cfg)
        result = execute_clean(task, cache_home=target)
        assert result["status"] == "success"
        assert not target.exists()

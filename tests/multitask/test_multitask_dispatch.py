# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the MultiTask run() dispatcher.

Covers:
- task="optuna" / task="spotoptim" raise ValueError naming spotforecast2
- Unknown task name raises ValueError
- "lazy", "defaults", "predict", "clean" dispatch to the right execute_* function
- run_task_* convenience methods delegate to execute_* functions
- TASK attribute set correctly from constructor and run(task=...) override
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.multitask import DefaultsTask, LazyTask, MultiTask

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _synth_df(n_weeks: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 24 * 7 * n_weeks
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    return pd.DataFrame({"a": rng.normal(100, 10, n)}, index=idx)


def _minimal_cfg(tmp_path: Path, **kwargs) -> ConfigMulti:
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        cache_home=tmp_path,
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


# ---------------------------------------------------------------------------
# Prohibited task names: optuna and spotoptim
# ---------------------------------------------------------------------------


class TestProhibitedTasks:
    """optuna and spotoptim raise ValueError with spotforecast2 in the message."""

    def test_optuna_raises_value_error(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        with pytest.raises(ValueError, match="spotforecast2"):
            mt.run(task="optuna")

    def test_spotoptim_raises_value_error(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        with pytest.raises(ValueError, match="spotforecast2"):
            mt.run(task="spotoptim")

    def test_optuna_error_message_contains_task_name(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        with pytest.raises(ValueError, match="optuna"):
            mt.run(task="optuna")

    def test_spotoptim_error_message_contains_task_name(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        with pytest.raises(ValueError, match="spotoptim"):
            mt.run(task="spotoptim")


# ---------------------------------------------------------------------------
# Unknown task
# ---------------------------------------------------------------------------


class TestUnknownTask:
    def test_unknown_task_raises_value_error(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        with pytest.raises(ValueError, match="Unknown task"):
            mt.run(task="no_such_task")

    def test_unknown_task_message_includes_task_name(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        with pytest.raises(ValueError, match="banana"):
            mt.run(task="banana")

    def test_unexpected_kwargs_raise_type_error(self, tmp_path):
        """Fail-safe: per-task options must not be silently dropped."""
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        with pytest.raises(TypeError, match="use_tuned_params"):
            mt.run(task="lazy", use_tuned_params=False)


# ---------------------------------------------------------------------------
# TASK attribute
# ---------------------------------------------------------------------------


class TestTaskAttribute:
    def test_default_task_is_lazy(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg)
        assert mt.TASK == "lazy"

    def test_constructor_task_defaults(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, task="defaults")
        assert mt.TASK == "defaults"

    def test_constructor_task_predict(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, task="predict")
        assert mt.TASK == "predict"

    def test_constructor_task_clean(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, task="clean")
        assert mt.TASK == "clean"


# ---------------------------------------------------------------------------
# Dispatch to correct execute_* functions (sentinel monkeypatching)
# ---------------------------------------------------------------------------


class TestDispatch:
    """Verify run(task=X) calls the correct execute_* function."""

    def test_run_lazy_calls_execute_lazy(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        sentinel: Dict[str, Any] = {"future_pred": pd.Series([1.0])}
        with patch(
            "spotforecast2_safe.multitask.multi.execute_lazy",
            return_value=sentinel,
        ) as mock_exec:
            mt.prepare_data()
            result = mt.run(task="lazy")
        mock_exec.assert_called_once()
        assert result is sentinel

    def test_run_defaults_calls_execute_defaults(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        sentinel: Dict[str, Any] = {"future_pred": pd.Series([2.0])}
        with patch(
            "spotforecast2_safe.multitask.multi.execute_defaults",
            return_value=sentinel,
        ) as mock_exec:
            mt.prepare_data()
            result = mt.run(task="defaults")
        mock_exec.assert_called_once()
        assert result is sentinel

    def test_run_predict_calls_execute_predict(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        sentinel: Dict[str, Any] = {"future_pred": pd.Series([3.0])}
        with patch(
            "spotforecast2_safe.multitask.multi.execute_predict",
            return_value=sentinel,
        ) as mock_exec:
            mt.prepare_data()
            result = mt.run(task="predict")
        mock_exec.assert_called_once()
        assert result is sentinel

    def test_run_clean_calls_execute_clean(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg)
        sentinel = {"status": "empty", "cache_dir": tmp_path, "deleted_items": []}
        with patch(
            "spotforecast2_safe.multitask.multi.execute_clean",
            return_value=sentinel,
        ) as mock_exec:
            result = mt.run(task="clean")
        mock_exec.assert_called_once()
        assert result is sentinel


# ---------------------------------------------------------------------------
# run_task_* convenience methods
# ---------------------------------------------------------------------------


class TestRunTaskMethods:
    """Verify the run_task_* helpers exist and delegate correctly."""

    def test_has_run_task_lazy(self, tmp_path):
        assert callable(MultiTask(_minimal_cfg(tmp_path)).run_task_lazy)

    def test_has_run_task_defaults(self, tmp_path):
        assert callable(MultiTask(_minimal_cfg(tmp_path)).run_task_defaults)

    def test_has_run_task_predict(self, tmp_path):
        assert callable(MultiTask(_minimal_cfg(tmp_path)).run_task_predict)

    def test_has_run_task_clean(self, tmp_path):
        assert callable(MultiTask(_minimal_cfg(tmp_path)).run_task_clean)

    def test_run_task_lazy_delegates_to_execute_lazy(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        sentinel = {"future_pred": pd.Series([1.0])}
        with patch(
            "spotforecast2_safe.multitask.multi.execute_lazy",
            return_value=sentinel,
        ) as mock_exec:
            mt.prepare_data()
            result = mt.run_task_lazy()
        mock_exec.assert_called_once()
        assert result is sentinel

    def test_run_task_defaults_delegates_to_execute_defaults(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg, dataframe=_synth_df())
        sentinel = {"future_pred": pd.Series([2.0])}
        with patch(
            "spotforecast2_safe.multitask.multi.execute_defaults",
            return_value=sentinel,
        ) as mock_exec:
            mt.prepare_data()
            result = mt.run_task_defaults()
        mock_exec.assert_called_once()
        assert result is sentinel

    def test_run_task_clean_delegates_to_execute_clean(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        mt = MultiTask(cfg)
        sentinel = {"status": "empty", "cache_dir": tmp_path, "deleted_items": []}
        with patch(
            "spotforecast2_safe.multitask.multi.execute_clean",
            return_value=sentinel,
        ) as mock_exec:
            result = mt.run_task_clean()
        mock_exec.assert_called_once()
        assert result is sentinel


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    """Verify class hierarchy for the multitask package."""

    def test_multitask_is_subclass_of_base_task(self):
        from spotforecast2_safe.multitask.base import BaseTask

        assert issubclass(MultiTask, BaseTask)

    def test_lazy_task_is_subclass_of_base_task(self):
        from spotforecast2_safe.multitask.base import BaseTask

        assert issubclass(LazyTask, BaseTask)

    def test_defaults_task_is_subclass_of_base_task(self):
        from spotforecast2_safe.multitask.base import BaseTask

        assert issubclass(DefaultsTask, BaseTask)

    def test_multitask_instance_is_base_task(self, tmp_path):
        from spotforecast2_safe.multitask.base import BaseTask

        assert isinstance(MultiTask(_minimal_cfg(tmp_path)), BaseTask)

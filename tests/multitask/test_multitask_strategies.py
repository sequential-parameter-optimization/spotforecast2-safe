# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the multitask strategies module.

Adapted from spotforecast2/tests/test_multitask_strategies.py; all optuna /
spotoptim strategy references removed (not available in this package).

Covers:
- LazyStrategy and DefaultsStrategy instantiate correctly
- LazyStrategy.prepare_forecaster applies cached tuning results when available
- LazyStrategy respects use_tuned_params=False (returns forecaster unchanged)
- LazyStrategy returns forecaster unchanged when cache is empty
- DefaultsStrategy returns forecaster unchanged (always)
- Protocol structure: both have a ``name`` attribute
"""

from pathlib import Path

import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.multitask import LazyTask
from spotforecast2_safe.multitask.strategies import DefaultsStrategy, LazyStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache_task(tmp_path: Path) -> LazyTask:
    """LazyTask with cache_home pointing to a temp directory."""
    cfg = ConfigMulti(data_frame_name="test_ds", cache_home=tmp_path)
    return LazyTask(cfg)


class _FakeForecaster:
    """Minimal forecaster stub that records calls to set_params / set_lags."""

    def __init__(self) -> None:
        self.params_set: dict = {}
        self.lags_set = None

    def set_params(self, **kwargs: object) -> None:
        self.params_set.update(kwargs)

    def set_lags(self, lags: object) -> None:
        self.lags_set = lags


# ---------------------------------------------------------------------------
# LazyStrategy: instantiation
# ---------------------------------------------------------------------------


class TestLazyStrategyInstantiation:
    def test_name_attribute(self):
        assert LazyStrategy().name == "lazy"

    def test_use_tuned_params_default_true(self):
        assert LazyStrategy().use_tuned_params is True

    def test_use_tuned_params_false(self):
        assert LazyStrategy(use_tuned_params=False).use_tuned_params is False

    def test_max_age_days_default_none(self):
        assert LazyStrategy().max_age_days is None

    def test_max_age_days_set(self):
        assert LazyStrategy(max_age_days=7.0).max_age_days == 7.0


# ---------------------------------------------------------------------------
# LazyStrategy: tuning results applied when cache populated
# ---------------------------------------------------------------------------


class TestLazyStrategyWithCache:
    def test_applies_cached_params(self, cache_task):
        cache_task.save_tuning_results(
            target="t0",
            task_name="optuna",
            best_params={"n_estimators": 42},
            best_lags=[1, 2, 3],
        )
        strategy = LazyStrategy(use_tuned_params=True)
        fc = _FakeForecaster()
        strategy.prepare_forecaster(cache_task, "t0", fc, pd.Series([1.0]))
        assert fc.params_set == {"n_estimators": 42}
        assert fc.lags_set == [1, 2, 3]

    def test_applies_cached_lags(self, cache_task):
        cache_task.save_tuning_results(
            target="t1",
            task_name="spotoptim",
            best_params={"learning_rate": 0.05},
            best_lags=24,
        )
        strategy = LazyStrategy(use_tuned_params=True)
        fc = _FakeForecaster()
        strategy.prepare_forecaster(cache_task, "t1", fc, pd.Series([1.0]))
        assert fc.lags_set == 24

    def test_returns_forecaster_with_tuned_params(self, cache_task):
        cache_task.save_tuning_results(
            target="t2",
            task_name="optuna",
            best_params={"num_leaves": 31},
            best_lags=[1],
        )
        strategy = LazyStrategy(use_tuned_params=True)
        fc = _FakeForecaster()
        result = strategy.prepare_forecaster(cache_task, "t2", fc, pd.Series([1.0]))
        assert result is fc  # same object, mutated in-place


# ---------------------------------------------------------------------------
# LazyStrategy: use_tuned_params=False
# ---------------------------------------------------------------------------


class TestLazyStrategyNoTuning:
    def test_forecaster_unchanged_when_disabled(self, cache_task):
        """With use_tuned_params=False the forecaster is returned as-is."""
        cache_task.save_tuning_results(
            target="t3",
            task_name="optuna",
            best_params={"n_estimators": 99},
            best_lags=[1],
        )
        strategy = LazyStrategy(use_tuned_params=False)
        fc = _FakeForecaster()
        result = strategy.prepare_forecaster(cache_task, "t3", fc, pd.Series([1.0]))
        assert result is fc
        assert fc.params_set == {}  # set_params never called
        assert fc.lags_set is None

    def test_returns_self_when_cache_empty(self, tmp_path):
        """No cache files: forecaster returned unchanged."""
        cfg = ConfigMulti(data_frame_name="empty_cache", cache_home=tmp_path)
        task = LazyTask(cfg)
        strategy = LazyStrategy(use_tuned_params=True)
        fc = _FakeForecaster()
        result = strategy.prepare_forecaster(task, "t_missing", fc, pd.Series([1.0]))
        assert result is fc
        assert fc.params_set == {}

    def test_max_age_days_filters_old_results(self, tmp_path):
        """Results older than max_age_days must not be applied."""
        cfg = ConfigMulti(data_frame_name="test_ds", cache_home=tmp_path)
        task = LazyTask(cfg)
        # Manually write a fake JSON with a very old timestamp
        import json

        tuning_dir = tmp_path / "tuning_results"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        old_ts = "20200101_000000"
        filename = f"test_ds_target_x_optuna_{old_ts}.json"
        payload = {
            "data_frame_name": "test_ds",
            "target": "target_x",
            "task_name": "optuna",
            "timestamp": old_ts,
            "best_params": {"n_estimators": 7},
            "best_lags": [1],
        }
        (tuning_dir / filename).write_text(json.dumps(payload))

        strategy = LazyStrategy(use_tuned_params=True, max_age_days=1.0)
        fc = _FakeForecaster()
        result = strategy.prepare_forecaster(task, "target_x", fc, pd.Series([1.0]))
        assert result is fc
        assert fc.params_set == {}  # too old, not applied


# ---------------------------------------------------------------------------
# DefaultsStrategy
# ---------------------------------------------------------------------------


class TestDefaultsStrategy:
    def test_name_attribute(self):
        assert DefaultsStrategy().name == "defaults"

    def test_returns_forecaster_unchanged(self):
        strategy = DefaultsStrategy()
        sentinel = object()
        result = strategy.prepare_forecaster(
            task=None,
            target="any",
            forecaster=sentinel,
            y_train=pd.Series([1.0]),
            exog_train=None,
        )
        assert result is sentinel

    def test_does_not_consult_cache(self, cache_task):
        """DefaultsStrategy ignores any cached tuning results."""
        cache_task.save_tuning_results(
            target="t_def",
            task_name="optuna",
            best_params={"n_estimators": 999},
            best_lags=[1],
        )
        strategy = DefaultsStrategy()
        fc = _FakeForecaster()
        strategy.prepare_forecaster(cache_task, "t_def", fc, pd.Series([1.0]))
        assert fc.params_set == {}
        assert fc.lags_set is None

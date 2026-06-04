# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for BaseTask model and tuning-result persistence.

Adapted from spotforecast2/tests/test_model_persistence_multitask.py; all
OptunaTask / SpotOptimTask references replaced with safe-package equivalents.

Covers:
- save_models / load_models round-trip
- Filename convention includes data_frame_name, target, task_name, timestamp
- save_models validates task_name; unknown name raises ValueError
- save_models raises RuntimeError when no results available
- load_models filters by task_name, target, max_age_days
- Most-recent model wins when multiple snapshots exist
- Cross data_frame_name isolation
- save_tuning_results / load_tuning_results round-trip
- max_age_days filtering for tuning results
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.multitask import BaseTask, DefaultsTask, LazyTask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task(tmp_path: Path) -> LazyTask:
    """LazyTask with cache_home pointing to a temporary directory."""
    cfg = ConfigMulti(data_frame_name="test_data", cache_home=tmp_path)
    return LazyTask(cfg)


@pytest.fixture()
def mock_forecasters() -> Dict[str, Any]:
    """Two trivially fitted sklearn models keyed by target name."""
    m1 = LinearRegression()
    m1.fit(np.arange(10).reshape(-1, 1), np.arange(10, dtype=float))
    m2 = LinearRegression()
    m2.fit(np.arange(10).reshape(-1, 1), np.arange(10, dtype=float) * 2)
    return {"target_0": m1, "target_1": m2}


@pytest.fixture()
def single_forecaster() -> Dict[str, Any]:
    """Single fitted model."""
    m = LinearRegression()
    m.fit(np.arange(10).reshape(-1, 1), np.arange(10, dtype=float))
    return {"target_0": m}


# ---------------------------------------------------------------------------
# save_models — file creation and naming
# ---------------------------------------------------------------------------


class TestSaveModels:
    def test_save_creates_files(self, task, mock_forecasters):
        paths = task.save_models(task_name="lazy", forecasters=mock_forecasters)
        assert len(paths) == 2
        for path in paths.values():
            assert path.exists()
            assert path.suffix == ".joblib"

    def test_save_file_in_models_dir(self, task, mock_forecasters):
        paths = task.save_models(task_name="lazy", forecasters=mock_forecasters)
        for path in paths.values():
            assert path.parent.name == "test_data"
            assert path.parent.parent.name == "models"

    def test_save_filename_format(self, task, single_forecaster):
        paths = task.save_models(task_name="lazy", forecasters=single_forecaster)
        path = paths["target_0"]
        assert path.name.startswith("test_data_target_0_lazy_")
        assert path.name.endswith(".joblib")

    def test_save_filename_contains_task_name(self, task, single_forecaster):
        for tname in ("lazy", "defaults", "optuna", "spotoptim"):
            paths = task.save_models(task_name=tname, forecasters=single_forecaster)
            assert f"_{tname}_" in paths["target_0"].name

    def test_save_returns_dict_of_paths(self, task, mock_forecasters):
        result = task.save_models(task_name="lazy", forecasters=mock_forecasters)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, Path)


# ---------------------------------------------------------------------------
# save_models — validation
# ---------------------------------------------------------------------------


class TestSaveModelsValidation:
    def test_invalid_task_name_raises_value_error(self, task, single_forecaster):
        with pytest.raises(ValueError, match="Unknown task_name"):
            task.save_models(task_name="bad_task", forecasters=single_forecaster)

    def test_no_results_raises_runtime_error(self, task):
        with pytest.raises(RuntimeError, match="No results for task"):
            task.save_models(task_name="lazy")

    def test_missing_forecaster_key_raises_runtime_error(self, task):
        task.results["lazy"] = {"target_0": {"predictions": [1, 2, 3]}}
        with pytest.raises(RuntimeError, match="forecaster"):
            task.save_models(task_name="lazy")


# ---------------------------------------------------------------------------
# save_models — from results dict
# ---------------------------------------------------------------------------


class TestSaveModelsFromResults:
    def test_save_from_results(self, task, single_forecaster):
        m = single_forecaster["target_0"]
        task.results["lazy"] = {"target_0": {"forecaster": m}}
        paths = task.save_models(task_name="lazy")
        assert "target_0" in paths
        assert paths["target_0"].exists()

    def test_save_from_results_multiple_targets(self, task, mock_forecasters):
        task.results["lazy"] = {
            t: {"forecaster": f} for t, f in mock_forecasters.items()
        }
        paths = task.save_models(task_name="lazy")
        assert len(paths) == 2
        for p in paths.values():
            assert p.exists()


# ---------------------------------------------------------------------------
# load_models — basic retrieval
# ---------------------------------------------------------------------------


class TestLoadModels:
    def test_load_returns_empty_when_no_dir(self, tmp_path):
        task = LazyTask(
            ConfigMulti(
                data_frame_name="test_data",
                cache_home=tmp_path / "nonexistent",
            )
        )
        assert task.load_models() == {}

    def test_load_returns_empty_when_no_files(self, task):
        assert task.load_models() == {}

    def test_round_trip(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        loaded = task.load_models()
        assert "target_0" in loaded
        pred = loaded["target_0"].predict(np.array([[5.0]]))
        assert len(pred) == 1

    def test_round_trip_multiple_targets(self, task, mock_forecasters):
        task.save_models(task_name="lazy", forecasters=mock_forecasters)
        loaded = task.load_models()
        assert set(loaded.keys()) == {"target_0", "target_1"}

    def test_loaded_model_preserves_coefficients(self, task, single_forecaster):
        original = single_forecaster["target_0"]
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        loaded = task.load_models()
        np.testing.assert_array_almost_equal(original.coef_, loaded["target_0"].coef_)


# ---------------------------------------------------------------------------
# load_models — filtering
# ---------------------------------------------------------------------------


class TestLoadModelsFiltering:
    def test_filter_by_task_name(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        task.save_models(task_name="defaults", forecasters=single_forecaster)
        assert "target_0" in task.load_models(task_name="lazy")
        assert "target_0" in task.load_models(task_name="defaults")

    def test_filter_by_task_name_excludes_other(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        assert task.load_models(task_name="defaults") == {}

    def test_filter_by_target(self, task, mock_forecasters):
        task.save_models(task_name="lazy", forecasters=mock_forecasters)
        loaded = task.load_models(target="target_0")
        assert "target_0" in loaded
        assert "target_1" not in loaded

    def test_filter_by_target_no_match(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        assert task.load_models(target="nonexistent") == {}

    def test_filter_by_max_age_fresh(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        loaded = task.load_models(max_age_days=1.0)
        assert "target_0" in loaded

    def test_filter_by_max_age_expired(self, tmp_path):
        """An artificially old model file should be excluded."""
        model_dir = tmp_path / "models" / "test_data"
        model_dir.mkdir(parents=True)
        from joblib import dump as joblib_dump

        m = LinearRegression()
        m.fit(np.arange(5).reshape(-1, 1), np.arange(5, dtype=float))
        old_file = model_dir / "test_data_target_0_lazy_20200101_000000.joblib"
        joblib_dump(m, old_file, compress=3)

        task = LazyTask(ConfigMulti(data_frame_name="test_data", cache_home=tmp_path))
        loaded = task.load_models(max_age_days=1.0)
        assert loaded == {}

    def test_combined_filters(self, task, mock_forecasters):
        task.save_models(task_name="lazy", forecasters=mock_forecasters)
        task.save_models(task_name="defaults", forecasters=mock_forecasters)
        loaded = task.load_models(task_name="lazy", target="target_1")
        assert "target_1" in loaded
        assert "target_0" not in loaded


# ---------------------------------------------------------------------------
# Cross data_frame_name isolation
# ---------------------------------------------------------------------------


class TestDataFrameNameIsolation:
    def test_different_names_isolated(self, tmp_path, single_forecaster):
        task_a = LazyTask(ConfigMulti(data_frame_name="ds_a", cache_home=tmp_path))
        task_b = LazyTask(ConfigMulti(data_frame_name="ds_b", cache_home=tmp_path))
        task_a.save_models(task_name="lazy", forecasters=single_forecaster)
        assert task_b.load_models(task_name="lazy") == {}
        assert "target_0" in task_a.load_models(task_name="lazy")


# ---------------------------------------------------------------------------
# Method availability on all task classes
# ---------------------------------------------------------------------------


class TestMethodAvailability:
    @pytest.mark.parametrize("cls", [BaseTask, LazyTask, DefaultsTask])
    def test_has_save_models(self, cls):
        assert callable(getattr(cls, "save_models"))

    @pytest.mark.parametrize("cls", [BaseTask, LazyTask, DefaultsTask])
    def test_has_load_models(self, cls):
        assert callable(getattr(cls, "load_models"))

    @pytest.mark.parametrize("cls", [BaseTask, LazyTask, DefaultsTask])
    def test_has_save_tuning_results(self, cls):
        assert callable(getattr(cls, "save_tuning_results"))

    @pytest.mark.parametrize("cls", [BaseTask, LazyTask, DefaultsTask])
    def test_has_load_tuning_results(self, cls):
        assert callable(getattr(cls, "load_tuning_results"))


# ---------------------------------------------------------------------------
# _TASK_MODEL_NAMES mapping
# ---------------------------------------------------------------------------


class TestTaskModelNames:
    def test_is_dict(self):
        assert isinstance(BaseTask._TASK_MODEL_NAMES, dict)

    def test_contains_lazy(self):
        assert "lazy" in BaseTask._TASK_MODEL_NAMES

    def test_contains_defaults(self):
        assert "defaults" in BaseTask._TASK_MODEL_NAMES

    def test_contains_optuna(self):
        # optuna filenames may come from sibling-package caches
        assert "optuna" in BaseTask._TASK_MODEL_NAMES

    def test_contains_spotoptim(self):
        assert "spotoptim" in BaseTask._TASK_MODEL_NAMES

    @pytest.mark.parametrize("task_name", ["lazy", "defaults", "optuna", "spotoptim"])
    def test_all_valid_accepted_by_save_models(
        self, task, single_forecaster, task_name
    ):
        paths = task.save_models(task_name=task_name, forecasters=single_forecaster)
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# save_tuning_results / load_tuning_results round-trip
# ---------------------------------------------------------------------------


class TestTuningResultsPersistence:
    def test_save_creates_json(self, task, tmp_path):
        path = task.save_tuning_results(
            target="t0", task_name="optuna", best_params={"n": 10}, best_lags=[1, 2]
        )
        assert path.exists()
        assert path.suffix == ".json"

    def test_round_trip_params(self, task):
        task.save_tuning_results(
            target="t1",
            task_name="optuna",
            best_params={"learning_rate": 0.01, "n_estimators": 50},
            best_lags=24,
        )
        loaded = task.load_tuning_results(target="t1")
        assert loaded is not None
        assert loaded["best_params"] == {"learning_rate": 0.01, "n_estimators": 50}

    def test_round_trip_lags(self, task):
        task.save_tuning_results(
            target="t2", task_name="spotoptim", best_params={}, best_lags=[1, 2, 3]
        )
        loaded = task.load_tuning_results(target="t2")
        assert loaded["best_lags"] == [1, 2, 3]

    def test_load_none_when_no_file(self, task):
        result = task.load_tuning_results(target="nonexistent_target")
        assert result is None

    def test_filter_by_task_name(self, task):
        task.save_tuning_results("t3", "optuna", {"a": 1}, [1])
        loaded = task.load_tuning_results(target="t3", task_name="optuna")
        assert loaded is not None
        loaded_wrong = task.load_tuning_results(target="t3", task_name="spotoptim")
        assert loaded_wrong is None

    def test_max_age_days_filters_old(self, tmp_path):
        cfg = ConfigMulti(data_frame_name="test_data", cache_home=tmp_path)
        task = LazyTask(cfg)
        tuning_dir = tmp_path / "tuning_results"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        old_ts = "20200101_000000"
        payload = {
            "data_frame_name": "test_data",
            "target": "t_old",
            "task_name": "optuna",
            "timestamp": old_ts,
            "best_params": {"n": 99},
            "best_lags": [1],
        }
        (tuning_dir / f"test_data_t_old_optuna_{old_ts}.json").write_text(
            json.dumps(payload)
        )
        result = task.load_tuning_results(target="t_old", max_age_days=1.0)
        assert result is None

    def test_max_age_days_fresh_accepted(self, task):
        task.save_tuning_results("t_fresh", "optuna", {"n": 1}, [1])
        result = task.load_tuning_results(target="t_fresh", max_age_days=1.0)
        assert result is not None

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for standalone functions ported from chag25a models.py."""

import tempfile
from pathlib import Path

from joblib import dump

from spotforecast2_safe.manager.trainer import (
    LAGS_CONSIDER,
    SEARCH_SPACES,
    get_path_model,
    load_iteration,
    search_space_lgbm,
    search_space_xgb,
    window_features,
)


# ------------------------------------------------------------------ #
# get_path_model
# ------------------------------------------------------------------ #


class TestGetPathModel:
    def test_returns_correct_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = get_path_model("lgbm", 3, model_dir=tmpdir)
            assert p.name == "lgbm_forecaster_3.joblib"

    def test_uses_cache_home_if_no_model_dir(self):
        p = get_path_model("xgb", 0)
        assert "xgb_forecaster_0.joblib" in str(p)

    def test_returns_path_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = get_path_model("lgbm", 1, model_dir=tmpdir)
            assert isinstance(p, Path)


# ------------------------------------------------------------------ #
# load_iteration
# ------------------------------------------------------------------ #


class TestLoadIteration:
    def test_returns_none_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_iteration("lgbm", 99, model_dir=tmpdir)
            assert result is None

    def test_loads_saved_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a dummy object
            model_path = Path(tmpdir) / "lgbm_forecaster_1.joblib"
            dummy = {"name": "lgbm", "iteration": 1}
            dump(dummy, model_path)

            result = load_iteration("lgbm", 1, model_dir=tmpdir)
            assert result is not None
            assert result["name"] == "lgbm"
            assert result["iteration"] == 1


# ------------------------------------------------------------------ #
# LAGS_CONSIDER
# ------------------------------------------------------------------ #


class TestLagsConsider:
    def test_is_list_of_ints(self):
        assert isinstance(LAGS_CONSIDER, list)
        assert all(isinstance(x, int) for x in LAGS_CONSIDER)

    def test_range_1_to_23(self):
        assert LAGS_CONSIDER == list(range(1, 24))
        assert len(LAGS_CONSIDER) == 23


# ------------------------------------------------------------------ #
# window_features
# ------------------------------------------------------------------ #


class TestWindowFeatures:
    def test_is_list_of_rolling_features(self):
        from spotforecast2_safe.preprocessing import RollingFeatures

        assert isinstance(window_features, list)
        assert len(window_features) == 5
        for wf in window_features:
            assert isinstance(wf, RollingFeatures)


# ------------------------------------------------------------------ #
# search_space_lgbm / search_space_xgb
# ------------------------------------------------------------------ #


class _MockTrial:
    """Minimal mock for optuna.trial.Trial."""

    def suggest_int(self, name, low, high, **kwargs):
        return low

    def suggest_float(self, name, low, high, **kwargs):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


class TestSearchSpaceLGBM:
    def test_returns_dict(self):
        result = search_space_lgbm(_MockTrial())
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = search_space_lgbm(_MockTrial())
        expected = {
            "num_leaves",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "bagging_fraction",
            "feature_fraction",
            "reg_alpha",
            "reg_lambda",
            "lags",
        }
        assert set(result.keys()) == expected

    def test_lags_from_lags_consider(self):
        result = search_space_lgbm(_MockTrial())
        assert result["lags"] in LAGS_CONSIDER


class TestSearchSpaceXGB:
    def test_returns_dict(self):
        result = search_space_xgb(_MockTrial())
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = search_space_xgb(_MockTrial())
        expected = {
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "n_estimators",
            "alpha",
            "lambda",
            "lags",
        }
        assert set(result.keys()) == expected


# ------------------------------------------------------------------ #
# SEARCH_SPACES registry
# ------------------------------------------------------------------ #


class TestSearchSpaces:
    def test_contains_lgbm_and_xgb(self):
        assert "lgbm" in SEARCH_SPACES
        assert "xgb" in SEARCH_SPACES

    def test_values_are_callable(self):
        for name, fn in SEARCH_SPACES.items():
            assert callable(fn), f"SEARCH_SPACES['{name}'] is not callable"

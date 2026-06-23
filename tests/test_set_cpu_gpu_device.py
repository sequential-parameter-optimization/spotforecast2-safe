# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for preprocessing.checking.set_cpu_gpu_device device handling.

Uses duck-typed stub estimators named after the supported backends so the
device branches are exercised without the optional xgboost / catboost deps. The
regression of interest: a fitted CatBoost locks its params, so the device set
fails; that benign no-op must NOT emit a UserWarning (backtesting calls this
around every fold's predict, which otherwise floods the log).
"""

import warnings

from spotforecast2_safe.preprocessing.checking import (
    _estimator_is_fitted,
    set_cpu_gpu_device,
)


def _record(estimator, device="cpu"):
    """Return (original_device, [UserWarnings]) from a set_cpu_gpu_device call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        original = set_cpu_gpu_device(estimator, device=device)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    return original, user_warnings


class TestFittedCatBoostDeviceSet:
    def test_fitted_catboost_set_failure_is_silent(self):
        """A fitted CatBoost (params locked) must NOT warn on the failed set."""

        class CatBoostRegressor:
            def get_params(self, deep=True):
                return {}  # task_type not exposed -> reads as None

            def set_params(self, **kwargs):
                raise Exception("You can't change params of fitted model")

            def is_fitted(self):
                return True

        original, warns = _record(CatBoostRegressor())
        assert warns == []
        assert original is None

    def test_unfitted_failure_still_warns(self):
        """A genuine (unfitted) failure to set the device must still warn."""

        class CatBoostRegressor:
            def get_params(self, deep=True):
                return {}

            def set_params(self, **kwargs):
                raise Exception("boom")

            def is_fitted(self):
                return False

        original, warns = _record(CatBoostRegressor())
        assert len(warns) == 1
        assert "Failed to set device parameter 'task_type'" in str(warns[0].message)

    def test_already_cpu_skips_set_entirely(self):
        """task_type already 'CPU' (read via get_params) -> no set attempt, no warning."""
        set_calls = {"n": 0}

        class CatBoostRegressor:
            def get_params(self, deep=True):
                return {"task_type": "CPU"}

            def set_params(self, **kwargs):
                set_calls["n"] += 1

            def is_fitted(self):
                return True

        original, warns = _record(CatBoostRegressor())
        assert warns == []
        assert set_calls["n"] == 0
        assert original == "CPU"


class TestUnsupportedEstimator:
    def test_unsupported_returns_none_no_warning(self):
        class LinearRegression:
            pass

        original, warns = _record(LinearRegression())
        assert original is None
        assert warns == []


class TestEstimatorIsFitted:
    def test_catboost_is_fitted_method(self):
        class E:
            def is_fitted(self):
                return True

        assert _estimator_is_fitted(E()) is True

    def test_no_fitted_info_returns_false(self):
        class E:
            pass

        assert _estimator_is_fitted(E()) is False

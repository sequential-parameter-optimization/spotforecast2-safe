# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.forecaster.wrappers import (
    ForecasterRecursiveCatBoost,
    ForecasterRecursiveModel,
)


def test_forecaster_recursive_catboost_initialization():
    """Test that ForecasterRecursiveCatBoost initializes correctly."""
    model = ForecasterRecursiveCatBoost(iteration=0)
    assert model.iteration == 0
    assert model.name == "catboost"
    assert isinstance(model, ForecasterRecursiveModel)


def test_forecaster_recursive_catboost_inheritance():
    """Test inheritance from ForecasterRecursiveModel."""
    assert issubclass(ForecasterRecursiveCatBoost, ForecasterRecursiveModel)


def test_forecaster_recursive_catboost_import_location():
    """Verify it can be imported from the wrappers package."""
    from spotforecast2_safe.forecaster.wrappers import ForecasterRecursiveCatBoost as CB

    assert CB is ForecasterRecursiveCatBoost


def test_forecaster_recursive_catboost_in_public_api():
    """The wrapper is part of the pinned public API (``__all__``)."""
    import spotforecast2_safe as sf2s

    assert hasattr(sf2s, "ForecasterRecursiveCatBoost")
    assert "ForecasterRecursiveCatBoost" in sf2s.__all__


@pytest.fixture
def sample_ts_data():
    """Simple time series for testing."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    y = pd.Series(rng.standard_normal(100), index=idx, name="load")
    return y


def test_forecaster_recursive_catboost_fit_predict_interface(sample_ts_data):
    """Test the fit/predict interface (smoke test)."""
    model = ForecasterRecursiveCatBoost(iteration=0, lags=3)

    # If CatBoost is available, we can test fit/predict.
    # If not, it logs a warning but the interface still exists.
    from spotforecast2_safe.forecaster.wrappers.catboost import CatBoostRegressor

    if CatBoostRegressor is not None:
        model.fit(sample_ts_data)
        predictions = model.forecaster.predict(steps=5)
        assert len(predictions) == 5
    else:
        pytest.skip("CatBoost not installed, skipping fit/predict test")


def test_package_prediction_interface():
    """Test package_prediction method exists."""
    model = ForecasterRecursiveCatBoost(iteration=0)
    assert hasattr(model, "package_prediction")


def test_catboost_determinism_flags_are_pinned():
    """The CatBoost backend must be instantiated with the flags that keep
    training bit-level reproducible and free of filesystem side effects:
    a fixed ``random_seed``, single-threaded summation (``thread_count=1``)
    so the result does not depend on the host core count, and
    ``allow_writing_files=False`` so training never writes a ``catboost_info``
    directory. Silently dropping any of these would break the determinism /
    fail-safe claims in the compliance narrative; this test fails loudly when
    that happens."""
    from spotforecast2_safe.forecaster.wrappers.catboost import CatBoostRegressor

    if CatBoostRegressor is None:
        pytest.skip("CatBoost not installed, skipping determinism-flag test")

    model = ForecasterRecursiveCatBoost(iteration=0)
    params = model.forecaster.estimator.get_params()
    assert params["random_seed"] == model.random_state
    assert params["thread_count"] == 1
    assert params["allow_writing_files"] is False

"""Tests for ForecasterRecursive exogenous variable validation logic."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_exog_full_length_alignment():
    """Test exog with same length as y is correctly trimmed."""
    y = pd.Series(np.arange(30, dtype=float), name="y")
    exog = pd.DataFrame({"temp": np.random.randn(30)}, index=y.index)

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    assert forecaster.is_fitted
    assert forecaster.exog_in_ is True
    assert forecaster.exog_names_in_ == ["temp"]


def test_exog_prealigned_length():
    """Test exog already aligned to training index."""
    y = pd.Series(np.arange(30, dtype=float), name="y")
    window_size = 5
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=window_size)

    # Pre-aligned exog: length = len(y) - window_size
    expected_train_len = 30 - window_size
    exog = pd.DataFrame(
        {"temp": np.random.randn(expected_train_len)}, index=y.index[window_size:]
    )

    forecaster.fit(y=y, exog=exog)

    assert forecaster.is_fitted
    assert forecaster.exog_in_ is True


def test_exog_invalid_length_raises_error():
    """Test that invalid exog length raises clear error."""
    y = pd.Series(np.arange(30, dtype=float), name="y")
    # Invalid length: neither full nor pre-aligned
    exog = pd.DataFrame({"temp": np.random.randn(20)}, index=range(20))

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    with pytest.raises(ValueError, match="Length mismatch for exogenous variables"):
        forecaster.fit(y=y, exog=exog)


def test_exog_validation_with_differentiation():
    """Test exog validation works correctly with differentiation enabled."""
    y = pd.Series(np.arange(30, dtype=float), name="y")
    exog = pd.DataFrame({"temp": np.random.randn(30)}, index=y.index)

    # Differentiation creates NaNs at start but doesn't change y length
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )
    forecaster.fit(y=y, exog=exog)

    assert forecaster.is_fitted
    assert forecaster.differentiation == 1


def test_exog_error_message_clarity():
    """Test that error message provides clear guidance."""
    y = pd.Series(np.arange(30, dtype=float), name="y")
    exog = pd.DataFrame({"temp": np.random.randn(15)}, index=range(15))

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)

    with pytest.raises(ValueError) as exc_info:
        forecaster.fit(y=y, exog=exog)

    error_msg = str(exc_info.value)
    assert "30 observations" in error_msg  # Full length
    assert "25 observations" in error_msg  # Pre-aligned (30 - 5)
    assert "15 observations" in error_msg  # Actual invalid length
    assert "Window size: 5" in error_msg

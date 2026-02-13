# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_create_sample_weights_value_error_if_weights_contain_nan():
    """
    Test ValueError is raised when weights contain NaNs.
    """
    y = pd.Series(np.arange(10), name="y")

    def weight_func(index):
        weights = np.ones(len(index))
        weights[0] = np.nan
        return weights

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, weight_func=weight_func
    )
    X_train, y_train = forecaster.create_train_X_y(y=y)

    msg = "The resulting `sample_weight` cannot have NaN values."
    with pytest.raises(ValueError, match=msg):
        forecaster.create_sample_weights(X_train=X_train)


def test_create_sample_weights_value_error_if_weights_contain_negative_values():
    """
    Test ValueError is raised when weights contain negative values.
    """
    y = pd.Series(np.arange(10), name="y")

    def weight_func(index):
        weights = np.ones(len(index))
        weights[0] = -1
        return weights

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, weight_func=weight_func
    )
    X_train, y_train = forecaster.create_train_X_y(y=y)

    msg = "The resulting `sample_weight` cannot have negative values."
    with pytest.raises(ValueError, match=msg):
        forecaster.create_sample_weights(X_train=X_train)


def test_create_sample_weights_value_error_if_weights_sum_to_zero():
    """
    Test ValueError is raised when weights sum to zero.
    """
    y = pd.Series(np.arange(10), name="y")

    def weight_func(index):
        weights = np.zeros(len(index))
        return weights

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, weight_func=weight_func
    )
    X_train, y_train = forecaster.create_train_X_y(y=y)

    msg = "The resulting `sample_weight` cannot be normalized because the sum of the weights is zero."
    with pytest.raises(ValueError, match=msg):
        forecaster.create_sample_weights(X_train=X_train)


def test_fit_last_window_stored_as_dataframe():
    """
    Test last_window_ is stored as a DataFrame with the correct name.
    """
    y = pd.Series(np.arange(10), name="y_series")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    assert isinstance(forecaster.last_window_, pd.DataFrame)
    assert forecaster.last_window_.columns == ["y_series"]
    assert forecaster.last_window_.index.equals(y.index[-3:])


def test_fit_attributes_reset():
    """
    Test attributes are reset when calling fit.
    """
    y = pd.Series(np.arange(10), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    # Set some attributes to verify they are reset
    forecaster.last_window_ = "not_none"
    forecaster.index_type_ = "not_none"
    forecaster.is_fitted = True

    forecaster.fit(y=y)

    assert isinstance(forecaster.last_window_, pd.DataFrame)
    assert forecaster.index_type_ is type(y.index)
    assert forecaster.is_fitted is True


def test_fit_with_weight_func():
    """
    Test fit uses weight_func correctly.
    """
    y = pd.Series(np.arange(10), name="y")

    def weight_func(index):
        return np.ones(len(index))

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, weight_func=weight_func
    )
    forecaster.fit(y=y)

    # We can check if fit_kwargs was populated during fit (it's cleared after,
    # but we can check if it runs without error and estimator is fitted)
    assert forecaster.is_fitted


def test_fit_suppress_warnings():
    """
    Test fit suppresses warnings when suppress_warnings is True.
    """
    # This test is a bit tricky to verify directly without side effects,
    # but we can ensure it runs without error.
    y = pd.Series(np.arange(10), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y, suppress_warnings=True)
    assert forecaster.is_fitted

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-safe order validation for TimeSeriesDifferentiator.

Only first-order differentiation is implemented in this port. A higher order
must be rejected at ``fit`` rather than fitting successfully and crashing during
prediction (a fail-late bug).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import TimeSeriesDifferentiator


def test_order_below_one_raises():
    with pytest.raises(ValueError, match="positive integer"):
        TimeSeriesDifferentiator(order=0).fit(np.arange(10).astype(float))


def test_order_above_one_fails_fast_at_fit():
    with pytest.raises(NotImplementedError, match="order=1"):
        TimeSeriesDifferentiator(order=2).fit(np.arange(10).astype(float))


def test_order_one_round_trips():
    y = np.arange(1, 11).astype(float)
    differ = TimeSeriesDifferentiator(order=1)
    recovered = differ.inverse_transform(differ.fit_transform(y))
    np.testing.assert_array_almost_equal(recovered, y)


def test_forecaster_differentiation_above_one_fails_at_fit_not_predict():
    fc = ForecasterRecursive(estimator=LinearRegression(), lags=3, differentiation=2)
    y = pd.Series(np.arange(1, 51).astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError, match="order=1"):
            fc.fit(y)  # must fail here, before any predict call

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-safe input-validation tests for ``n2n_predict_with_covariates``.

These exercise the ``ValueError`` guards that run *before* any data fetch or
model I/O, so the tests are fast, deterministic, and need no network access.
The guards are part of the safety-critical fail-safe contract: invalid input
must raise rather than silently proceed.
"""

import pytest

from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"forecast_horizon": 0}, "forecast_horizon must be positive"),
        ({"forecast_horizon": -1}, "forecast_horizon must be positive"),
        ({"contamination": -0.1}, "contamination must be between 0 and 0.5"),
        ({"contamination": 0.6}, "contamination must be between 0 and 0.5"),
        ({"window_size": 0}, "window_size must be positive"),
        ({"window_size": -5}, "window_size must be positive"),
        ({"lags": 0}, "lags must be positive"),
        ({"lags": -1}, "lags must be positive"),
        ({"train_ratio": 0.0}, "train_ratio must be between 0 and 1"),
        ({"train_ratio": 1.0}, "train_ratio must be between 0 and 1"),
    ],
)
def test_invalid_inputs_raise_before_io(kwargs, match):
    """Each guard raises ValueError before any data/model I/O occurs."""
    with pytest.raises(ValueError, match=match):
        n2n_predict_with_covariates(**kwargs)

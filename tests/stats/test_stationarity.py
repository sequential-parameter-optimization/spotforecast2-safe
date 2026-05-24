# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Augmented Dickey-Fuller stationarity test."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.stats.stationarity import augmented_dickey_fuller


def test_augmented_dickey_fuller_on_white_noise_rejects():
    """White noise is stationary — ADF should reject the unit-root H0."""
    rng = np.random.default_rng(42)
    y = pd.Series(rng.standard_normal(1000))
    result = augmented_dickey_fuller(y)
    assert result["p_value"] < 0.05


def test_augmented_dickey_fuller_on_random_walk_does_not_reject():
    """A random walk has a unit root — ADF should NOT reject H0."""
    rng = np.random.default_rng(42)
    y = pd.Series(np.cumsum(rng.standard_normal(1000)))
    result = augmented_dickey_fuller(y)
    assert result["p_value"] > 0.05


def test_augmented_dickey_fuller_returns_expected_keys():
    """Result Series carries the documented index."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.standard_normal(200))
    result = augmented_dickey_fuller(y)
    expected = {
        "statistic",
        "p_value",
        "n_lags",
        "n_obs",
        "critical_1%",
        "critical_5%",
        "critical_10%",
    }
    assert set(result.index) == expected


def test_augmented_dickey_fuller_rejects_invalid_y():
    """Non-Series input raises TypeError via check_y."""
    with pytest.raises(TypeError):
        augmented_dickey_fuller([1.0, 2.0, 3.0])


def test_augmented_dickey_fuller_rejects_missing_values():
    """NaN in the series raises ValueError via check_y."""
    y = pd.Series([1.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        augmented_dickey_fuller(y)

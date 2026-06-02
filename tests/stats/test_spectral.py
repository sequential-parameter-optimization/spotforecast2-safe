# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the periodogram primitive."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.stats.spectral import PeriodogramResult, compute_periodogram


def test_compute_periodogram_returns_periodogram_result():
    """Return type is the documented dataclass."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.standard_normal(128))
    result = compute_periodogram(y)
    assert isinstance(result, PeriodogramResult)


def test_compute_periodogram_returns_dataframe_with_power_column():
    """Spectrum has the right shape and a single ``power`` column."""
    rng = np.random.default_rng(0)
    n = 256
    y = pd.Series(rng.standard_normal(n))
    result = compute_periodogram(y)
    assert list(result.spectrum.columns) == ["power"]
    # scipy.signal.periodogram returns n//2 + 1 frequency bins
    assert result.spectrum.shape == (n // 2 + 1, 1)


def test_compute_periodogram_top_peaks_match_known_frequency():
    """Synthetic sine with period 32 yields a top peak at period ≈ 32."""
    n = 1024
    period = 32
    t = np.arange(n)
    y = pd.Series(np.sin(2 * np.pi * t / period))
    result = compute_periodogram(y, max_peaks=1)
    detected_period = float(result.top_periods.index[0])
    assert abs(detected_period - period) <= 1.0


def test_compute_periodogram_respects_max_peaks():
    """``max_peaks`` caps the size of ``top_periods``."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.standard_normal(256))
    result = compute_periodogram(y, max_peaks=3)
    assert len(result.top_periods) == 3


def test_compute_periodogram_rejects_invalid_y():
    """Non-Series input raises TypeError via check_y."""
    with pytest.raises(TypeError):
        compute_periodogram([1.0, 2.0, 3.0])


def test_compute_periodogram_rejects_missing_values():
    """NaN in the series raises ValueError via check_y."""
    y = pd.Series([1.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        compute_periodogram(y)


def test_compute_periodogram_empty_spectrum():
    """A single-sample series has only the DC component -> empty top_periods."""
    result = compute_periodogram(pd.Series([5.0]))
    assert isinstance(result, PeriodogramResult)
    assert result.top_periods.empty

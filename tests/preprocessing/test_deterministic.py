# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Fourier deterministic-process primitive."""

import pandas as pd

from spotforecast2_safe.preprocessing.deterministic import (
    build_deterministic_process,
)


def test_build_deterministic_process_returns_DP_with_expected_columns():
    """Default config emits constant + linear-trend + 2*order Fourier columns."""
    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    dp = build_deterministic_process(idx, periods=[24], fourier_order=3)
    features = dp.in_sample()
    # 1 (constant) + 1 (linear trend) + 2 * 3 (sin/cos pairs) = 8
    assert features.shape == (48, 8)


def test_build_deterministic_process_respects_fourier_order():
    """Doubling fourier_order doubles the number of Fourier columns."""
    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    dp_low = build_deterministic_process(idx, periods=[24], fourier_order=2)
    dp_high = build_deterministic_process(idx, periods=[24], fourier_order=4)
    cols_low = dp_low.in_sample().shape[1]
    cols_high = dp_high.in_sample().shape[1]
    assert cols_high - cols_low == 2 * (4 - 2)


def test_build_deterministic_process_multiple_periods():
    """Multiple periods add disjoint Fourier bases."""
    idx = pd.date_range("2026-01-01", periods=168, freq="h")
    dp = build_deterministic_process(
        idx, periods=[24, 168], fourier_order=2
    )
    features = dp.in_sample()
    # 1 + 1 + 2*2 + 2*2 = 10
    assert features.shape == (168, 10)


def test_build_deterministic_process_trend_order_zero():
    """trend_order=0 drops the linear trend column."""
    idx = pd.date_range("2026-01-01", periods=24, freq="h")
    dp = build_deterministic_process(
        idx, periods=[24], fourier_order=1, trend_order=0
    )
    features = dp.in_sample()
    # 1 (constant) + 0 (trend) + 2 (one pair) = 3
    assert features.shape == (24, 3)


def test_build_deterministic_process_no_constant():
    """constant=False drops the bias column."""
    idx = pd.date_range("2026-01-01", periods=24, freq="h")
    dp = build_deterministic_process(
        idx, periods=[24], fourier_order=1, constant=False
    )
    features = dp.in_sample()
    # 0 (constant) + 1 (trend) + 2 = 3
    assert features.shape == (24, 3)

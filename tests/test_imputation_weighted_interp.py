# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the new ``"weighted_interp"`` imputation method.

Covers: linear interpolation semantics, zero-weight generation on the
pre-fill NaN mask, boundary-NaN fallback, and the existing unknown-method
ValueError guard now extended to accept ``"weighted_interp"``.
"""

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.imputation import WeightFunction, apply_imputation


def _cfg(method: str, window_size: int = 3, imputation_window_size=None):
    return SimpleNamespace(
        imputation_method=method,
        targets=["A"],
        window_size=window_size,
        imputation_window_size=imputation_window_size,
    )


def _log():
    return logging.getLogger("test_wi")


# ---------------------------------------------------------------------------
# Interpolation semantics
# ---------------------------------------------------------------------------


class TestWeightedInterpInterpolation:
    def test_interior_gap_is_linearly_interpolated(self):
        """Interior NaN must be bridged by linear time-interpolation."""
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({"A": [10.0, None, 30.0, 40.0, 50.0]}, index=idx)

        result, wf = apply_imputation(df.copy(), _cfg("weighted_interp"), _log())

        # Slot 1 must be linearly interpolated between 10 and 30 -> 20.0
        assert (
            abs(result["A"].iloc[1] - 20.0) < 1e-6
        ), f"Expected 20.0 but got {result['A'].iloc[1]}"

    def test_multi_slot_interior_gap_interpolated(self):
        """Multiple consecutive interior NaNs must be linearly bridged."""
        idx = pd.date_range("2024-01-01", periods=6, freq="h")
        df = pd.DataFrame({"A": [0.0, None, None, None, 40.0, 50.0]}, index=idx)

        result, wf = apply_imputation(df.copy(), _cfg("weighted_interp"), _log())

        # Linear interpolation: 0, 10, 20, 30, 40, 50
        expected = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        assert result["A"].tolist() == pytest.approx(expected, abs=1e-5)

    def test_boundary_nan_fallback(self):
        """Trailing NaN (cannot be bracketed) must be filled by ffill fallback."""
        idx = pd.date_range("2024-01-01", periods=4, freq="h")
        df = pd.DataFrame({"A": [10.0, 20.0, 30.0, None]}, index=idx)

        result, wf = apply_imputation(df.copy(), _cfg("weighted_interp"), _log())

        # Trailing NaN falls back to ffill: should be 30.0
        assert abs(result["A"].iloc[-1] - 30.0) < 1e-6

    def test_no_nans_remain(self):
        """After weighted_interp no NaNs must remain for a standard gap."""
        idx = pd.date_range("2024-01-01", periods=8, freq="h")
        df = pd.DataFrame({"A": [1.0, None, None, 4.0, 5.0, None, 7.0, 8.0]}, index=idx)
        result, wf = apply_imputation(df.copy(), _cfg("weighted_interp"), _log())
        assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------


class TestWeightedInterpWeights:
    def test_flagged_slots_receive_zero_weight(self):
        """Slots in the pre-fill NaN mask must have weight 0 in the penalty zone."""
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {"A": [1.0, 2.0, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=idx
        )

        result, wf = apply_imputation(
            df.copy(), _cfg("weighted_interp", window_size=2), _log()
        )

        assert isinstance(wf, WeightFunction)
        # The gap slots (idx[2], idx[3]) must have weight 0
        assert wf(pd.Index([idx[2]])) is None or wf.weights_series.loc[idx[2]] == 0.0
        assert wf(pd.Index([idx[3]])) is None or wf.weights_series.loc[idx[3]] == 0.0

    def test_clean_slots_before_gap_have_positive_weight(self):
        """Slots well before the gap must retain weight 1."""
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {"A": [1.0, 2.0, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=idx
        )
        result, wf = apply_imputation(
            df.copy(), _cfg("weighted_interp", window_size=1), _log()
        )

        assert isinstance(wf, WeightFunction)
        # idx[0] is 2 steps before the gap with window=1: should be weight 1
        w = wf.weights_series.loc[idx[0]]
        assert w == 1.0, f"Expected weight 1.0 at idx[0], got {w}"

    def test_returns_none_weight_func_when_all_zero(self):
        """When the entire series is NaN (all gap), weight_func must be None."""
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({"A": [float("nan")] * 5}, index=idx)
        result, wf = apply_imputation(
            df.copy(), _cfg("weighted_interp", window_size=1), _log()
        )
        assert wf is None


# ---------------------------------------------------------------------------
# Unknown method still raises
# ---------------------------------------------------------------------------


def test_unknown_method_raises():
    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    cfg = SimpleNamespace(
        imputation_method="bogus",
        targets=["A"],
        window_size=2,
        imputation_window_size=None,
    )
    with pytest.raises(ValueError, match="Unknown imputation_method"):
        apply_imputation(df.copy(), cfg, _log())

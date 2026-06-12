# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for processing.shape_check forecast shape plausibility check.

Covers the pure computation contract of check_forecast_shape and the
ShapeCheckReport properties.  No logging or raising on implausible results
is expected from the library functions.
"""

import math

import pandas as pd
import pytest

from spotforecast2_safe.processing.shape_check import (
    ShapeCheckReport,
    check_forecast_shape,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IDX = pd.date_range("2026-06-11 00:00", periods=24, freq="h", tz="UTC")
PROFILE = pd.Series([float(i % 12) for i in range(24)], index=IDX, name="reference")


# ---------------------------------------------------------------------------
# ShapeCheckReport properties
# ---------------------------------------------------------------------------


class TestShapeCheckReport:
    def test_plausible_when_both_metrics_exceed_thresholds(self):
        r = ShapeCheckReport(
            n_overlap=24, corr=0.85, range_ratio=0.9, min_corr=0.6, min_range_ratio=0.5
        )
        assert r.plausible

    def test_not_plausible_when_corr_below_threshold(self):
        r = ShapeCheckReport(
            n_overlap=24, corr=0.3, range_ratio=0.9, min_corr=0.6, min_range_ratio=0.5
        )
        assert not r.plausible

    def test_not_plausible_when_range_ratio_below_threshold(self):
        r = ShapeCheckReport(
            n_overlap=24, corr=0.9, range_ratio=0.2, min_corr=0.6, min_range_ratio=0.5
        )
        assert not r.plausible

    def test_not_plausible_on_nan_corr(self):
        r = ShapeCheckReport(
            n_overlap=24,
            corr=float("nan"),
            range_ratio=0.9,
            min_corr=0.6,
            min_range_ratio=0.5,
        )
        assert not r.plausible

    def test_not_plausible_on_nan_range_ratio(self):
        r = ShapeCheckReport(
            n_overlap=24,
            corr=0.9,
            range_ratio=float("nan"),
            min_corr=0.6,
            min_range_ratio=0.5,
        )
        assert not r.plausible

    def test_plausible_at_exact_boundary(self):
        """Exactly at threshold -> plausible (>= not >)."""
        r = ShapeCheckReport(
            n_overlap=24, corr=0.6, range_ratio=0.5, min_corr=0.6, min_range_ratio=0.5
        )
        assert r.plausible

    def test_skipped_when_n_overlap_zero_and_nan_metrics(self):
        r = ShapeCheckReport(
            n_overlap=0,
            corr=float("nan"),
            range_ratio=float("nan"),
            min_corr=0.6,
            min_range_ratio=0.5,
        )
        assert r.skipped

    def test_not_skipped_when_metrics_computed(self):
        r = ShapeCheckReport(
            n_overlap=24, corr=0.8, range_ratio=0.9, min_corr=0.6, min_range_ratio=0.5
        )
        assert not r.skipped

    def test_frozen_dataclass(self):
        r = ShapeCheckReport(
            n_overlap=24, corr=0.8, range_ratio=0.9, min_corr=0.6, min_range_ratio=0.5
        )
        with pytest.raises((AttributeError, TypeError)):
            r.corr = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# check_forecast_shape
# ---------------------------------------------------------------------------


class TestCheckForecastShape:
    def test_identical_profile_corr_one(self):
        """Identical forecast and reference -> corr == 1.0, plausible."""
        report = check_forecast_shape(PROFILE, PROFILE)
        assert math.isclose(report.corr, 1.0, abs_tol=1e-9)
        assert math.isclose(report.range_ratio, 1.0, abs_tol=1e-9)
        assert report.plausible

    def test_inverted_profile_corr_negative(self):
        """Inverted forecast -> corr < 0, not plausible."""
        inverted = PROFILE.max() - PROFILE
        report = check_forecast_shape(inverted, PROFILE)
        assert report.corr < 0
        assert not report.plausible

    def test_flat_forecast_range_ratio_below_threshold(self):
        """Constant forecast -> range == 0, range_ratio == 0 < 0.5, not plausible."""
        flat = pd.Series(5.0, index=IDX)
        report = check_forecast_shape(flat, PROFILE)
        assert report.range_ratio < 0.5
        assert not report.plausible

    def test_zero_range_reference_produces_nan_ratio(self):
        """Constant reference -> range == 0, range_ratio is NaN."""
        flat_ref = pd.Series(50_000.0, index=IDX)
        report = check_forecast_shape(PROFILE, flat_ref)
        assert math.isnan(report.range_ratio)
        assert not report.plausible

    def test_short_overlap_returns_skipped(self):
        """Fewer than min_overlap points -> skipped (n_overlap == 0)."""
        short = PROFILE.iloc[:5]
        report = check_forecast_shape(short, PROFILE, min_overlap=12)
        assert report.skipped
        assert math.isnan(report.corr)
        assert math.isnan(report.range_ratio)

    def test_overlap_exactly_at_min_is_not_skipped(self):
        """Exactly min_overlap points -> not skipped."""
        exactly = PROFILE.iloc[:12]
        report = check_forecast_shape(exactly, PROFILE, min_overlap=12)
        assert not report.skipped
        assert report.n_overlap == 12

    def test_n_overlap_reflects_intersection(self):
        """Partial index overlap -> n_overlap is the intersection size."""
        half = PROFILE.iloc[12:]  # 12 points overlap with PROFILE
        report = check_forecast_shape(half, PROFILE, min_overlap=1)
        assert report.n_overlap == 12

    def test_deterministic(self):
        """Two calls with same input return identical results."""
        r1 = check_forecast_shape(PROFILE, PROFILE)
        r2 = check_forecast_shape(PROFILE, PROFILE)
        assert r1 == r2

    def test_raises_on_non_series_y(self):
        with pytest.raises(TypeError, match="pd.Series"):
            check_forecast_shape([1, 2, 3], PROFILE)  # type: ignore[arg-type]

    def test_raises_on_non_series_reference(self):
        with pytest.raises(TypeError, match="pd.Series"):
            check_forecast_shape(PROFILE, [1, 2, 3])  # type: ignore[arg-type]

    def test_raises_on_empty_y(self):
        with pytest.raises(ValueError, match="empty"):
            check_forecast_shape(pd.Series(dtype=float), PROFILE)

    def test_raises_on_empty_reference(self):
        with pytest.raises(ValueError, match="empty"):
            check_forecast_shape(PROFILE, pd.Series(dtype=float))

    def test_thresholds_passed_through_to_report(self):
        """min_corr / min_range_ratio must be stored in the report."""
        report = check_forecast_shape(
            PROFILE, PROFILE, min_corr=0.7, min_range_ratio=0.4
        )
        assert report.min_corr == 0.7
        assert report.min_range_ratio == 0.4

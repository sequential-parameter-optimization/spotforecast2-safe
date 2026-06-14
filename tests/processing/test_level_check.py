# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for processing.shape_check level / debias functions.

check_forecast_level measures a systematic flat offset; apply_level_correction
removes it.  Both are pure: no logging, no raising on a biased result.
"""

import math

import pandas as pd
import pytest

from spotforecast2_safe.processing.shape_check import (
    LevelCheckReport,
    apply_level_correction,
    check_forecast_level,
)

IDX = pd.date_range("2026-06-13 00:00", periods=24, freq="h", tz="UTC")
# A non-flat reference so median != endpoints and the range is non-zero.
ACTUAL = pd.Series([43_000.0 + 250.0 * (i % 12) for i in range(24)], index=IDX)


class TestLevelCheckReport:
    def test_biased_when_rel_offset_exceeds_tol(self):
        r = LevelCheckReport(
            n_overlap=24,
            statistic="median",
            forecast_level=45_600.0,
            reference_level=43_858.0,
            offset=1_742.0,
            rel_offset=0.04,
            tol=0.02,
        )
        assert r.biased and not r.skipped

    def test_not_biased_at_or_below_tol(self):
        r = LevelCheckReport(
            n_overlap=24,
            statistic="median",
            forecast_level=1.0,
            reference_level=100.0,
            offset=2.0,
            rel_offset=0.02,
            tol=0.02,
        )
        assert not r.biased  # strictly greater-than required

    def test_nan_rel_offset_not_biased(self):
        r = LevelCheckReport(
            n_overlap=0,
            statistic="median",
            forecast_level=float("nan"),
            reference_level=float("nan"),
            offset=float("nan"),
            rel_offset=float("nan"),
            tol=0.02,
        )
        assert not r.biased
        assert r.skipped

    def test_frozen(self):
        r = LevelCheckReport(
            n_overlap=24,
            statistic="mean",
            forecast_level=1.0,
            reference_level=1.0,
            offset=0.0,
            rel_offset=0.0,
            tol=0.02,
        )
        with pytest.raises((AttributeError, TypeError)):
            r.offset = 9.0  # type: ignore[misc]


class TestCheckForecastLevel:
    def test_flat_high_forecast_is_biased(self):
        rep = check_forecast_level(ACTUAL + 1_800.0, ACTUAL, tol=0.02)
        assert rep.biased
        assert rep.offset == pytest.approx(1_800.0, abs=1e-6)
        assert rep.rel_offset > 0
        assert rep.n_overlap == 24

    def test_flat_low_forecast_negative_offset(self):
        rep = check_forecast_level(ACTUAL - 1_800.0, ACTUAL)
        assert rep.offset < 0 and rep.rel_offset < 0
        assert rep.biased

    def test_well_centred_not_biased(self):
        rep = check_forecast_level(ACTUAL + 10.0, ACTUAL, tol=0.02)
        assert not rep.biased

    def test_mean_statistic(self):
        rep = check_forecast_level(ACTUAL + 500.0, ACTUAL, statistic="mean")
        assert rep.statistic == "mean"
        assert rep.offset == pytest.approx(500.0, abs=1e-6)

    def test_zero_reference_level_nan_rel(self):
        zero_ref = pd.Series(0.0, index=IDX)
        rep = check_forecast_level(ACTUAL, zero_ref)
        assert math.isnan(rep.rel_offset)
        assert not rep.biased  # NaN rel offset -> not biased

    def test_short_overlap_skipped(self):
        rep = check_forecast_level(ACTUAL.iloc[:5], ACTUAL, min_overlap=12)
        assert rep.skipped
        assert math.isnan(rep.offset)

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError, match="median"):
            check_forecast_level(ACTUAL, ACTUAL, statistic="mode")

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            check_forecast_level([1, 2, 3], ACTUAL)  # type: ignore[arg-type]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            check_forecast_level(pd.Series(dtype=float), ACTUAL)

    def test_deterministic(self):
        a = check_forecast_level(ACTUAL + 5.0, ACTUAL)
        b = check_forecast_level(ACTUAL + 5.0, ACTUAL)
        assert a == b


class TestApplyLevelCorrection:
    def test_removes_flat_offset(self):
        biased = ACTUAL + 1_800.0
        corrected = apply_level_correction(biased, ACTUAL)
        assert check_forecast_level(corrected, ACTUAL).offset == pytest.approx(
            0.0, abs=1e-6
        )

    def test_preserves_index_name_and_shape(self):
        biased = (ACTUAL + 1_800.0).rename("y0")
        corrected = apply_level_correction(biased, ACTUAL)
        assert corrected.index.equals(biased.index)
        assert corrected.name == "y0"
        assert len(corrected) == len(biased)

    def test_does_not_mutate_input(self):
        biased = ACTUAL + 1_800.0
        before = biased.copy()
        apply_level_correction(biased, ACTUAL)
        pd.testing.assert_series_equal(biased, before)

    def test_shape_is_preserved_only_level_shifts(self):
        biased = ACTUAL + 1_800.0
        corrected = apply_level_correction(biased, ACTUAL)
        # constant shift => differences between consecutive points unchanged
        pd.testing.assert_series_equal(biased.diff(), corrected.diff())

    def test_short_overlap_raises(self):
        with pytest.raises(ValueError, match="min_overlap"):
            apply_level_correction(ACTUAL.iloc[:5], ACTUAL, min_overlap=12)

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError, match="median"):
            apply_level_correction(ACTUAL, ACTUAL, statistic="mode")

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            apply_level_correction(ACTUAL, [1, 2, 3])  # type: ignore[arg-type]

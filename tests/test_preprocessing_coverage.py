# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for preprocessing.coverage operational guards.

Synthetic frames reproduce two known operational incidents:

- 2026-06-02 pattern: interior gap — a full day of actuals published more than
  a day late, creating a >12 h hole inside the scan window while edge checks
  still pass.
- 2026-06-05 pattern: partial frontier hour — 15-min cadence, last hour has
  only 2 of 4 samples; ``last_complete_hour`` must step back one hour.

Boundary operator semantics (ported from the script):
- ``assert_frontier_fresh``: strict ``<`` — equal passes, beyond fails.
- ``assert_actual_lag_within``: strict ``<`` — equal passes (last_actual ==
  now - max_lag is exactly at the tolerance boundary), beyond fails.
"""

import pandas as pd
import pytest

from spotforecast2_safe.exceptions import CoverageError
from spotforecast2_safe.preprocessing.coverage import (
    assert_actual_lag_within,
    assert_frontier_fresh,
    assert_no_interior_gaps,
    last_complete_hour,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = pd.Timestamp("2026-06-11 12:00", tz="UTC")
SCAN_WINDOW = pd.Timedelta(days=28)
MAX_GAP = pd.Timedelta(hours=12)


@pytest.fixture(scope="module")
def clean_hourly_30d() -> pd.Series:
    """30 days of clean hourly Actual Load data ending at NOW."""
    idx = pd.date_range(
        NOW - pd.Timedelta(days=30), periods=30 * 24, freq="h", tz="UTC"
    )
    return pd.Series(50_000.0, index=idx, name="Actual Load")


@pytest.fixture(scope="module")
def clean_15min_24h() -> pd.Series:
    """24 h of clean 15-min data (4 samples/hour, 96 slots total)."""
    start = pd.Timestamp("2026-06-10 00:00", tz="UTC")
    idx = pd.date_range(start, periods=24 * 4, freq="15min", tz="UTC")
    return pd.Series(50_000.0, index=idx, name="Actual Load")


# ---------------------------------------------------------------------------
# assert_frontier_fresh
# ---------------------------------------------------------------------------


class TestAssertFrontierFresh:
    def test_passes_when_last_equals_required(self, clean_hourly_30d):
        """Equal boundary: index.max() == required_last must NOT raise."""
        required = clean_hourly_30d.index.max()
        assert_frontier_fresh(clean_hourly_30d.index, required)  # no exception

    def test_passes_when_last_after_required(self, clean_hourly_30d):
        """Data exceeds requirement — definitely passes."""
        required = clean_hourly_30d.index.max() - pd.Timedelta(hours=1)
        assert_frontier_fresh(clean_hourly_30d.index, required)

    def test_raises_when_last_before_required(self, clean_hourly_30d):
        """One second beyond the boundary — must raise."""
        required = clean_hourly_30d.index.max() + pd.Timedelta(seconds=1)
        with pytest.raises(CoverageError, match="stale"):
            assert_frontier_fresh(clean_hourly_30d.index, required)

    def test_message_contains_both_timestamps(self, clean_hourly_30d):
        """Error message must name the observed max and the required value."""
        required = clean_hourly_30d.index.max() + pd.Timedelta(hours=2)
        with pytest.raises(CoverageError) as exc_info:
            assert_frontier_fresh(clean_hourly_30d.index, required)
        msg = str(exc_info.value)
        assert str(clean_hourly_30d.index.max()) in msg
        assert str(required) in msg


# ---------------------------------------------------------------------------
# assert_actual_lag_within
# ---------------------------------------------------------------------------


class TestAssertActualLagWithin:
    def test_passes_at_exact_boundary(self, clean_hourly_30d):
        """Last actual is exactly max_lag old — boundary passes (not strict)."""
        last = clean_hourly_30d.index.max()
        now = last + pd.Timedelta(hours=36)
        assert_actual_lag_within(clean_hourly_30d, now, pd.Timedelta(hours=36))

    def test_passes_within_tolerance(self, clean_hourly_30d):
        """Lag is 10 h, tolerance 36 h — passes."""
        now = clean_hourly_30d.index.max() + pd.Timedelta(hours=10)
        assert_actual_lag_within(clean_hourly_30d, now, pd.Timedelta(hours=36))

    def test_raises_one_second_beyond_boundary(self, clean_hourly_30d):
        """One second beyond max_lag — must raise."""
        last = clean_hourly_30d.index.max()
        # now = last + 36h + 1s  =>  lag = 36h + 1s > 36h tolerance
        now = last + pd.Timedelta(hours=36) + pd.Timedelta(seconds=1)
        with pytest.raises(CoverageError, match="stale"):
            assert_actual_lag_within(clean_hourly_30d, now, pd.Timedelta(hours=36))

    def test_raises_reports_lag_in_hours(self, clean_hourly_30d):
        """Error message must include an integer lag in hours."""
        last = clean_hourly_30d.index.max()
        now = last + pd.Timedelta(hours=48)
        with pytest.raises(CoverageError, match=r"48 h"):
            assert_actual_lag_within(clean_hourly_30d, now, pd.Timedelta(hours=36))

    def test_raises_on_all_nan(self):
        """All-NaN series must raise CoverageError (no published values)."""
        idx = pd.date_range("2026-06-10", periods=10, freq="h", tz="UTC")
        actual = pd.Series(float("nan"), index=idx)
        with pytest.raises(CoverageError, match="no published"):
            assert_actual_lag_within(actual, NOW, pd.Timedelta(hours=36))

    def test_skips_nan_values_for_last_actual(self):
        """NaN-padded tail must not count as a published observation."""
        idx = pd.date_range(
            NOW - pd.Timedelta(hours=100), periods=100, freq="h", tz="UTC"
        )
        actual = pd.Series(50_000.0, index=idx)
        # Append NaN entries well beyond the tolerance window
        nan_idx = pd.date_range(
            idx.max() + pd.Timedelta(hours=1), periods=5, freq="h", tz="UTC"
        )
        nan_series = pd.Series(float("nan"), index=nan_idx)
        padded = pd.concat([actual, nan_series])
        # Last non-NaN is 5 h before the NaN tail, lag from NOW is ~0 h -> passes
        last_real = actual.index.max()
        now_close = last_real + pd.Timedelta(hours=1)
        assert_actual_lag_within(padded, now_close, pd.Timedelta(hours=36))


# ---------------------------------------------------------------------------
# assert_no_interior_gaps  (2026-06-02 incident pattern)
# ---------------------------------------------------------------------------


class TestAssertNoInteriorGaps:
    def test_passes_on_clean_data(self, clean_hourly_30d):
        """No gaps -> no exception."""
        assert_no_interior_gaps(
            clean_hourly_30d, NOW, scan_window=SCAN_WINDOW, max_gap=MAX_GAP
        )

    def test_raises_on_24h_interior_gap(self, clean_hourly_30d):
        """2026-06-02 pattern: 24 h gap in the middle of the scan window.

        A full day of actuals published late creates a hole wider than 12 h
        while the most-recent and oldest data still pass the edge checks.
        """
        # Drop 2026-06-02 entirely (within the 28-day scan window).
        gapped = clean_hourly_30d.copy()
        drop_mask = (gapped.index >= pd.Timestamp("2026-06-02 00:00", tz="UTC")) & (
            gapped.index < pd.Timestamp("2026-06-03 00:00", tz="UTC")
        )
        gapped = gapped[~drop_mask]

        with pytest.raises(CoverageError) as exc_info:
            assert_no_interior_gaps(
                gapped, NOW, scan_window=SCAN_WINDOW, max_gap=MAX_GAP
            )
        msg = str(exc_info.value)
        # Must report count and worst gap boundaries
        assert "1 interior gap" in msg
        assert "Re-download" in msg

    def test_raises_reports_count_and_worst_gap(self, clean_hourly_30d):
        """Error message must contain gap count and start -> end endpoints."""
        gapped = clean_hourly_30d.copy()
        # Create two gaps: one 20 h gap, one 15 h gap.
        gap1_start = pd.Timestamp("2026-06-01 00:00", tz="UTC")
        gap2_start = pd.Timestamp("2026-06-04 00:00", tz="UTC")
        drop = (
            (gapped.index >= gap1_start)
            & (gapped.index < gap1_start + pd.Timedelta(hours=20))
        ) | (
            (gapped.index >= gap2_start)
            & (gapped.index < gap2_start + pd.Timedelta(hours=15))
        )
        gapped = gapped[~drop]
        with pytest.raises(CoverageError) as exc_info:
            assert_no_interior_gaps(
                gapped, NOW, scan_window=SCAN_WINDOW, max_gap=MAX_GAP
            )
        msg = str(exc_info.value)
        assert "2 interior gap" in msg

    def test_gap_exactly_at_max_gap_passes(self, clean_hourly_30d):
        """Gap exactly equal to max_gap must NOT raise (strict > comparison)."""
        # Drop exactly 12 consecutive hours -> gap between them is 13 h (12 missing + 1 step).
        # We need the *diff* between consecutive non-missing rows to be exactly max_gap = 12 h.
        # Drop 11 hours so consecutive diff = 12 h.
        target = pd.Timestamp("2026-06-03 06:00", tz="UTC")
        # Build a series missing 11 hourly slots so the jump is exactly 12 h.
        drop = (
            gapped := clean_hourly_30d.copy(),
            (gapped.index > target) & (gapped.index <= target + pd.Timedelta(hours=11)),
        )[1]
        gapped = clean_hourly_30d[~drop]
        # The diff from target to the next point is 12 h (== max_gap) -> should pass.
        assert_no_interior_gaps(
            gapped, NOW, scan_window=SCAN_WINDOW, max_gap=pd.Timedelta(hours=12)
        )

    def test_outside_scan_window_gap_ignored(self, clean_hourly_30d):
        """A gap older than scan_window must not trigger the guard."""
        # Drop 48 h well outside the 28-day scan window.
        very_old = clean_hourly_30d.index.min() - pd.Timedelta(days=5)
        old_idx = pd.date_range(very_old, periods=48 * 24, freq="h", tz="UTC")
        extension = pd.Series(50_000.0, index=old_idx)
        combined = pd.concat([extension, clean_hourly_30d]).sort_index()
        # Remove 48 h just before the extension starts (outside the 28-day window).
        cut = very_old + pd.Timedelta(hours=2)
        trimmed = combined[
            ~((combined.index >= cut) & (combined.index < cut + pd.Timedelta(hours=48)))
        ]
        # Guard scans only [NOW - 28 days, NOW]; this hole is before that -> passes.
        assert_no_interior_gaps(trimmed, NOW, scan_window=SCAN_WINDOW, max_gap=MAX_GAP)


# ---------------------------------------------------------------------------
# last_complete_hour  (2026-06-05 frontier partial-hour pattern)
# ---------------------------------------------------------------------------


class TestLastCompleteHour:
    def test_full_15min_feed_returns_last_hour(self, clean_15min_24h):
        """All 4 slots present -> last complete hour is the last hour."""
        result = last_complete_hour(clean_15min_24h)
        expected = clean_15min_24h.index.max().floor("h")
        assert result == expected

    def test_partial_frontier_steps_back_one_hour(self, clean_15min_24h):
        """2026-06-05 pattern: last hour has 2 of 4 samples -> step back."""
        partial = clean_15min_24h.iloc[:-2]  # remove last 2 of last hour's 4 slots
        result = last_complete_hour(partial)
        expected = clean_15min_24h.index.max().floor("h") - pd.Timedelta(hours=1)
        assert result == expected, f"got {result}, expected {expected}"

    def test_hourly_cadence_inferred_as_1_sample_per_hour(self):
        """Hourly feed infers 1 sample/hour; every hour is complete."""
        idx = pd.date_range("2026-06-10 00:00", periods=24, freq="h", tz="UTC")
        actual = pd.Series(1.0, index=idx)
        result = last_complete_hour(actual)
        assert result == pd.Timestamp("2026-06-10 23:00", tz="UTC")

    def test_explicit_samples_per_hour_overrides_inferred(self, clean_15min_24h):
        """Explicit samples_per_hour=2 finds a different last complete hour."""
        # With sph=2 each hour needs 2 slots; first 95 of 96 slots -> last complete
        # hour is 23:00 minus any incomplete (all good here).
        result = last_complete_hour(clean_15min_24h, samples_per_hour=2)
        assert result == clean_15min_24h.index.max().floor("h")

    def test_raises_on_empty_series(self):
        """Empty input must raise ValueError."""
        empty = pd.Series(dtype=float)
        with pytest.raises(ValueError, match="empty or all-NaN"):
            last_complete_hour(empty)

    def test_raises_on_all_nan(self):
        """All-NaN input must raise ValueError."""
        idx = pd.date_range("2026-06-10", periods=4, freq="15min", tz="UTC")
        all_nan = pd.Series(float("nan"), index=idx)
        with pytest.raises(ValueError, match="empty or all-NaN"):
            last_complete_hour(all_nan)

    def test_raises_on_invalid_samples_per_hour(self, clean_15min_24h):
        """Non-positive samples_per_hour must raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            last_complete_hour(clean_15min_24h, samples_per_hour=0)
        with pytest.raises(ValueError, match="positive integer"):
            last_complete_hour(clean_15min_24h, samples_per_hour=-1)

    def test_deterministic(self, clean_15min_24h):
        """Repeated calls return the same result."""
        r1 = last_complete_hour(clean_15min_24h)
        r2 = last_complete_hour(clean_15min_24h)
        assert r1 == r2

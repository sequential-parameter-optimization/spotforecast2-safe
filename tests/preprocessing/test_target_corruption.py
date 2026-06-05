# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the target-corruption detector and abort/heal/truncate policy.

Fixtures use synthetic 15-min data replicating the 2026-06-03..05 incident
pattern: multi-GW intra-hour dropouts interspersed with clean hours, at
values within plausible ENTSO-E absolute levels (40–75 GW).  Level-based
detection cannot flag these; only dynamics (range/step rules) can.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.exceptions import TargetCorruptionError
from spotforecast2_safe.preprocessing.target_corruption import (
    apply_target_corruption_policy,
    detect_target_corruption,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DAYS = 7
CADENCE = "15min"
SLOTS_PER_HOUR = 4
BASE_MW = 55_000.0  # plausible ENTSO-E DE level


@pytest.fixture(scope="module")
def clean_15min() -> pd.DataFrame:
    """One week of clean 15-min data, no anomalies."""
    idx = pd.date_range(
        "2026-05-27", periods=N_DAYS * 24 * SLOTS_PER_HOUR, freq=CADENCE, tz="UTC"
    )
    rng = np.random.default_rng(42)
    vals = BASE_MW + rng.normal(0, 500, len(idx))
    return pd.DataFrame({"load": vals}, index=idx)


@pytest.fixture(scope="module")
def corrupt_15min() -> pd.DataFrame:
    """15-min data with a multi-GW intra-hour dropout on the last day.

    The dropout is at hour 12 of the last day: slot 12:15 drops by 11 GW
    then recovers at 12:30.  All values remain within 40–75 GW so level
    detection cannot flag them.
    """
    idx = pd.date_range(
        "2026-05-27", periods=N_DAYS * 24 * SLOTS_PER_HOUR, freq=CADENCE, tz="UTC"
    )
    rng = np.random.default_rng(7)
    vals = BASE_MW + rng.normal(0, 500, len(idx))
    # Inject: last day = idx[-24*4:]; hour 12 starts at offset 12*4=48 from day start.
    dropout_slot = len(idx) - 24 * SLOTS_PER_HOUR + 12 * SLOTS_PER_HOUR + 1
    vals[dropout_slot] = BASE_MW - 11_000  # 11 GW step drop (still 44 GW, plausible)
    return pd.DataFrame({"load": vals}, index=idx)


@pytest.fixture(scope="module")
def hourly_clean() -> pd.DataFrame:
    """Three days of hourly data, no anomalies."""
    idx = pd.date_range("2026-06-01", periods=3 * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(99)
    vals = BASE_MW + rng.normal(0, 500, len(idx))
    return pd.DataFrame({"load": vals}, index=idx)


@pytest.fixture(scope="module")
def hourly_corrupt() -> pd.DataFrame:
    """Hourly data with a large step (range rule vacuous, step rule should fire)."""
    idx = pd.date_range("2026-06-01", periods=3 * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(100)
    vals = BASE_MW + rng.normal(0, 500, len(idx))
    # Insert a 12 GW step at slot 12
    vals[12] = BASE_MW - 12_000
    return pd.DataFrame({"load": vals}, index=idx)


# ---------------------------------------------------------------------------
# detect_target_corruption
# ---------------------------------------------------------------------------


class TestDetectTargetCorruption:
    def test_range_rule_flags_corrupt_hour(self, corrupt_15min):
        mask = detect_target_corruption(
            corrupt_15min,
            targets=["load"],
            range_mw=5_000,
            step_mw=None,
            window_days=3,
        )
        assert mask.any(), "Range rule must flag at least one slot"

    def test_step_rule_flags_corrupt_hour(self, corrupt_15min):
        mask = detect_target_corruption(
            corrupt_15min,
            targets=["load"],
            range_mw=None,
            step_mw=8_000,
            window_days=3,
        )
        assert mask.any(), "Step rule must flag at least one slot"

    def test_whole_hour_is_flagged(self, corrupt_15min):
        """All four slots in a flagged hour must be True."""
        mask = detect_target_corruption(
            corrupt_15min,
            targets=["load"],
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
        )
        flagged_idx = corrupt_15min.index[mask]
        floored = flagged_idx.floor("h")
        # For every flagged hour, all slots in that hour in the frame must be flagged.
        for h in set(floored):
            slots_in_hour = corrupt_15min.index[corrupt_15min.index.floor("h") == h]
            assert mask.loc[
                slots_in_hour
            ].all(), f"All slots in hour {h} must be flagged"

    def test_clean_data_no_flags(self, clean_15min):
        mask = detect_target_corruption(
            clean_15min,
            targets=["load"],
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
        )
        assert not mask.any(), "Clean data must produce no flags"

    def test_window_scoping_excludes_old_corruption(self):
        """A dropout outside the look-back window must not be flagged."""
        idx = pd.date_range(
            "2026-05-01", periods=30 * 24 * SLOTS_PER_HOUR, freq=CADENCE, tz="UTC"
        )
        vals = [BASE_MW] * len(idx)
        # Inject corruption 25 days before the end — outside a 3-day window.
        vals[5 * 24 * SLOTS_PER_HOUR + 3] = BASE_MW - 11_000
        df = pd.DataFrame({"load": vals}, index=idx)
        mask = detect_target_corruption(
            df, targets=["load"], range_mw=5_000, step_mw=8_000, window_days=3
        )
        assert not mask.any(), "Corruption outside the window must not be flagged"

    def test_window_clamped_to_index_min(self):
        """window_days longer than the series must not raise."""
        n = 4 * SLOTS_PER_HOUR  # 16 slots = 4 hours
        idx = pd.date_range("2026-06-01", periods=n, freq=CADENCE, tz="UTC")
        vals = [BASE_MW] * n
        # Inject an 11 GW step drop at slot 1 (00:15 UTC) -> flags the 00:00 hour
        vals[1] = BASE_MW - 11_000
        df = pd.DataFrame({"load": vals}, index=idx)
        # window_days=30 >> data length: should clamp and still flag
        mask = detect_target_corruption(
            df, targets=["load"], range_mw=5_000, step_mw=8_000, window_days=30
        )
        assert mask.any()

    def test_detector_inert_without_window_days(self, corrupt_15min):
        mask = detect_target_corruption(
            corrupt_15min,
            targets=["load"],
            range_mw=5_000,
            step_mw=8_000,
            window_days=None,
        )
        assert not mask.any(), "Detector must be inert when window_days is None"

    def test_detector_inert_without_thresholds(self, corrupt_15min):
        mask = detect_target_corruption(
            corrupt_15min,
            targets=["load"],
            range_mw=None,
            step_mw=None,
            window_days=3,
        )
        assert not mask.any(), "Detector must be inert when both thresholds are None"

    def test_detector_inert_only_range_mw_no_window(self, corrupt_15min):
        """range_mw alone (no window_days) must be inert."""
        mask = detect_target_corruption(
            corrupt_15min,
            targets=["load"],
            range_mw=5_000,
            step_mw=None,
            window_days=None,
        )
        assert not mask.any()

    def test_hourly_range_rule_skipped_step_fires(self, hourly_corrupt):
        """Range rule is vacuous on hourly cadence; step rule still fires."""
        # Range rule alone: should NOT flag (no intra-hour slots)
        mask_range = detect_target_corruption(
            hourly_corrupt,
            targets=["load"],
            range_mw=5_000,
            step_mw=None,
            window_days=3,
        )
        assert not mask_range.any(), "Range rule must be skipped for hourly cadence"

        # Step rule alone: should flag
        mask_step = detect_target_corruption(
            hourly_corrupt,
            targets=["load"],
            range_mw=None,
            step_mw=8_000,
            window_days=3,
        )
        assert mask_step.any(), "Step rule must fire on hourly cadence"


# ---------------------------------------------------------------------------
# apply_target_corruption_policy — noop
# ---------------------------------------------------------------------------


class TestPolicyNoop:
    def test_noop_returns_identical_object(self, clean_15min):
        log = logging.getLogger("test_noop")
        df_out, report = apply_target_corruption_policy(
            clean_15min,
            targets=["load"],
            policy="abort",
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
            max_heal_hours=0,
            anchor_zone_hours=168,
            cutoff=None,
            logger=log,
        )
        assert df_out is clean_15min, "Noop must return the exact same object"
        assert not report.fired
        assert report.action == "noop"
        assert report.n_flagged_cells == 0
        assert report.n_flagged_hours == 0
        assert report.spans == []
        assert report.first_flagged_hour is None
        assert report.pre_policy_last_target is None

    def test_noop_when_detector_inert(self, corrupt_15min):
        """All-None knobs must produce noop even on corrupt data."""
        log = logging.getLogger("test_noop_inert")
        df_out, report = apply_target_corruption_policy(
            corrupt_15min,
            targets=["load"],
            policy="abort",
            range_mw=None,
            step_mw=None,
            window_days=None,
            max_heal_hours=0,
            anchor_zone_hours=168,
            cutoff=None,
            logger=log,
        )
        assert df_out is corrupt_15min
        assert report.action == "noop"


# ---------------------------------------------------------------------------
# apply_target_corruption_policy — abort
# ---------------------------------------------------------------------------


class TestPolicyAbort:
    def test_abort_raises_typed_exception(self, corrupt_15min):
        log = logging.getLogger("test_abort")
        with pytest.raises(TargetCorruptionError) as exc_info:
            apply_target_corruption_policy(
                corrupt_15min,
                targets=["load"],
                policy="abort",
                range_mw=5_000,
                step_mw=8_000,
                window_days=3,
                max_heal_hours=0,
                anchor_zone_hours=168,
                cutoff=None,
                logger=log,
            )
        msg = str(exc_info.value)
        assert "abort" in msg
        assert "ENTSO-E" in msg

    def test_abort_message_contains_span_info(self, corrupt_15min):
        log = logging.getLogger("test_abort_spans")
        with pytest.raises(TargetCorruptionError) as exc_info:
            apply_target_corruption_policy(
                corrupt_15min,
                targets=["load"],
                policy="abort",
                range_mw=5_000,
                step_mw=8_000,
                window_days=3,
                max_heal_hours=0,
                anchor_zone_hours=168,
                cutoff=None,
                logger=log,
            )
        # spans are embedded in the message
        msg = str(exc_info.value)
        assert "span" in msg.lower() or "2026-06" in msg


# ---------------------------------------------------------------------------
# apply_target_corruption_policy — heal
# ---------------------------------------------------------------------------


class TestPolicyHeal:
    def _make_df_with_interior_gap(self):
        """Short frame with corruption far from the cutoff."""
        idx = pd.date_range(
            "2026-06-01", periods=24 * SLOTS_PER_HOUR, freq=CADENCE, tz="UTC"
        )
        vals = [BASE_MW] * len(idx)
        # Dropout at hour 6 (24 slots from start), well within heal budget
        vals[24 + 1] = BASE_MW - 10_000
        return pd.DataFrame({"load": vals}, index=idx)

    def test_heal_nans_flagged_slots_only(self):
        log = logging.getLogger("test_heal")
        df = self._make_df_with_interior_gap()
        cutoff = df.index.max() + pd.Timedelta(
            hours=200
        )  # far future: no anchor conflict

        df_out, report = apply_target_corruption_policy(
            df,
            targets=["load"],
            policy="heal",
            range_mw=5_000,
            step_mw=8_000,
            window_days=2,
            max_heal_hours=4,  # budget covers the 1 h
            anchor_zone_hours=0,  # no zone
            cutoff=cutoff,
            logger=log,
        )
        assert report.fired
        assert report.action == "heal"
        # Flagged slots are NaN in df_out
        flagged_mask = detect_target_corruption(
            df, targets=["load"], range_mw=5_000, step_mw=8_000, window_days=2
        )
        assert df_out.loc[flagged_mask, "load"].isna().all()
        # Clean slots are unchanged
        assert not df_out.loc[~flagged_mask, "load"].isna().any()

    def test_heal_anchor_zone_refusal(self):
        log = logging.getLogger("test_heal_anchor")
        df = self._make_df_with_interior_gap()
        # cutoff set to just after the corrupt hour: zone covers the corruption
        cutoff = pd.Timestamp("2026-06-01 07:00", tz="UTC")
        with pytest.raises(TargetCorruptionError, match="anchor zone"):
            apply_target_corruption_policy(
                df,
                targets=["load"],
                policy="heal",
                range_mw=5_000,
                step_mw=8_000,
                window_days=2,
                max_heal_hours=24,
                anchor_zone_hours=168,
                cutoff=cutoff,
                logger=log,
            )

    def test_heal_over_budget_refusal(self):
        log = logging.getLogger("test_heal_budget")
        df = self._make_df_with_interior_gap()
        cutoff = df.index.max() + pd.Timedelta(hours=200)

        with pytest.raises(TargetCorruptionError, match="heal budget"):
            apply_target_corruption_policy(
                df,
                targets=["load"],
                policy="heal",
                range_mw=5_000,
                step_mw=8_000,
                window_days=2,
                max_heal_hours=0,  # budget = 0 → always refuses when flags found
                anchor_zone_hours=0,
                cutoff=cutoff,
                logger=log,
            )


# ---------------------------------------------------------------------------
# apply_target_corruption_policy — truncate
# ---------------------------------------------------------------------------


class TestPolicyTruncate:
    def test_truncate_nans_from_first_flagged_hour_to_end(self, corrupt_15min):
        log = logging.getLogger("test_truncate")
        df_out, report = apply_target_corruption_policy(
            corrupt_15min,
            targets=["load"],
            policy="truncate",
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
            max_heal_hours=0,
            anchor_zone_hours=168,
            cutoff=None,
            logger=log,
        )
        assert report.fired
        assert report.action == "truncate"
        assert report.first_flagged_hour is not None

        # All slots from first_flagged_hour onward must be NaN.
        tail = df_out.loc[report.first_flagged_hour :, "load"]
        assert tail.isna().all(), "Truncated tail must be all-NaN"

        # Slots before the first_flagged_hour must be intact.
        pre = df_out.loc[: report.first_flagged_hour - pd.Timedelta(seconds=1), "load"]
        assert not pre.isna().any(), "Pre-truncation slots must be untouched"

    def test_truncate_report_fields(self, corrupt_15min):
        log = logging.getLogger("test_truncate_report")
        df_out, report = apply_target_corruption_policy(
            corrupt_15min,
            targets=["load"],
            policy="truncate",
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
            max_heal_hours=0,
            anchor_zone_hours=168,
            cutoff=None,
            logger=log,
        )
        assert report.n_flagged_cells > 0
        assert report.n_flagged_hours > 0
        assert len(report.spans) > 0
        assert isinstance(report.first_flagged_hour, pd.Timestamp)
        assert isinstance(report.pre_policy_last_target, pd.Timestamp)

    def test_truncate_does_not_touch_pre_flagged_hours(self, corrupt_15min):
        log = logging.getLogger("test_truncate_clean")
        df_original = corrupt_15min.copy()
        df_out, report = apply_target_corruption_policy(
            corrupt_15min,
            targets=["load"],
            policy="truncate",
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
            max_heal_hours=0,
            anchor_zone_hours=168,
            cutoff=None,
            logger=log,
        )
        # df_out and corrupt_15min may differ (copy was made) but pre-truncation
        # values in df_out must equal the original.
        pre_mask = df_out.index < report.first_flagged_hour
        pd.testing.assert_series_equal(
            df_out.loc[pre_mask, "load"], df_original.loc[pre_mask, "load"]
        )

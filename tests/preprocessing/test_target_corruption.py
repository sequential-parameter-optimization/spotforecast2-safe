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


# ---------------------------------------------------------------------------
# E1  Gap-spanning ramp must NOT flag
# ---------------------------------------------------------------------------


class TestDetectorGapSpanning:
    """E1: A large value ramp that spans missing rows must not produce false flags."""

    def _make_gapped_df(self, use_nan: bool) -> pd.DataFrame:
        """15-min frame with a 6-hour block missing (rows dropped or NaN).

        Values resume ~9 GW higher after the gap — a legitimate morning ramp
        across missing data.  step_mw=6000 is smaller than the ramp (9000 MW),
        so the step rule would flag if it saw the transition as adjacent.
        """
        idx = pd.date_range(
            "2026-06-01", periods=2 * 24 * SLOTS_PER_HOUR, freq=CADENCE, tz="UTC"
        )
        vals = [50_000.0] * len(idx)
        # After hour 6 the value rises to 59 GW (a 9 GW ramp across the gap).
        gap_start_slot = 6 * SLOTS_PER_HOUR  # 06:00
        gap_end_slot = 12 * SLOTS_PER_HOUR  # 12:00 (exclusive upper bound)
        for i in range(gap_end_slot, len(idx)):
            vals[i] = 59_000.0
        df_full = pd.DataFrame({"load": vals}, index=idx)
        if use_nan:
            # Rows present, values NaN across the gap.
            df_full.loc[idx[gap_start_slot:gap_end_slot], "load"] = float("nan")
            return df_full
        else:
            # Rows entirely removed.
            keep = list(range(gap_start_slot)) + list(range(gap_end_slot, len(idx)))
            return df_full.iloc[keep]

    def test_gap_spanning_ramp_rows_dropped(self):
        """Rows dropped across the gap: step rule must not flag the boundary."""
        df = self._make_gapped_df(use_nan=False)
        mask = detect_target_corruption(
            df,
            targets=["load"],
            range_mw=5_000,
            step_mw=6_000,
            window_days=3,
        )
        assert not mask.any(), (
            "A legitimate ramp across removed rows must not be flagged "
            "(gap boundary is not adjacent — dt != cadence)."
        )

    def test_gap_spanning_ramp_nan_rows(self):
        """Rows present but NaN across the gap: step rule must not flag boundary."""
        df = self._make_gapped_df(use_nan=True)
        mask = detect_target_corruption(
            df,
            targets=["load"],
            range_mw=5_000,
            step_mw=6_000,
            window_days=3,
        )
        assert not mask.any(), (
            "NaN boundary transition (NaN -> value) yields NaN diff which "
            "compares False against step_mw; must not flag."
        )


# ---------------------------------------------------------------------------
# E2  Duplicate timestamps
# ---------------------------------------------------------------------------


class TestDetectorDuplicates:
    """E2: Duplicate timestamps must not crash and must not false-flag the pair.

    The critical semantics: ``dt == 0`` between two rows at the same timestamp
    is never equal to the cadence, so the step rule treats them as non-adjacent
    and does not flag the duplicate pair itself.  A genuine corrupt step placed
    elsewhere in the same frame must still be detected.

    We design the duplicate to have the same value as its partner (so no
    adjacent transition carries a >step_mw diff from or to the duplicate row),
    keeping the duplicate's neighbourhood clean.  Only the genuine corrupt step
    fires.
    """

    def _make_dup_df(self) -> pd.DataFrame:
        """Frame with a duplicate entry (same value) and a genuine corrupt step.

        The duplicate row at 01:00 carries the same value as the original
        01:00 slot, so no large transition is created by the duplicate itself.
        The genuine corrupt step is at 04:15 (11 GW drop).
        """
        idx = pd.date_range(
            "2026-06-01", periods=24 * SLOTS_PER_HOUR, freq=CADENCE, tz="UTC"
        )
        vals = [BASE_MW] * len(idx)
        # Genuine corrupt step at 04:15 (slot 17): 11 GW drop.
        vals[17] = BASE_MW - 11_000
        df = pd.DataFrame({"load": vals}, index=idx)
        # Duplicate slot at 01:00 with the SAME value (clean duplicate).
        dup_ts = idx[4]  # 01:00
        extra = pd.DataFrame({"load": [BASE_MW]}, index=[dup_ts])
        return pd.concat([df, extra]).sort_index()

    def test_no_exception_on_duplicates(self):
        """detect_target_corruption must not raise on a duplicated index."""
        df = self._make_dup_df()
        assert df.index.has_duplicates
        try:
            detect_target_corruption(
                df,
                targets=["load"],
                range_mw=5_000,
                step_mw=8_000,
                window_days=3,
            )
        except Exception as exc:
            pytest.fail(f"detect_target_corruption raised on duplicate index: {exc}")

    def test_duplicate_pair_does_not_flag(self):
        """The duplicate pair must not flag: dt==0 between the two rows is not adjacent."""
        df = self._make_dup_df()
        mask = detect_target_corruption(
            df,
            targets=["load"],
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
        )
        # The duplicate is at 01:00; the 01:00 hour must be clean (only the
        # genuine corrupt step at 04:15 should appear in flagged hours).
        dup_ts = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")
        slots_in_dup_hour = df.index[
            (df.index >= dup_ts) & (df.index < dup_ts + pd.Timedelta(hours=1))
        ]
        assert not mask.loc[slots_in_dup_hour].any(), (
            "The hour containing the duplicate pair must not be flagged "
            "(dt==0 is not adjacent; the duplicate row carries no anomalous step)."
        )

    def test_genuine_corrupt_step_still_flags_with_duplicates(self):
        """A genuine corrupt step elsewhere in the same frame must still be detected."""
        df = self._make_dup_df()
        mask = detect_target_corruption(
            df,
            targets=["load"],
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
        )
        # Genuine corrupt slot 17 is at 04:15; the step flags the 04:00 hour.
        genuine_hour = pd.Timestamp("2026-06-01 04:00:00", tz="UTC")
        slots_in_genuine_hour = df.index[
            (df.index >= genuine_hour)
            & (df.index < genuine_hour + pd.Timedelta(hours=1))
        ]
        assert mask.loc[
            slots_in_genuine_hour
        ].any(), (
            "Genuine corrupt step must be flagged even when duplicates are present."
        )


# ---------------------------------------------------------------------------
# E3  DST safety
# ---------------------------------------------------------------------------


class TestDetectorDST:
    """E3: Detector runs correctly across DST transitions (Europe/Berlin)."""

    def test_fall_back_no_raise(self):
        """Detector must not raise on a 15-min Europe/Berlin index that spans
        the 2025-10-26 fall-back night (ambiguous 02:xx wall time)."""
        idx = pd.date_range(
            "2025-10-24 00:00",
            periods=4 * 24 * 7,
            freq=CADENCE,
            tz="Europe/Berlin",
        )
        vals = [BASE_MW] * len(idx)
        df = pd.DataFrame({"load": vals}, index=idx)
        try:
            mask = detect_target_corruption(
                df, targets=["load"], range_mw=5_000, step_mw=8_000, window_days=7
            )
        except Exception as exc:
            pytest.fail(
                f"detect_target_corruption raised on fall-back DST index: {exc}"
            )
        assert not mask.any(), "Clean DST week must produce no flags."

    def test_fall_back_dropout_is_flagged(self):
        """An 11 GW step on the DST fall-back day must be flagged."""
        idx = pd.date_range(
            "2025-10-24 00:00",
            periods=4 * 24 * 7,
            freq=CADENCE,
            tz="Europe/Berlin",
        )
        vals = [BASE_MW] * len(idx)
        # Find a slot in the 01:00 UTC hour on the fall-back day.
        idx_utc = idx.tz_convert("UTC")
        target_hour_utc = pd.Timestamp("2025-10-26 01:00:00", tz="UTC")
        slots_in_hour = [
            i for i, ts in enumerate(idx_utc) if ts.floor("h") == target_hour_utc
        ]
        assert len(slots_in_hour) >= 2, "Need at least 2 slots to inject a step."
        vals[slots_in_hour[1]] = BASE_MW - 11_000
        df = pd.DataFrame({"load": vals}, index=idx)
        mask = detect_target_corruption(
            df, targets=["load"], range_mw=5_000, step_mw=8_000, window_days=7
        )
        assert mask.any(), "11 GW step on DST day must be flagged."

    def test_spring_forward_no_raise(self):
        """Detector must not raise on a 15-min Europe/Berlin index that spans
        the 2026-03-29 spring-forward night."""
        idx = pd.date_range(
            "2026-03-27 00:00",
            periods=4 * 24 * 7,
            freq=CADENCE,
            tz="Europe/Berlin",
        )
        vals = [BASE_MW] * len(idx)
        df = pd.DataFrame({"load": vals}, index=idx)
        try:
            mask = detect_target_corruption(
                df, targets=["load"], range_mw=5_000, step_mw=8_000, window_days=7
            )
        except Exception as exc:
            pytest.fail(
                f"detect_target_corruption raised on spring-forward DST index: {exc}"
            )
        assert not mask.any(), "Clean spring-forward DST week must produce no flags."


# ---------------------------------------------------------------------------
# E4  Step at the first slot of an hour flags both that hour and predecessor
# ---------------------------------------------------------------------------


class TestDetectorPredecessorHour:
    """E4: A step between the last slot of hour H and the first slot of H+1
    must flag BOTH hour H and hour H+1."""

    def test_step_at_first_slot_flags_both_hours(self):
        """Step between 00:45 and 01:00 must flag both the 00:00 and 01:00 hours."""
        idx = pd.date_range(
            "2026-06-01", periods=8 * SLOTS_PER_HOUR, freq=CADENCE, tz="UTC"
        )
        vals = [BASE_MW] * len(idx)
        # idx[4] = 01:00 is the first slot of hour 1.
        # Step between idx[3]=00:45 and idx[4]=01:00 (both one cadence apart).
        vals[4] = BASE_MW - 11_000
        df = pd.DataFrame({"load": vals}, index=idx)

        mask = detect_target_corruption(
            df, targets=["load"], range_mw=5_000, step_mw=8_000, window_days=7
        )

        h00 = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
        h01 = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")

        slots_h00 = df.index[(df.index >= h00) & (df.index < h01)]
        slots_h01 = df.index[
            (df.index >= h01) & (df.index < h01 + pd.Timedelta(hours=1))
        ]

        assert mask.loc[slots_h00].all(), (
            "All slots in the predecessor hour (00:00) must be flagged when "
            "the step falls at the boundary between 00:45 and 01:00."
        )
        assert mask.loc[
            slots_h01
        ].all(), "All slots in the step's own hour (01:00) must be flagged."


# ---------------------------------------------------------------------------
# Deviation rule (dropout vs reference column)
# ---------------------------------------------------------------------------

# Chapter-style thresholds: the injected dropout is deliberately
# SUB-THRESHOLD for the dynamics rules (steps 5.8 GW < 6 GW, intra-hour
# range 5.8 GW < 8 GW) while sitting 11.6 GW under the reference — the
# 2026-06-07 frontier pattern that motivated the rule.
DEV_RANGE_MW = 8_000
DEV_STEP_MW = 6_000
DEV_MW = 8_000


def _make_deviation_frame(
    *,
    offsets=(5_800, 11_600, 11_600, 5_800),
    dropout_day: int = 2,
    dropout_hour: int = 7,
    nan_tail_slots: int = 8,
) -> pd.DataFrame:
    """Two-column frame: constant forecast, actual with a dropout + NaN tail.

    Constant base values keep the injected dynamics exact (deterministic
    sub-threshold steps); ``offsets`` are subtracted from the forecast at
    the four slots of ``dropout_hour`` on ``dropout_day``.  The last
    ``nan_tail_slots`` actual slots are NaN while the forecast continues —
    the ENTSO-E publication-lag frontier.
    """
    idx = pd.date_range(
        "2026-06-05", periods=3 * 24 * SLOTS_PER_HOUR, freq=CADENCE, tz="UTC"
    )
    forecast = pd.Series(BASE_MW, index=idx)
    actual = forecast.copy()
    start = dropout_day * 24 * SLOTS_PER_HOUR + dropout_hour * SLOTS_PER_HOUR
    for i, off in enumerate(offsets):
        actual.iloc[start + i] = BASE_MW - off
    if nan_tail_slots:
        actual.iloc[-nan_tail_slots:] = np.nan
    return pd.DataFrame({"Actual Load": actual, "Forecasted Load": forecast})


class TestDetectorDeviation:
    """Deviation rule: sustained dropout below a published reference."""

    def _detect(self, df, **kwargs):
        params = dict(
            targets=["Actual Load"],
            range_mw=DEV_RANGE_MW,
            step_mw=DEV_STEP_MW,
            window_days=3,
            deviation_mw=DEV_MW,
            deviation_ref="Forecasted Load",
        )
        params.update(kwargs)
        return detect_target_corruption(df, **params)

    def test_dynamics_rules_miss_the_dropout(self):
        """Control: range/step alone must NOT flag the sub-threshold dropout."""
        df = _make_deviation_frame()
        mask = self._detect(df, deviation_mw=None, deviation_ref=None)
        assert not mask.any(), "sub-threshold dropout must evade dynamics rules"

    def test_deviation_rule_flags_sustained_dropout(self):
        df = _make_deviation_frame()
        mask = self._detect(df)
        dropout_hour = pd.Timestamp("2026-06-07 07:00:00", tz="UTC")
        slots = df.index[
            (df.index >= dropout_hour)
            & (df.index < dropout_hour + pd.Timedelta(hours=1))
        ]
        assert mask.loc[slots].all(), "deviation rule must flag the dropout hour"
        # Nothing else flags: the surrounding clean hours stay clean.
        assert mask.sum() == SLOTS_PER_HOUR

    def test_detector_inert_with_only_deviation_and_no_window(self):
        df = _make_deviation_frame()
        mask = detect_target_corruption(
            df,
            targets=["Actual Load"],
            range_mw=None,
            step_mw=None,
            window_days=None,
            deviation_mw=DEV_MW,
            deviation_ref="Forecasted Load",
        )
        assert not mask.any(), "window_days=None must keep the detector inert"

    def test_deviation_only_configuration_activates_detector(self):
        """deviation_mw alone (range/step None) must activate the detector."""
        df = _make_deviation_frame()
        mask = self._detect(df, range_mw=None, step_mw=None)
        assert mask.any()

    def test_single_slot_dip_not_flagged_with_slots_2(self):
        df = _make_deviation_frame(offsets=(11_600,))
        # One isolated slot 11.6 GW below: steps are 5.8+5.8 GW? No — a
        # single 11.6 GW offset creates 11.6 GW steps, so disable the step
        # rule to isolate the deviation-slots semantics.
        mask = self._detect(df, range_mw=None, step_mw=None, deviation_slots=2)
        assert not mask.any(), "single-slot blip must not flag at deviation_slots=2"

    def test_single_slot_dip_flagged_with_slots_1(self):
        df = _make_deviation_frame(offsets=(11_600,))
        mask = self._detect(df, range_mw=None, step_mw=None, deviation_slots=1)
        assert mask.any(), "deviation_slots=1 must flag a single-slot dropout"

    def test_missing_ref_column_is_inert(self):
        df = _make_deviation_frame()
        mask = self._detect(df, deviation_ref="NoSuchColumn")
        assert not mask.any(), "absent reference column must skip the rule silently"

    def test_ref_none_is_inert(self):
        df = _make_deviation_frame()
        mask = self._detect(df, deviation_ref=None)
        assert not mask.any()

    def test_frontier_nan_not_flagged(self):
        """Publication-lag tail (forecast published, actual NaN) never flags."""
        df = _make_deviation_frame()
        mask = self._detect(df)
        tail = df.index[df["Actual Load"].isna()]
        assert len(tail) > 0
        assert not mask.loc[
            tail
        ].any(), "NaN-actual frontier slots must never be flagged"

    def test_nan_breaks_consecutive_run(self):
        """A NaN between two below-threshold slots must break the run."""
        df = _make_deviation_frame(offsets=(11_600, 11_600))
        start = 2 * 24 * SLOTS_PER_HOUR + 7 * SLOTS_PER_HOUR
        df.iloc[start + 1, df.columns.get_loc("Actual Load")] = np.nan
        mask = self._detect(df, range_mw=None, step_mw=None, deviation_slots=2)
        assert not mask.any(), "a NaN gap must break the consecutive-slot requirement"

    def test_positive_deviation_not_flagged(self):
        """Actuals far ABOVE the forecast are under-forecasting, not corruption."""
        df = _make_deviation_frame(offsets=(-12_000, -12_000, -12_000, -12_000))
        mask = self._detect(df, range_mw=None, step_mw=None)
        assert not mask.any(), "the deviation rule is dropout-only by design"

    def test_hourly_cadence_clamps_slots_to_one(self):
        """On hourly data the sustained requirement collapses to one slot."""
        idx = pd.date_range("2026-06-05", periods=3 * 24, freq="h", tz="UTC")
        forecast = pd.Series(BASE_MW, index=idx)
        actual = forecast.copy()
        actual.iloc[60] = BASE_MW - 12_000  # one corrupt hour
        df = pd.DataFrame({"Actual Load": actual, "Forecasted Load": forecast})
        mask = detect_target_corruption(
            df,
            targets=["Actual Load"],
            range_mw=None,
            step_mw=None,
            window_days=3,
            deviation_mw=DEV_MW,
            deviation_ref="Forecasted Load",
            deviation_slots=2,
        )
        assert mask.any(), "hourly cadence: a single corrupt hour must flag"

    def test_truncate_keeps_reference_column(self):
        """policy='truncate' with scoped targets NaNs the actual only."""
        df = _make_deviation_frame()
        n_forecast_obs = int(df["Forecasted Load"].notna().sum())
        df_out, report = apply_target_corruption_policy(
            df,
            targets=["Actual Load"],
            policy="truncate",
            range_mw=DEV_RANGE_MW,
            step_mw=DEV_STEP_MW,
            window_days=3,
            max_heal_hours=0,
            anchor_zone_hours=168,
            cutoff=None,
            logger=logging.getLogger("test-deviation"),
            deviation_mw=DEV_MW,
            deviation_ref="Forecasted Load",
        )
        assert report.fired
        assert report.action == "truncate"
        assert report.first_flagged_hour == pd.Timestamp(
            "2026-06-07 07:00:00", tz="UTC"
        )
        assert (
            df_out.loc[report.first_flagged_hour :, "Actual Load"].isna().all()
        ), "actual must be NaN from the first flagged hour onward"
        assert (
            int(df_out["Forecasted Load"].notna().sum()) == n_forecast_obs
        ), "the reference column must survive truncate untouched"

    def test_abort_on_deviation(self):
        df = _make_deviation_frame()
        with pytest.raises(TargetCorruptionError, match="corrupt hour"):
            apply_target_corruption_policy(
                df,
                targets=["Actual Load"],
                policy="abort",
                range_mw=DEV_RANGE_MW,
                step_mw=DEV_STEP_MW,
                window_days=3,
                max_heal_hours=0,
                anchor_zone_hours=168,
                cutoff=None,
                logger=logging.getLogger("test-deviation"),
                deviation_mw=DEV_MW,
                deviation_ref="Forecasted Load",
            )

    def test_dst_fall_back_deviation_flagged(self):
        """Deviation rule works on a Europe/Berlin frame across fall-back."""
        idx = pd.date_range(
            "2025-10-24 00:00",
            periods=4 * 24 * 4,
            freq=CADENCE,
            tz="Europe/Berlin",
        )
        forecast = pd.Series(BASE_MW, index=idx)
        actual = forecast.copy()
        # Sustained 12 GW dropout in the 01:00 UTC hour of the fall-back day.
        idx_utc = idx.tz_convert("UTC")
        target_hour_utc = pd.Timestamp("2025-10-26 01:00:00", tz="UTC")
        slots = [i for i, ts in enumerate(idx_utc) if ts.floor("h") == target_hour_utc]
        assert len(slots) >= 2
        for i in slots[:2]:
            actual.iloc[i] = BASE_MW - 12_000
        df = pd.DataFrame({"Actual Load": actual, "Forecasted Load": forecast})
        mask = detect_target_corruption(
            df,
            targets=["Actual Load"],
            range_mw=None,
            step_mw=None,
            window_days=7,
            deviation_mw=DEV_MW,
            deviation_ref="Forecasted Load",
        )
        assert mask.any(), "deviation dropout on the DST day must be flagged"

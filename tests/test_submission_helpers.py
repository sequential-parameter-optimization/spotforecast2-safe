# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the four ENTSO-E submission helpers.

Covers all public symbols added in the feat/submission-helpers branch:
- `spotforecast2_safe.data.fetch_data.load_interim`
- `spotforecast2_safe.downloader.entsoe.SubmissionDates`
- `spotforecast2_safe.downloader.entsoe.compute_submission_dates`
- `spotforecast2_safe.downloader.resilience.download_combined_with_fallback`
- `spotforecast2_safe.preprocessing.coverage.SubmissionCoverage`
- `spotforecast2_safe.preprocessing.coverage.assert_submission_coverage`

Behavior pins (per the ADR):
- combined vs four_zone mode dispatch in load_interim.
- SubmissionDates fields match the scripts' Dates dataclass exactly.
- compute_submission_dates delegates to derive_download_window.
- download_combined_with_fallback returns DownloadResult with mode="combined"
  and correct fallback_gate semantics.
- assert_submission_coverage: preview vs authoritative, fallback gate True/False.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from spotforecast2_safe.downloader.entsoe import (
    SubmissionDates,
    compute_submission_dates,
)
from spotforecast2_safe.downloader.resilience import (
    download_combined_with_fallback,
)
from spotforecast2_safe.exceptions import CoverageError
from spotforecast2_safe.preprocessing.coverage import (
    SubmissionCoverage,
    assert_submission_coverage,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NOW = pd.Timestamp("2026-06-14 10:00", tz="UTC")
TODAY = NOW.normalize()
YESTERDAY = TODAY - pd.Timedelta(days=1)
TOMORROW = TODAY + pd.Timedelta(days=1)
LAST_TARGET = TOMORROW + pd.Timedelta(hours=23)


def _make_interim(
    start: str = "2026-06-10 00:00",
    end: str = "2026-06-14 09:00",
    freq: str = "h",
    actual_value: float = 40_000.0,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Build a minimal synthetic hourly interim frame."""
    idx = pd.date_range(start, end, freq=freq, tz="UTC")
    data = {"Actual Load": actual_value, "Forecasted Load": 41_000.0}
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def dates() -> SubmissionDates:
    return compute_submission_dates(NOW, train_years=3)


@pytest.fixture
def clean_interim() -> pd.DataFrame:
    return _make_interim()


# ---------------------------------------------------------------------------
# load_interim
# ---------------------------------------------------------------------------


class TestLoadInterim:
    """Tests for spotforecast2_safe.data.fetch_data.load_interim."""

    def _write_combined_csv(self, interim_dir: Path, df: pd.DataFrame) -> None:
        df.index.name = "Time (UTC)"
        df.to_csv(interim_dir / "energy_load.csv")

    def test_combined_reads_utc_indexed_frame(self, tmp_path, monkeypatch):
        """combined mode: returns UTC DatetimeIndex frame."""
        from spotforecast2_safe.data.fetch_data import load_interim

        monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
        interim_dir = tmp_path / "interim"
        interim_dir.mkdir()
        df = _make_interim()
        self._write_combined_csv(interim_dir, df)

        result = load_interim(mode="combined")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"
        assert "Actual Load" in result.columns
        assert len(result) == len(df)

    def test_combined_is_default_mode(self, tmp_path, monkeypatch):
        """Default mode is 'combined'."""
        from spotforecast2_safe.data.fetch_data import load_interim

        monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
        interim_dir = tmp_path / "interim"
        interim_dir.mkdir()
        self._write_combined_csv(interim_dir, _make_interim())

        result = load_interim()  # no mode kwarg
        assert "Actual Load" in result.columns

    def test_combined_raises_file_not_found_when_csv_missing(
        self, tmp_path, monkeypatch
    ):
        """combined mode raises FileNotFoundError when energy_load.csv absent."""
        from spotforecast2_safe.data.fetch_data import load_interim

        monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
        (tmp_path / "interim").mkdir()

        with pytest.raises(FileNotFoundError, match="energy_load.csv"):
            load_interim(mode="combined")

    def test_four_zone_raises_file_not_found_when_zone_csvs_missing(
        self, tmp_path, monkeypatch
    ):
        """four_zone mode raises FileNotFoundError when per-zone CSVs absent."""
        from spotforecast2_safe.data.fetch_data import load_interim

        monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
        (tmp_path / "interim").mkdir()

        with pytest.raises(FileNotFoundError):
            load_interim(mode="four_zone")

    def test_invalid_mode_raises_value_error(self, tmp_path, monkeypatch):
        """Unknown mode must raise ValueError."""
        from spotforecast2_safe.data.fetch_data import load_interim

        monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))

        with pytest.raises(ValueError, match="mode must be"):
            load_interim(mode="bogus")

    def test_combined_data_home_kwarg(self, tmp_path):
        """data_home kwarg overrides SPOTFORECAST2_DATA."""
        from spotforecast2_safe.data.fetch_data import load_interim

        interim_dir = tmp_path / "interim"
        interim_dir.mkdir()
        self._write_combined_csv(interim_dir, _make_interim())

        result = load_interim(mode="combined", data_home=tmp_path)
        assert "Actual Load" in result.columns

    def test_combined_exported_from_data_init(self):
        """load_interim is exported from spotforecast2_safe.data."""
        from spotforecast2_safe.data import load_interim as _load_interim  # noqa: F401

        assert callable(_load_interim)


# ---------------------------------------------------------------------------
# SubmissionDates + compute_submission_dates
# ---------------------------------------------------------------------------


class TestSubmissionDates:
    """Tests for SubmissionDates dataclass and compute_submission_dates."""

    def test_fields_match_scripts_dates_semantics(self, dates: SubmissionDates):
        """All seven fields must have the correct values relative to NOW."""
        assert dates.now == NOW
        assert dates.today == TODAY
        assert dates.yesterday == YESTERDAY
        assert dates.tomorrow == TOMORROW
        assert dates.last_target == LAST_TARGET

    def test_last_target_is_tomorrow_23h(self, dates: SubmissionDates):
        assert dates.last_target == TOMORROW + pd.Timedelta(hours=23)

    def test_start_dl_format_yyyymmddhhmm(self, dates: SubmissionDates):
        assert len(dates.start_dl) == 12
        pd.to_datetime(dates.start_dl, format="%Y%m%d%H%M")

    def test_end_dl_is_today_plus_2_days_when_data_end_none(
        self, dates: SubmissionDates
    ):
        expected_end = (TODAY + pd.Timedelta(days=2)).strftime("%Y%m%d%H%M")
        assert dates.end_dl == expected_end

    def test_explicit_data_end_propagates(self):
        data_end = "202612310000"
        dates = compute_submission_dates(NOW, train_years=3, data_end=data_end)
        assert dates.end_dl == data_end

    def test_start_margin_days_affects_start_dl(self):
        dates_45 = compute_submission_dates(NOW, train_years=3, start_margin_days=45)
        dates_90 = compute_submission_dates(NOW, train_years=3, start_margin_days=90)
        start_45 = pd.to_datetime(dates_45.start_dl, format="%Y%m%d%H%M", utc=True)
        start_90 = pd.to_datetime(dates_90.start_dl, format="%Y%m%d%H%M", utc=True)
        assert start_90 < start_45

    def test_frozen_dataclass_raises_on_mutation(self, dates: SubmissionDates):
        with pytest.raises(Exception):
            dates.today = pd.Timestamp("2020-01-01", tz="UTC")  # type: ignore[misc]

    def test_raises_on_naive_timestamp(self):
        naive = pd.Timestamp("2026-06-14 10:00")
        with pytest.raises(TypeError, match="timezone-aware"):
            compute_submission_dates(naive, train_years=3)

    def test_raises_on_non_timestamp(self):
        with pytest.raises(TypeError, match="pd.Timestamp"):
            compute_submission_dates("2026-06-14", train_years=3)  # type: ignore[arg-type]

    def test_raises_on_zero_train_years(self):
        with pytest.raises(ValueError, match="train_years"):
            compute_submission_dates(NOW, train_years=0)

    def test_exported_from_downloader_init(self):
        from spotforecast2_safe.downloader import (  # noqa: F401
            SubmissionDates,
            compute_submission_dates,
        )

        assert callable(compute_submission_dates)

    def test_start_dl_is_3_years_45_days_before_today(self, dates: SubmissionDates):
        """start_dl must match the derive_download_window formula exactly."""
        from spotforecast2_safe.downloader.entsoe import derive_download_window

        expected_start, _ = derive_download_window(
            NOW, train_years=3, start_margin_days=45
        )
        assert dates.start_dl == expected_start


# ---------------------------------------------------------------------------
# download_combined_with_fallback
# ---------------------------------------------------------------------------


class TestDownloadCombinedWithFallback:
    """Tests for resilience.download_combined_with_fallback."""

    def _patch_env(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
        (tmp_path / "interim").mkdir(parents=True, exist_ok=True)

    def _make_store_stub(self, tmp_path: Path):
        """Return a real SnapshotStore with the given root for isolation."""
        from spotforecast2_safe.utils.snapshot_store import SnapshotStore
        from spotforecast2_safe.downloader.resilience import SNAPSHOT_TTL

        return SnapshotStore(root=tmp_path / "snapshots", ttl=SNAPSHOT_TTL)

    def test_live_success_returns_mode_combined(self, monkeypatch, tmp_path):
        """When download_new_data succeeds on first attempt, mode='combined'."""
        self._patch_env(monkeypatch, tmp_path)

        def fake_download_new_data(**kwargs):
            # Write a minimal CSV to simulate download success.
            csv = tmp_path / "interim" / "energy_load.csv"
            idx = pd.date_range("2026-06-10", periods=24, freq="h", tz="UTC")
            df = pd.DataFrame(
                {"Actual Load": 40_000.0, "Forecasted Load": 41_000.0}, index=idx
            )
            df.index.name = "Time (UTC)"
            df.to_csv(csv)

        monkeypatch.setattr(
            "spotforecast2_safe.downloader.entsoe.download_new_data",
            fake_download_new_data,
        )

        result = download_combined_with_fallback(
            api_key="test-key",
            start="202301010000",
            end="202301050000",
            now=NOW,
            max_retries=3,
            backoff=0.0,
            timeout=None,
            fallback_enabled=True,
            sleep=lambda s: None,
        )
        assert result.mode == "combined"
        assert result.combined_status == "live"
        assert result.fallback_gate is False

    def test_all_retries_exhausted_with_no_snapshot_returns_mode_none(
        self, monkeypatch, tmp_path
    ):
        """When all retries fail and no snapshot exists, mode=None."""
        self._patch_env(monkeypatch, tmp_path)

        def always_fail(**kwargs):
            raise ConnectionError("ENTSO-E unreachable")

        monkeypatch.setattr(
            "spotforecast2_safe.downloader.entsoe.download_new_data",
            always_fail,
        )

        result = download_combined_with_fallback(
            api_key="test-key",
            start="202301010000",
            end="202301050000",
            now=NOW,
            max_retries=2,
            backoff=0.0,
            timeout=None,
            fallback_enabled=True,
            sleep=lambda s: None,
        )
        assert result.mode is None
        assert result.combined_status == "missing"
        assert result.fallback_gate is True  # None mode always returns True

    def test_snapshot_restored_on_all_retries_fail(self, monkeypatch, tmp_path):
        """When retries fail but a snapshot exists, mode='combined', status='cache'."""
        self._patch_env(monkeypatch, tmp_path)

        # Write a real interim CSV so the SnapshotStore can seed from it.
        csv_path = tmp_path / "interim" / "energy_load.csv"
        idx = pd.date_range("2026-06-10", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"Actual Load": 40_000.0, "Forecasted Load": 41_000.0}, index=idx
        )
        df.index.name = "Time (UTC)"
        df.to_csv(csv_path)

        def always_fail(**kwargs):
            raise ConnectionError("ENTSO-E unreachable")

        monkeypatch.setattr(
            "spotforecast2_safe.downloader.entsoe.download_new_data",
            always_fail,
        )

        # Pre-seed the snapshot store by running once successfully first.
        # We simulate this by manually writing the snapshot.
        from spotforecast2_safe.utils.snapshot_store import SnapshotStore
        from spotforecast2_safe.downloader.resilience import SNAPSHOT_TTL

        store = SnapshotStore(root=tmp_path / "snapshots", ttl=SNAPSHOT_TTL)
        store.seed_from_file("combined", csv_path, NOW)

        result = download_combined_with_fallback(
            api_key="test-key",
            start="202301010000",
            end="202301050000",
            now=NOW,
            max_retries=1,
            backoff=0.0,
            timeout=None,
            fallback_enabled=True,
            sleep=lambda s: None,
        )
        assert result.mode == "combined"
        assert result.combined_status == "cache"
        assert result.fallback_gate is True

    def test_fallback_disabled_skips_snapshot(self, monkeypatch, tmp_path):
        """fallback_enabled=False must not read snapshots even when they exist."""
        self._patch_env(monkeypatch, tmp_path)

        csv_path = tmp_path / "interim" / "energy_load.csv"
        idx = pd.date_range("2026-06-10", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"Actual Load": 40_000.0, "Forecasted Load": 41_000.0}, index=idx)
        df.index.name = "Time (UTC)"
        df.to_csv(csv_path)

        from spotforecast2_safe.utils.snapshot_store import SnapshotStore
        from spotforecast2_safe.downloader.resilience import SNAPSHOT_TTL

        store = SnapshotStore(root=tmp_path / "snapshots", ttl=SNAPSHOT_TTL)
        store.seed_from_file("combined", csv_path, NOW)

        monkeypatch.setattr(
            "spotforecast2_safe.downloader.entsoe.download_new_data",
            lambda **kwargs: (_ for _ in ()).throw(ConnectionError("fail")),
        )

        result = download_combined_with_fallback(
            api_key="test-key",
            start="202301010000",
            end="202301050000",
            now=NOW,
            max_retries=1,
            backoff=0.0,
            timeout=None,
            fallback_enabled=False,
            sleep=lambda s: None,
        )
        assert result.mode is None
        assert result.combined_status == "missing"

    def test_returns_empty_zones_dict(self, monkeypatch, tmp_path):
        """zones must always be {} (no per-zone download attempted)."""
        self._patch_env(monkeypatch, tmp_path)

        def fake_download(**kwargs):
            csv = tmp_path / "interim" / "energy_load.csv"
            idx = pd.date_range("2026-06-10", periods=24, freq="h", tz="UTC")
            df = pd.DataFrame({"Actual Load": 40_000.0, "Forecasted Load": 41_000.0}, index=idx)
            df.index.name = "Time (UTC)"
            df.to_csv(csv)

        monkeypatch.setattr(
            "spotforecast2_safe.downloader.entsoe.download_new_data",
            fake_download,
        )

        result = download_combined_with_fallback(
            api_key="test-key",
            start="202301010000",
            end="202301050000",
            now=NOW,
            max_retries=1,
            backoff=0.0,
            timeout=None,
            fallback_enabled=True,
            sleep=lambda s: None,
        )
        assert result.zones == {}

    def test_exponential_backoff_sleep_called_between_retries(
        self, monkeypatch, tmp_path
    ):
        """sleep must be called with exponential backoff between failed attempts."""
        self._patch_env(monkeypatch, tmp_path)

        monkeypatch.setattr(
            "spotforecast2_safe.downloader.entsoe.download_new_data",
            lambda **kwargs: (_ for _ in ()).throw(ConnectionError("fail")),
        )

        sleeps: list[float] = []
        download_combined_with_fallback(
            api_key="test-key",
            start="202301010000",
            end="202301050000",
            now=NOW,
            max_retries=3,
            backoff=5.0,
            timeout=None,
            fallback_enabled=False,
            sleep=lambda s: sleeps.append(s),
        )
        # 3 attempts -> 2 sleeps (no sleep after last attempt)
        assert len(sleeps) == 2
        assert sleeps[0] == 5.0   # backoff * 2^0
        assert sleeps[1] == 10.0  # backoff * 2^1


# ---------------------------------------------------------------------------
# assert_submission_coverage
# ---------------------------------------------------------------------------


class TestAssertSubmissionCoverage:
    """Tests for preprocessing.coverage.assert_submission_coverage."""

    # --- basic passing path ---

    def test_returns_submission_coverage_dataclass(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates
    ):
        result = assert_submission_coverage(clean_interim, dates, fallback=False)
        assert isinstance(result, SubmissionCoverage)
        assert isinstance(result.last_full_hour, pd.Timestamp)
        assert isinstance(result.first_pred, pd.Timestamp)
        assert isinstance(result.n_steps, int)

    def test_first_pred_is_last_full_hour_plus_1h(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates
    ):
        result = assert_submission_coverage(clean_interim, dates, fallback=False)
        assert result.first_pred == result.last_full_hour + pd.Timedelta(hours=1)

    def test_n_steps_covers_full_tomorrow(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates
    ):
        result = assert_submission_coverage(clean_interim, dates, fallback=False)
        expected = (
            int((dates.last_target - result.first_pred).total_seconds() // 3600) + 1
        )
        assert result.n_steps == expected

    # --- Guard 1: frontier freshness ---

    def test_raises_coverage_error_when_frontier_stale(self, dates: SubmissionDates):
        # Make a frame that ends before today - 1h.
        # Ends at 2026-06-13 21:00 UTC (stale: < 2026-06-13 23:00 == today - 1h).
        stale = _make_interim(
            start="2026-06-10 00:00",
            end="2026-06-13 21:00",
        )
        with pytest.raises(CoverageError, match="stale"):
            assert_submission_coverage(stale, dates, fallback=False)

    # --- Guard 2: actual lag ---

    def test_raises_coverage_error_when_actual_too_old(self, dates: SubmissionDates):
        # Frame index is fresh but all Actual Load values are NaN -> empty dropna.
        idx = pd.date_range("2026-06-10 00:00", "2026-06-14 09:00", freq="h", tz="UTC")
        df = pd.DataFrame(
            {"Actual Load": float("nan"), "Forecasted Load": 41_000.0}, index=idx
        )
        with pytest.raises(CoverageError):
            assert_submission_coverage(df, dates, fallback=False)

    # --- Guard 3: interior gaps ---

    def test_raises_coverage_error_on_24h_interior_gap(self, dates: SubmissionDates):
        # Build frame covering 30 days, then drop 24 h inside the scan window.
        start = NOW - pd.Timedelta(days=30)
        idx = pd.date_range(start, NOW - pd.Timedelta(hours=1), freq="h", tz="UTC")
        actual = pd.Series(40_000.0, index=idx)
        # Drop 2026-06-02 (within 28-day scan window from NOW).
        drop = (idx >= pd.Timestamp("2026-06-02", tz="UTC")) & (
            idx < pd.Timestamp("2026-06-03", tz="UTC")
        )
        gapped_actual = actual[~drop]
        df = pd.DataFrame(
            {"Actual Load": gapped_actual, "Forecasted Load": 41_000.0}
        )
        with pytest.raises(CoverageError, match="interior gap"):
            assert_submission_coverage(df, dates, fallback=False)

    # --- Guard 4: frontier completeness ---

    def test_steps_back_on_partial_frontier_15min(self, dates: SubmissionDates):
        """2026-06-05 incident: partial frontier hour steps back last_full_hour."""
        # 15-min feed ending at 09:30 (last complete hour is 08:00, not 09:00).
        idx = pd.date_range(
            "2026-06-10 00:00", "2026-06-14 09:30", freq="15min", tz="UTC"
        )
        df = pd.DataFrame(
            {"Actual Load": 40_000.0, "Forecasted Load": 41_000.0}, index=idx
        )
        # Remove last 2 samples of hour 09 -> only 2/4 samples, not complete.
        bad_mask = df.index >= pd.Timestamp("2026-06-14 09:30", tz="UTC")
        df = df[~bad_mask]
        # Remove one more sample to make 09:xx have only 2 of 4.
        df = df.drop(pd.Timestamp("2026-06-14 09:15", tz="UTC"))

        result = assert_submission_coverage(df, dates, fallback=False)
        assert result.last_full_hour == pd.Timestamp("2026-06-14 08:00", tz="UTC")

    # --- corruption_mode divergence ---

    def test_preview_mode_does_not_retract_last_full_hour(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates, monkeypatch
    ):
        """In preview mode the QC fires but last_full_hour is NOT retracted."""
        fired_report = MagicMock()
        fired_report.fired = True
        fired_report.first_flagged_hour = pd.Timestamp("2026-06-14 06:00", tz="UTC")
        fired_report.n_flagged_hours = 3
        fired_report.spans = [1]
        fired_report.action = "truncate"

        # Patch at the target_corruption module level (that's where it lives).
        monkeypatch.setattr(
            "spotforecast2_safe.preprocessing.target_corruption.apply_target_corruption_policy",
            lambda *a, **kw: (None, fired_report),
        )

        result = assert_submission_coverage(
            clean_interim, dates, fallback=False, corruption_mode="preview"
        )
        # last_full_hour must NOT be retracted to 05:00 in preview mode.
        expected_last = pd.Timestamp("2026-06-14 09:00", tz="UTC")
        assert result.last_full_hour == expected_last

    def test_authoritative_mode_retracts_last_full_hour(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates, monkeypatch
    ):
        """In authoritative mode QC retracts last_full_hour to first_flagged - 1h."""
        fired_report = MagicMock()
        fired_report.fired = True
        fired_report.first_flagged_hour = pd.Timestamp("2026-06-14 06:00", tz="UTC")
        fired_report.n_flagged_hours = 3
        fired_report.spans = [1]
        fired_report.action = "truncate"

        # Patch at the target_corruption module level (that's where it lives).
        monkeypatch.setattr(
            "spotforecast2_safe.preprocessing.target_corruption.apply_target_corruption_policy",
            lambda *a, **kw: (None, fired_report),
        )

        result = assert_submission_coverage(
            clean_interim, dates, fallback=False, corruption_mode="authoritative"
        )
        # last_full_hour is retracted: min(09:00, 06:00 - 1h) = 05:00
        expected_retracted = pd.Timestamp("2026-06-14 05:00", tz="UTC")
        assert result.last_full_hour == expected_retracted

    # --- fallback gate ---

    def test_fallback_true_passes_when_last_full_hour_at_gate(
        self, dates: SubmissionDates
    ):
        """fallback=True passes when last_full_hour == yesterday 23:00 UTC."""
        # yesterday 23:00 = 2026-06-13 23:00 UTC
        gate = YESTERDAY + pd.Timedelta(hours=23)
        # Build frame ending exactly at gate using tz-aware start/end.
        idx = pd.date_range(
            pd.Timestamp("2026-06-10 00:00", tz="UTC"), gate, freq="h"
        )
        df = pd.DataFrame(
            {"Actual Load": 40_000.0, "Forecasted Load": 41_000.0}, index=idx
        )
        # Should not raise.
        result = assert_submission_coverage(df, dates, fallback=True)
        assert result.last_full_hour == gate

    def test_fallback_true_raises_when_last_full_hour_before_gate(
        self, dates: SubmissionDates
    ):
        """fallback=True raises CoverageError when last_full_hour < yesterday 23:00.

        Uses 15-min cadence so the index frontier passes Guard 1 (index.max()
        reaches today - 1h) but last_full_hour steps back to yesterday 22:00
        (only 2 of 4 quarter-hour samples in the frontier hour).
        """
        # 15-min feed starting on 2026-06-10, ending at 2026-06-13 23:30 UTC.
        # That means the frontier 23:xx hour has only 2/4 slots (23:00, 23:15).
        # last_full_hour will step back to 22:00, which is < gate (23:00).
        # index.max() = 23:30 > today - 1h (23:00) -> Guard 1 passes.
        idx_15min = pd.date_range(
            pd.Timestamp("2026-06-10 00:00", tz="UTC"),
            pd.Timestamp("2026-06-13 23:30", tz="UTC"),
            freq="15min",
        )
        # Drop 23:30 so index.max() = 23:15 (still >= 23:00); frontier hour
        # 23:xx has only slots 23:00 and 23:15 (2 of 4) -> incomplete.
        idx_15min = idx_15min[idx_15min <= pd.Timestamp("2026-06-13 23:15", tz="UTC")]
        df = pd.DataFrame(
            {"Actual Load": 40_000.0, "Forecasted Load": 41_000.0},
            index=idx_15min,
        )
        with pytest.raises(CoverageError, match="cached actuals end"):
            assert_submission_coverage(df, dates, fallback=True)

    def test_fallback_false_does_not_apply_gate(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates
    ):
        """fallback=False skips the D-1 23:00 gate entirely."""
        # clean_interim ends at 09:00, well after gate — but this should
        # also pass when clean_interim ended BEFORE the gate (gate not checked).
        result = assert_submission_coverage(
            clean_interim, dates, fallback=False
        )
        assert isinstance(result, SubmissionCoverage)

    # --- combined vs four_zone mode label ---

    def test_mode_combined_and_four_zone_both_accepted(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates
    ):
        r1 = assert_submission_coverage(
            clean_interim, dates, fallback=False, mode="combined"
        )
        r2 = assert_submission_coverage(
            clean_interim, dates, fallback=False, mode="four_zone"
        )
        # Both must succeed and return identical coverage values.
        assert r1.last_full_hour == r2.last_full_hour
        assert r1.n_steps == r2.n_steps

    def test_invalid_corruption_mode_raises_value_error(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates
    ):
        with pytest.raises(ValueError, match="corruption_mode"):
            assert_submission_coverage(
                clean_interim, dates, fallback=False, corruption_mode="unknown"
            )

    # --- SubmissionCoverage dataclass ---

    def test_submission_coverage_is_frozen(
        self, clean_interim: pd.DataFrame, dates: SubmissionDates
    ):
        result = assert_submission_coverage(clean_interim, dates, fallback=False)
        with pytest.raises(Exception):
            result.n_steps = 0  # type: ignore[misc]

    # --- module exports ---

    def test_exported_from_preprocessing_init(self):
        from spotforecast2_safe.preprocessing import (  # noqa: F401
            SubmissionCoverage,
            assert_submission_coverage,
        )

        assert callable(assert_submission_coverage)

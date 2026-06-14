# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2_safe.downloader.resilience.

Covers the download-resilience policy layer: the fallback decision tree,
per-zone snapshot fallback, combined-series fallback, and the D-1 gate flag.
Mocks monkeypatch ``spotforecast2_safe.downloader.entsoe.download_zone_loads``
(honoring ``on_zone_failure="collect"`` and returning ``ZoneResult`` dicts)
and ``download_new_data``.

All tests run without network access and without an ENTSOE_API_KEY.
Environment variables SPOTFORECAST2_DATA and SPOTFORECAST2_CACHE are set via
monkeypatch BEFORE any get_data_home() call so the sf2-safe lazy imports see
the tmp_path-based directories.

Shared anchor for all tests:
    now = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")

Tests NOT ported here (belong to the lecture-scripts suite):
    - test_assert_coverage_stale_cache_raises_abort2
    - test_assert_coverage_stale_frontier_raises_abort1
    - test_exit5_wiring_from_script
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from spotforecast2_safe.downloader import resilience as resil

NOW = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_zone_csv(path: Path, col: str, n_rows: int = 8) -> None:
    """Write a minimal 15-min cadence zone CSV to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2026-06-10 00:00", periods=n_rows, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {col: range(1000, 1000 + n_rows), f"{col}_forecast": range(900, 900 + n_rows)},
        index=idx,
    )
    df.index.name = "Time (UTC)"
    df.to_csv(path)


def _make_combined_csv(path: Path, n_rows: int = 8) -> None:
    """Write a minimal combined energy_load CSV to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2026-06-10 00:00", periods=n_rows, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "Actual Load": range(40000, 40000 + n_rows),
            "Forecasted Load": range(39000, 39000 + n_rows),
            "Actual": range(40000, 40000 + n_rows),
        },
        index=idx,
    )
    df.index.name = "Time (UTC)"
    df.to_csv(path)


def _write_snap(
    kind: str, data_home: Path, ts: pd.Timestamp, col: str | None = None
) -> Path:
    """Write a snapshot file directly via SnapshotStore for test control."""
    from spotforecast2_safe.utils.snapshot_store import SnapshotStore

    store = SnapshotStore(root=data_home / "snapshots", ttl=resil.SNAPSHOT_TTL)
    tmp_src = data_home / f"_tmp_snap_{kind}.csv"
    if col is not None:
        _make_zone_csv(tmp_src, col)
    else:
        _make_combined_csv(tmp_src)
    snapped = store.write(kind, tmp_src, ts)
    tmp_src.unlink(missing_ok=True)
    assert snapped is not None
    return snapped


def _patched_dl_zone(
    monkeypatch,
    data_home: Path,
    fail_zones: set[str] | None = None,
) -> None:
    """Monkeypatch download_zone_loads to return ZoneResult dicts (collect mode)."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod
    from spotforecast2_safe.downloader.entsoe import ZoneResult

    def _fake_download_zone_loads(
        api_key,
        zones=None,
        start=None,
        end=None,
        force=False,
        keep_forecast_future=False,
        timeout=60.0,
        on_zone_failure="raise",
    ):
        if on_zone_failure == "raise":
            if zones is None:
                raise RuntimeError("zones must be provided in fake")
            for col in zones:
                if fail_zones and col in fail_zones:
                    raise RuntimeError(f"simulated failure for zone {col}")
                path = data_home / "interim" / f"zone_{col}.csv"
                _make_zone_csv(path, col)
            return None
        else:
            # collect mode: return ZoneResult dict
            if zones is None:
                return {}
            results = {}
            for col in zones:
                if fail_zones and col in fail_zones:
                    results[col] = ZoneResult(
                        column=col,
                        area=str(zones.get(col, "")),
                        ok=False,
                        error=RuntimeError(f"simulated failure for {col}"),
                        interim_path=None,
                    )
                else:
                    path = data_home / "interim" / f"zone_{col}.csv"
                    _make_zone_csv(path, col)
                    results[col] = ZoneResult(
                        column=col,
                        area=str(zones.get(col, "")),
                        ok=True,
                        error=None,
                        interim_path=path,
                    )
            return results

    monkeypatch.setattr(entsoe_mod, "download_zone_loads", _fake_download_zone_loads)


def _patched_dl_combined(
    monkeypatch,
    data_home: Path,
    fail: bool = False,
) -> None:
    """Monkeypatch download_new_data to write combined CSV or raise RuntimeError."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    def _fake_download_new_data(
        api_key,
        country_code="DE",
        start=None,
        end=None,
        force=False,
        keep_forecast_future=False,
        timeout=60.0,
    ):
        if fail:
            raise RuntimeError("simulated combined download failure")
        path = data_home / "interim" / "energy_load.csv"
        _make_combined_csv(path)

    monkeypatch.setattr(entsoe_mod, "download_new_data", _fake_download_new_data)


def _setup_env(monkeypatch, tmp_path: Path) -> Path:
    """Set SPOTFORECAST2_DATA and _CACHE env vars; return data_home."""
    data_home = tmp_path / "data"
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("SPOTFORECAST2_DATA", str(data_home))
    monkeypatch.setenv("SPOTFORECAST2_CACHE", str(cache_home))
    return data_home


# ---------------------------------------------------------------------------
# Scenario 1: all four live
# ---------------------------------------------------------------------------


def test_all_four_live(monkeypatch, tmp_path):
    """All zones succeed live -> 4 snapshots written, mode four_zone, gate False."""
    data_home = _setup_env(monkeypatch, tmp_path)
    _patched_dl_zone(monkeypatch, data_home)
    _patched_dl_combined(monkeypatch, data_home)

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=True,
        sleep=lambda s: None,
    )

    assert result.mode == "four_zone"
    assert result.fallback_gate is False
    for col in resil.ZONE_COLUMNS:
        assert result.zones[col].status == "live"
    # Check that snapshots were written
    snap_root = data_home / "snapshots"
    for col in resil.ZONE_COLUMNS:
        snap_dir = snap_root / f"zone_{col}"
        snaps = list(snap_dir.glob("*.csv"))
        assert len(snaps) == 1, f"expected 1 snapshot for {col}, got {len(snaps)}"


# ---------------------------------------------------------------------------
# Scenario 2: tennet fails + valid snapshot
# ---------------------------------------------------------------------------


def test_tennet_fails_valid_snapshot(monkeypatch, tmp_path, caplog):
    """tennet live fails + valid snapshot -> status cache with age, gate True, WARNING logged."""
    data_home = _setup_env(monkeypatch, tmp_path)
    _patched_dl_zone(monkeypatch, data_home, fail_zones={"load_tennet"})
    _patched_dl_combined(monkeypatch, data_home)

    # Pre-write a valid snapshot for tennet (48 h old, within 72 h TTL)
    snap_ts = NOW - pd.Timedelta(hours=48)
    _write_snap("zone_load_tennet", data_home, snap_ts, col="load_tennet")

    with caplog.at_level(logging.WARNING, logger="spotforecast2_safe.downloader.resilience"):
        result = resil.download_with_fallback(
            "fake_key",
            start="202206100000",
            end="202206120000",
            now=NOW,
            max_retries=1,
            backoff=0.0,
            timeout=None,
            fallback_enabled=True,
            sleep=lambda s: None,
        )

    assert result.mode == "four_zone"
    assert result.zones["load_tennet"].status == "cache"
    assert result.zones["load_tennet"].age is not None
    # age should be ~48 h
    assert abs(result.zones["load_tennet"].age.total_seconds() - 48 * 3600) < 60
    for col in ("load_amprion", "load_transnetbw", "load_50hertz"):
        assert result.zones[col].status == "live"
    assert result.fallback_gate is True
    # Check that a WARNING was logged about the snapshot restore
    assert any(
        "load_tennet" in r.message and "snapshot" in r.message.lower()
        for r in caplog.records
    ), f"Expected WARNING about tennet snapshot in: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Scenario 3: two zones fail, both snapshots valid
# ---------------------------------------------------------------------------


def test_two_zones_fail_both_snapshots_valid(monkeypatch, tmp_path):
    """Two zones fail + valid snapshots -> both cache, mode four_zone."""
    data_home = _setup_env(monkeypatch, tmp_path)
    _patched_dl_zone(monkeypatch, data_home, fail_zones={"load_tennet", "load_50hertz"})
    _patched_dl_combined(monkeypatch, data_home)

    snap_ts = NOW - pd.Timedelta(hours=24)
    _write_snap("zone_load_tennet", data_home, snap_ts, col="load_tennet")
    _write_snap("zone_load_50hertz", data_home, snap_ts, col="load_50hertz")

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=True,
        sleep=lambda s: None,
    )

    assert result.mode == "four_zone"
    assert result.zones["load_tennet"].status == "cache"
    assert result.zones["load_50hertz"].status == "cache"
    assert result.zones["load_amprion"].status == "live"
    assert result.zones["load_transnetbw"].status == "live"


# ---------------------------------------------------------------------------
# Scenario 4: one zone fails, no snapshot, combined live
# ---------------------------------------------------------------------------


def test_one_zone_fails_no_snapshot_combined_live(monkeypatch, tmp_path):
    """One zone fails, no snapshot -> combined live -> mode combined, combined snapshot written."""
    data_home = _setup_env(monkeypatch, tmp_path)
    _patched_dl_zone(monkeypatch, data_home, fail_zones={"load_tennet"})
    _patched_dl_combined(monkeypatch, data_home)

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=True,
        sleep=lambda s: None,
    )

    assert result.mode == "combined"
    assert result.combined_status == "live"
    assert result.zones["load_tennet"].status == "missing"
    # Combined snapshot was written
    snap_dir = data_home / "snapshots" / "combined"
    assert any(snap_dir.glob("*.csv")), "expected combined snapshot"
    # Statuses of the zones that succeeded
    for col in ("load_amprion", "load_transnetbw", "load_50hertz"):
        assert result.zones[col].status == "live"


# ---------------------------------------------------------------------------
# Scenario 5: one zone fails, no snapshot, combined fails, combined snapshot valid
# ---------------------------------------------------------------------------


def test_one_zone_fails_combined_fails_combined_snapshot(monkeypatch, tmp_path):
    """Combined download also fails but a valid combined snapshot exists -> cache+age."""
    data_home = _setup_env(monkeypatch, tmp_path)
    _patched_dl_zone(monkeypatch, data_home, fail_zones={"load_tennet"})
    _patched_dl_combined(monkeypatch, data_home, fail=True)

    snap_ts = NOW - pd.Timedelta(hours=36)
    _write_snap("combined", data_home, snap_ts)

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=True,
        sleep=lambda s: None,
    )

    assert result.mode == "combined"
    assert result.combined_status == "cache"
    assert result.combined_age is not None
    assert abs(result.combined_age.total_seconds() - 36 * 3600) < 60


# ---------------------------------------------------------------------------
# Scenario 6: one zone fails, no snapshot, combined fails, no combined snapshot
# ---------------------------------------------------------------------------


def test_one_zone_fails_no_snapshot_combined_fails_no_snapshot(monkeypatch, tmp_path):
    """Fully unrecoverable -> mode None, report names missing zone and combined status."""
    data_home = _setup_env(monkeypatch, tmp_path)
    _patched_dl_zone(monkeypatch, data_home, fail_zones={"load_tennet"})
    _patched_dl_combined(monkeypatch, data_home, fail=True)

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=True,
        sleep=lambda s: None,
    )

    assert result.mode is None
    report = result.report()
    assert "load_tennet" in report
    assert "missing" in report.lower()
    assert "combined" in report.lower()


# ---------------------------------------------------------------------------
# Scenario 7: only a 96 h old snapshot exists
# ---------------------------------------------------------------------------


def test_expired_snapshot_treated_as_missing(monkeypatch, tmp_path):
    """96 h-old snapshot is beyond TTL; SnapshotStore.newest_valid returns None -> scenario-6."""
    data_home = _setup_env(monkeypatch, tmp_path)
    _patched_dl_zone(monkeypatch, data_home, fail_zones={"load_tennet"})
    _patched_dl_combined(monkeypatch, data_home, fail=True)

    # Write an expired snapshot (96 h old > 72 h TTL)
    expired_ts = NOW - pd.Timedelta(hours=96)
    _write_snap("zone_load_tennet", data_home, expired_ts, col="load_tennet")

    # SnapshotStore.newest_valid should ignore the expired snapshot
    snap = resil.make_store().newest_valid("zone_load_tennet", NOW)
    assert snap is None, "expired snapshot should not be returned"

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=True,
        sleep=lambda s: None,
    )

    assert result.mode is None


# ---------------------------------------------------------------------------
# Scenario: --no-fallback skips snapshot READs but still WRITEs on success
# ---------------------------------------------------------------------------


def test_no_fallback_skips_reads_still_writes(monkeypatch, tmp_path):
    """With fallback_enabled=False: snapshot not read on failure, but written on success."""
    data_home = _setup_env(monkeypatch, tmp_path)
    # tennet fails; has a valid snapshot; but fallback_enabled=False means it should NOT be used
    _patched_dl_zone(monkeypatch, data_home, fail_zones={"load_tennet"})
    _patched_dl_combined(monkeypatch, data_home, fail=True)

    snap_ts = NOW - pd.Timedelta(hours=12)
    _write_snap("zone_load_tennet", data_home, snap_ts, col="load_tennet")

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=False,
        sleep=lambda s: None,
    )

    # tennet should be "missing" even though a snapshot existed
    assert result.zones["load_tennet"].status == "missing"
    assert result.mode is None  # combined also failed and no fallback for combined

    # But the OTHER zones that succeeded should have had snapshots written
    for col in ("load_amprion", "load_transnetbw", "load_50hertz"):
        snap_dir = data_home / "snapshots" / f"zone_{col}"
        assert any(snap_dir.glob("*.csv")), f"expected snapshot for {col} to be written"

    # The failing zone (load_tennet) must NOT have had a NEW snapshot written.
    # Only the pre-seeded snapshot should be there.
    tennet_snap_dir = data_home / "snapshots" / "zone_load_tennet"
    tennet_snaps_after = (
        list(tennet_snap_dir.glob("*.csv")) if tennet_snap_dir.exists() else []
    )
    assert len(tennet_snaps_after) == 1, (
        f"expected exactly 1 pre-seeded snapshot for load_tennet; got {tennet_snaps_after}"
    )

    # No combined snapshot should exist (combined download also failed).
    combined_snap_dir = data_home / "snapshots" / "combined"
    combined_snaps = (
        list(combined_snap_dir.glob("*.csv")) if combined_snap_dir.exists() else []
    )
    assert len(combined_snaps) == 0, f"expected no combined snapshot; got {combined_snaps}"


# ---------------------------------------------------------------------------
# Unit tests: _fmt_age and DownloadResult.report
# ---------------------------------------------------------------------------


def test_fmt_age_none():
    """_fmt_age(None) returns '?'."""
    assert resil._fmt_age(None) == "?"


def test_fmt_age_exact_hours():
    """_fmt_age formats whole hours correctly."""
    assert resil._fmt_age(pd.Timedelta(hours=48)) == "48h00m"


def test_fmt_age_hours_and_minutes():
    """_fmt_age formats hours and minutes correctly."""
    assert resil._fmt_age(pd.Timedelta(hours=2, minutes=35)) == "2h35m"


def test_download_result_report_all_live():
    """report() covers live zones and n/a combined."""
    zones = {col: resil.ZoneOutcome(column=col, status="live") for col in resil.ZONE_COLUMNS}
    result = resil.DownloadResult(zones=zones, mode="four_zone")
    report = result.report()
    assert "live (fresh download)" in report
    assert "four_zone" in report
    assert "n/a" in report


def test_download_result_report_cache_and_missing():
    """report() shows age for cache zones and MISSING label for missing ones."""
    zones = {
        "load_amprion": resil.ZoneOutcome(
            column="load_amprion", status="cache", age=pd.Timedelta(hours=12)
        ),
        "load_tennet": resil.ZoneOutcome(column="load_tennet", status="missing"),
        "load_transnetbw": resil.ZoneOutcome(column="load_transnetbw", status="live"),
        "load_50hertz": resil.ZoneOutcome(column="load_50hertz", status="live"),
    }
    result = resil.DownloadResult(
        zones=zones, combined_status="missing", mode=None
    )
    report = result.report()
    assert "12h00m" in report
    assert "MISSING" in report
    assert "mode: None" in report


def test_download_result_report_combined_cache():
    """report() appends age for combined cache status."""
    zones = {col: resil.ZoneOutcome(column=col, status="live") for col in resil.ZONE_COLUMNS}
    result = resil.DownloadResult(
        zones=zones,
        combined_status="cache",
        combined_age=pd.Timedelta(hours=36),
        mode="combined",
    )
    report = result.report()
    assert "36h00m" in report
    assert "cache" in report.lower()


# ---------------------------------------------------------------------------
# Unit tests: make_store
# ---------------------------------------------------------------------------


def test_make_store_returns_snapshot_store(monkeypatch, tmp_path):
    """make_store() returns a SnapshotStore with the expected root."""
    from spotforecast2_safe.utils.snapshot_store import SnapshotStore

    data_home = _setup_env(monkeypatch, tmp_path)
    store = resil.make_store()
    assert isinstance(store, SnapshotStore)
    assert store.root == data_home / "snapshots"
    assert store.ttl == resil.SNAPSHOT_TTL


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_names_present():
    """All documented public names are accessible on the module."""
    assert hasattr(resil, "download_with_fallback")
    assert hasattr(resil, "DownloadResult")
    assert hasattr(resil, "ZoneOutcome")
    assert hasattr(resil, "make_store")
    assert hasattr(resil, "ZONE_COLUMNS")
    assert hasattr(resil, "SNAPSHOT_TTL")
    assert hasattr(resil, "_ZONE_AREAS")
    assert hasattr(resil, "_fmt_age")


def test_zone_columns_constant():
    """ZONE_COLUMNS lists exactly the four German TSO load columns."""
    assert resil.ZONE_COLUMNS == [
        "load_amprion",
        "load_tennet",
        "load_transnetbw",
        "load_50hertz",
    ]


def test_snapshot_ttl_is_72h():
    """SNAPSHOT_TTL is 72 hours."""
    assert resil.SNAPSHOT_TTL == pd.Timedelta(hours=72)


# ---------------------------------------------------------------------------
# Retry / backoff tests (max_retries > 1)
# ---------------------------------------------------------------------------


def test_retry_zone_succeeds_on_second_attempt(monkeypatch, tmp_path):
    """Zone fails attempt 1 then succeeds attempt 2 -> status live, sleep called once."""
    data_home = _setup_env(monkeypatch, tmp_path)

    import spotforecast2_safe.downloader.entsoe as entsoe_mod
    from spotforecast2_safe.downloader.entsoe import ZoneResult

    call_count = {"n": 0}

    def _fake_dl_zone(
        api_key,
        zones=None,
        start=None,
        end=None,
        force=False,
        keep_forecast_future=False,
        timeout=60.0,
        on_zone_failure="raise",
    ):
        call_count["n"] += 1
        results = {}
        for col in zones:
            if col == "load_tennet" and call_count["n"] == 1:
                # First call: tennet fails
                results[col] = ZoneResult(
                    column=col,
                    area=str(zones.get(col, "")),
                    ok=False,
                    error=RuntimeError("transient failure"),
                    interim_path=None,
                )
            else:
                # All others succeed; tennet succeeds on call 2+
                path = data_home / "interim" / f"zone_{col}.csv"
                _make_zone_csv(path, col)
                results[col] = ZoneResult(
                    column=col,
                    area=str(zones.get(col, "")),
                    ok=True,
                    error=None,
                    interim_path=path,
                )
        return results

    monkeypatch.setattr(entsoe_mod, "download_zone_loads", _fake_dl_zone)
    _patched_dl_combined(monkeypatch, data_home)

    sleep_calls: list[float] = []

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=2,
        backoff=5.0,
        timeout=None,
        fallback_enabled=True,
        sleep=sleep_calls.append,
    )

    assert result.mode == "four_zone"
    assert result.zones["load_tennet"].status == "live", (
        f"expected 'live' after retry, got {result.zones['load_tennet'].status!r}"
    )
    # sleep should have been called exactly once (between attempt 1 and attempt 2)
    assert len(sleep_calls) == 1, f"expected 1 sleep call, got {sleep_calls}"
    # backoff * 2**(attempt-1) = 5.0 * 2**0 = 5.0
    assert sleep_calls[0] == pytest.approx(5.0), (
        f"expected sleep(5.0), got sleep({sleep_calls[0]})"
    )


def test_retry_only_retries_failed_zones(monkeypatch, tmp_path):
    """On the second attempt only still-failing zones are passed to download_zone_loads."""
    data_home = _setup_env(monkeypatch, tmp_path)

    import spotforecast2_safe.downloader.entsoe as entsoe_mod
    from spotforecast2_safe.downloader.entsoe import ZoneResult

    # Track which zone sets were requested each call
    requested_zones_per_call: list[set[str]] = []

    def _fake_dl_zone(
        api_key,
        zones=None,
        start=None,
        end=None,
        force=False,
        keep_forecast_future=False,
        timeout=60.0,
        on_zone_failure="raise",
    ):
        requested_zones_per_call.append(set(zones.keys()))
        results = {}
        for col in zones:
            if col == "load_tennet":
                # tennet fails on every call (to force snapshot fallback)
                results[col] = ZoneResult(
                    column=col,
                    area=str(zones.get(col, "")),
                    ok=False,
                    error=RuntimeError("persistent failure"),
                    interim_path=None,
                )
            else:
                path = data_home / "interim" / f"zone_{col}.csv"
                _make_zone_csv(path, col)
                results[col] = ZoneResult(
                    column=col,
                    area=str(zones.get(col, "")),
                    ok=True,
                    error=None,
                    interim_path=path,
                )
        return results

    monkeypatch.setattr(entsoe_mod, "download_zone_loads", _fake_dl_zone)
    _patched_dl_combined(monkeypatch, data_home)

    # Pre-seed a valid snapshot for tennet so the result is four_zone (not combined)
    snap_ts = NOW - pd.Timedelta(hours=12)
    _write_snap("zone_load_tennet", data_home, snap_ts, col="load_tennet")

    result = resil.download_with_fallback(
        "fake_key",
        start="202206100000",
        end="202206120000",
        now=NOW,
        max_retries=2,
        backoff=1.0,
        timeout=None,
        fallback_enabled=True,
        sleep=lambda s: None,
    )

    # Both calls should have been made (tennet always fails)
    assert len(requested_zones_per_call) == 2, (
        f"expected 2 download_zone_loads calls, got {len(requested_zones_per_call)}"
    )
    # First call: all four zones
    assert requested_zones_per_call[0] == set(resil.ZONE_COLUMNS), (
        f"first call should request all zones, got {requested_zones_per_call[0]}"
    )
    # Second call: only tennet (the only one that failed on attempt 1)
    assert requested_zones_per_call[1] == {"load_tennet"}, (
        f"second call should request only load_tennet, got {requested_zones_per_call[1]}"
    )
    # tennet ends as cache (snapshot restored), others live
    assert result.zones["load_tennet"].status == "cache"
    assert result.mode == "four_zone"

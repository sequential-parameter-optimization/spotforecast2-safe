# SPDX-FileCopyrightText: 2024-2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2_safe.utils.snapshot_store.

Ported and adapted from bart26k-lecture/scripts/tests/test_team4_resilience.py
(store-level mechanics only; all domain knowledge stripped).

All tests use ``tmp_path`` and a fixed ``NOW`` anchor — no wall-clock dependence.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from spotforecast2_safe.utils.snapshot_store import (
    SnapshotStore,
    parse_snapshot_timestamp,
)

NOW = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")
TTL = pd.Timedelta(hours=72)
_TS_FMT = "%Y%m%dT%H%M%SZ"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_csv(path: Path, content: str = "a,b\n1,2\n3,4\n") -> Path:
    """Write a small CSV to ``path``, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _store(root: Path) -> SnapshotStore:
    return SnapshotStore(root=root, ttl=TTL)


def _write_snap_direct(
    store: SnapshotStore, kind: str, ts: pd.Timestamp, content: str = "x\n1\n"
) -> Path:
    """Write a snapshot file directly into the store directory (bypasses write())."""
    kind_dir = store.root / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    filename = ts.strftime(_TS_FMT) + ".csv"
    dest = kind_dir / filename
    dest.write_text(content)
    return dest


# ---------------------------------------------------------------------------
# SnapshotStore.__post_init__ validation
# ---------------------------------------------------------------------------


def test_snapshot_store_rejects_non_positive_ttl(tmp_path):
    with pytest.raises(ValueError, match="ttl must be positive"):
        SnapshotStore(root=tmp_path, ttl=pd.Timedelta(0))


def test_snapshot_store_rejects_relative_root():
    with pytest.raises(ValueError, match="root must be an absolute path"):
        SnapshotStore(root=Path("relative/path"), ttl=TTL)


# ---------------------------------------------------------------------------
# parse_snapshot_timestamp
# ---------------------------------------------------------------------------


def test_parse_snapshot_timestamp_valid():
    ts = parse_snapshot_timestamp(Path("20260611T040000Z.csv"))
    assert ts == pd.Timestamp("2026-06-11 04:00:00", tz="UTC")


def test_parse_snapshot_timestamp_invalid_warns(caplog):
    with caplog.at_level(logging.WARNING):
        result = parse_snapshot_timestamp(Path("not_a_timestamp.csv"))
    assert result is None
    assert any("not_a_timestamp.csv" in r.message for r in caplog.records)


def test_parse_snapshot_timestamp_no_extension():
    # stem without .csv extension — parse uses stem directly
    ts = parse_snapshot_timestamp(Path("20260611T040000Z"))
    assert ts == pd.Timestamp("2026-06-11 04:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# write — basic correctness
# ---------------------------------------------------------------------------


def test_write_creates_snapshot(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")
    snap = store.write("demo", src, NOW)

    assert snap is not None
    assert snap.exists()
    assert snap.name == NOW.strftime(_TS_FMT) + ".csv"
    assert snap.read_text() == src.read_text()


def test_write_returns_none_when_source_missing(tmp_path, caplog):
    store = _store(tmp_path / "snaps")
    with caplog.at_level(logging.WARNING):
        result = store.write("demo", tmp_path / "nonexistent.csv", NOW)
    assert result is None
    assert any("nonexistent" in r.message for r in caplog.records)


def test_write_creates_kind_dir_lazily(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")
    kind_dir = store.root / "newkind"
    assert not kind_dir.exists()
    store.write("newkind", src, NOW)
    assert kind_dir.is_dir()


# ---------------------------------------------------------------------------
# write — atomicity
# ---------------------------------------------------------------------------


def test_write_atomicity_no_tmp_after_success(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")
    store.write("demo", src, NOW)

    kind_dir = store.root / "demo"
    tmp_files = list(kind_dir.glob("*.tmp"))
    assert len(tmp_files) == 0, f"unexpected .tmp files: {tmp_files}"


def test_write_atomicity_simulated_crash(monkeypatch, tmp_path):
    """Simulated crash between tmp-write and os.replace: previous snapshot intact,
    exactly one .tmp left, newest_valid ignores the .tmp and returns the old snap."""
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")

    # First successful write
    ts1 = NOW - pd.Timedelta(hours=2)
    snap1 = store.write("demo", src, ts1)
    assert snap1 is not None and snap1.exists()

    # Simulate crash during second write
    original_replace = os.replace
    crashed = {"did_crash": False}

    def _crashing_replace(src_path, dst_path):
        if not crashed["did_crash"]:
            crashed["did_crash"] = True
            raise OSError("simulated crash during replace")
        return original_replace(src_path, dst_path)

    monkeypatch.setattr("os.replace", _crashing_replace)

    ts2 = NOW
    with pytest.raises(OSError):
        store.write("demo", src, ts2)

    monkeypatch.setattr("os.replace", original_replace)

    # Previous snapshot still intact
    assert snap1.exists(), "previous snapshot should survive a crash"

    # Exactly one .tmp file present
    kind_dir = store.root / "demo"
    tmp_files = list(kind_dir.glob("*.tmp"))
    assert len(tmp_files) == 1, f"expected 1 .tmp after crash; got {tmp_files}"

    # newest_valid must ignore .tmp and return snap1
    recovered = store.newest_valid("demo", NOW)
    assert recovered == snap1, f"expected snap1; got {recovered}"


# ---------------------------------------------------------------------------
# newest_valid — TTL
# ---------------------------------------------------------------------------


def test_newest_valid_returns_none_when_no_dir(tmp_path):
    store = _store(tmp_path / "snaps")
    assert store.newest_valid("nonexistent", NOW) is None


def test_newest_valid_within_ttl(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")
    ts = NOW - pd.Timedelta(hours=24)
    snap = store.write("demo", src, ts)

    result = store.newest_valid("demo", NOW)
    assert result == snap


def test_newest_valid_expired_snapshot_invisible(tmp_path):
    store = _store(tmp_path / "snaps")
    # 96 h > 72 h TTL
    old_ts = NOW - pd.Timedelta(hours=96)
    _write_snap_direct(store, "demo", old_ts)

    result = store.newest_valid("demo", NOW)
    assert result is None


def test_newest_valid_picks_most_recent(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")
    ts_old = NOW - pd.Timedelta(hours=48)
    ts_new = NOW - pd.Timedelta(hours=12)
    store.write("demo", src, ts_old)
    snap_new = store.write("demo", src, ts_new)

    result = store.newest_valid("demo", NOW)
    assert result == snap_new


def test_newest_valid_ignores_tmp_files(tmp_path):
    store = _store(tmp_path / "snaps")
    # Drop a .tmp file in the kind dir
    kind_dir = store.root / "demo"
    kind_dir.mkdir(parents=True)
    (kind_dir / "20260611T040000Z.csv.tmp").write_text("partial")

    result = store.newest_valid("demo", NOW)
    assert result is None


def test_newest_valid_skips_unparseable_with_warning(tmp_path, caplog):
    store = _store(tmp_path / "snaps")
    kind_dir = store.root / "demo"
    kind_dir.mkdir(parents=True)
    (kind_dir / "garbage_name.csv").write_text("x")

    with caplog.at_level(logging.WARNING):
        result = store.newest_valid("demo", NOW)

    assert result is None
    assert any("garbage_name.csv" in r.message for r in caplog.records)


def test_newest_valid_exact_boundary_is_valid(tmp_path):
    """A snapshot stamped exactly NOW - TTL is still valid (strict < cutoff)."""
    store = _store(tmp_path / "snaps")
    exact_ts = NOW - TTL  # age == TTL exactly
    snap = _write_snap_direct(store, "demo", exact_ts)

    result = store.newest_valid("demo", NOW)
    assert result == snap, "snapshot at exact TTL boundary must be returned as valid"


# ---------------------------------------------------------------------------
# restore — round-trip
# ---------------------------------------------------------------------------


def test_restore_round_trip(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv", content="col1,col2\n10,20\n30,40\n")
    snap = store.write("demo", src, NOW)

    dest = tmp_path / "out" / "restored.csv"
    store.restore(snap, dest)

    assert dest.exists()
    assert dest.read_bytes() == snap.read_bytes()


def test_restore_creates_parent_dirs(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")
    snap = store.write("demo", src, NOW)

    dest = tmp_path / "deep" / "nested" / "dir" / "out.csv"
    assert not dest.parent.exists()
    store.restore(snap, dest)
    assert dest.exists()


def test_restore_is_atomic(monkeypatch, tmp_path):
    """Restore writes .tmp then replaces; dest is not left half-written on crash."""
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")
    snap = store.write("demo", src, NOW)

    dest = tmp_path / "out.csv"
    dest.write_text("ORIGINAL")

    original_replace = os.replace
    crashed = {"did_crash": False}

    def _crashing_replace(s, d):
        if not crashed["did_crash"]:
            crashed["did_crash"] = True
            raise OSError("simulated crash")
        return original_replace(s, d)

    monkeypatch.setattr("os.replace", _crashing_replace)

    with pytest.raises(OSError):
        store.restore(snap, dest)

    monkeypatch.setattr("os.replace", original_replace)

    # dest should still contain original content
    assert dest.read_text() == "ORIGINAL"


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def test_prune_deletes_expired_keeps_valid(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")

    # Valid snapshot (24 h old)
    ts_valid = NOW - pd.Timedelta(hours=24)
    snap_valid = store.write("demo", src, ts_valid)

    # Expired snapshot (96 h old)
    ts_expired = NOW - pd.Timedelta(hours=96)
    snap_expired = _write_snap_direct(store, "demo", ts_expired)

    count = store.prune(NOW)

    assert count == 1
    assert not snap_expired.exists()
    assert snap_valid.exists()


def test_prune_deletes_stale_tmp_files(tmp_path):
    store = _store(tmp_path / "snaps")
    kind_dir = store.root / "demo"
    kind_dir.mkdir(parents=True)
    tmp_file = kind_dir / "20260611T040000Z.csv.tmp"
    tmp_file.write_text("partial")

    count = store.prune(NOW)
    assert count == 1
    assert not tmp_file.exists()


def test_prune_exact_boundary_keeps_snapshot(tmp_path):
    """A snapshot stamped exactly NOW - TTL must NOT be pruned (strict < cutoff)."""
    store = _store(tmp_path / "snaps")
    exact_ts = NOW - TTL  # age == TTL exactly
    snap = _write_snap_direct(store, "demo", exact_ts)

    count = store.prune(NOW)
    assert count == 0, "snapshot at exact TTL boundary must be kept"
    assert snap.exists(), "snapshot file must still be present after prune"


def test_prune_returns_zero_when_nothing_expired(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")
    ts = NOW - pd.Timedelta(hours=10)
    store.write("demo", src, ts)

    count = store.prune(NOW)
    assert count == 0


def test_prune_returns_zero_on_missing_root(tmp_path):
    store = _store(tmp_path / "nonexistent_root")
    assert store.prune(NOW) == 0


def test_prune_crosses_multiple_kinds(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")

    expired_ts = NOW - pd.Timedelta(hours=96)
    _write_snap_direct(store, "alpha", expired_ts)
    _write_snap_direct(store, "beta", expired_ts)
    store.write("gamma", src, NOW - pd.Timedelta(hours=10))  # valid

    count = store.prune(NOW)
    assert count == 2


# ---------------------------------------------------------------------------
# seed_from_file
# ---------------------------------------------------------------------------


def test_seed_from_file_uses_mtime(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")

    # Set mtime to 10 h ago (within TTL)
    mtime_target = (NOW - pd.Timedelta(hours=10)).timestamp()
    os.utime(str(src), (mtime_target, mtime_target))

    snap = store.seed_from_file("demo", src, NOW)

    assert snap is not None
    assert snap.exists()
    # Filename should encode the mtime, not NOW
    mtime_ts = pd.Timestamp(mtime_target, unit="s", tz="UTC")
    expected_name = mtime_ts.strftime(_TS_FMT) + ".csv"
    assert snap.name == expected_name


def test_seed_from_file_skips_when_mtime_older_than_ttl(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")

    # Back-date mtime to 80 h ago (> 72 h TTL)
    mtime_target = (NOW - pd.Timedelta(hours=80)).timestamp()
    os.utime(str(src), (mtime_target, mtime_target))

    snap = store.seed_from_file("demo", src, NOW)

    assert snap is None
    kind_dir = store.root / "demo"
    if kind_dir.exists():
        assert not any(kind_dir.glob("*.csv"))


def test_seed_from_file_exact_boundary_mtime_is_accepted(tmp_path):
    """A source file whose mtime is exactly NOW - TTL must be accepted (strict <)."""
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")

    # Set mtime to exactly NOW - TTL (boundary, should still be within TTL)
    exact_mtime = (NOW - TTL).timestamp()
    os.utime(str(src), (exact_mtime, exact_mtime))

    snap = store.seed_from_file("demo", src, NOW)
    assert snap is not None, "source file at exact TTL boundary mtime must be seeded"
    assert snap.exists()


def test_seed_from_file_skips_when_newer_snapshot_exists(tmp_path):
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")

    # File mtime: 10 h ago
    mtime = NOW - pd.Timedelta(hours=10)
    os.utime(str(src), (mtime.timestamp(), mtime.timestamp()))

    # Existing snapshot 1 min newer than the file's mtime
    newer_ts = mtime + pd.Timedelta(minutes=1)
    _write_snap_direct(store, "demo", newer_ts)

    snap = store.seed_from_file("demo", src, NOW)
    assert snap is None

    # Still only 1 snapshot in the dir
    kind_dir = store.root / "demo"
    assert len(list(kind_dir.glob("*.csv"))) == 1


def test_seed_from_file_returns_none_when_source_missing(tmp_path):
    store = _store(tmp_path / "snaps")
    snap = store.seed_from_file("demo", tmp_path / "missing.csv", NOW)
    assert snap is None


def test_seed_from_file_does_not_duplicate_with_equal_ts(tmp_path):
    """If a snapshot with timestamp == file mtime exists, no new snapshot is written."""
    store = _store(tmp_path / "snaps")
    src = _make_csv(tmp_path / "src.csv")

    mtime = NOW - pd.Timedelta(hours=5)
    os.utime(str(src), (mtime.timestamp(), mtime.timestamp()))

    # Write a snapshot with exactly the same timestamp as the file mtime
    _write_snap_direct(store, "demo", mtime)

    snap = store.seed_from_file("demo", src, NOW)
    assert snap is None

    kind_dir = store.root / "demo"
    assert len(list(kind_dir.glob("*.csv"))) == 1


# ---------------------------------------------------------------------------
# age_of
# ---------------------------------------------------------------------------


def test_age_of_correctness(tmp_path):
    store = _store(tmp_path / "snaps")
    snap = Path("20260610T040000Z.csv")  # 24 h before NOW
    age = store.age_of(snap, NOW)
    assert age == pd.Timedelta(hours=24)


def test_age_of_returns_none_on_unparseable(tmp_path, caplog):
    store = _store(tmp_path / "snaps")
    with caplog.at_level(logging.WARNING):
        age = store.age_of(Path("bad_name.csv"), NOW)
    assert age is None


def test_age_of_zero_for_same_timestamp(tmp_path):
    store = _store(tmp_path / "snaps")
    snap = Path(NOW.strftime(_TS_FMT) + ".csv")
    age = store.age_of(snap, NOW)
    assert age == pd.Timedelta(0)

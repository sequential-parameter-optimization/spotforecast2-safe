# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2_safe.submission_cli.

Covers:
- each mapper's success passthrough
- each failure -> correct AbortReason
- COVERAGE vs STALE_CACHE split (by the "yesterday 23:00 UTC" substring)
- corruption_mode is forwarded verbatim
- entsoe_predictions tolerant path (ValueError -> empty Series; KeyError propagates)

All tests run without network access and without ENTSOE_API_KEY.
Env vars SPOTFORECAST2_DATA and SPOTFORECAST2_CACHE are set via monkeypatch
so that lazy sf2-safe imports see tmp_path-based directories.

Monkeypatch style mirrors tests/downloader/test_resilience.py:
- monkeypatch the module attribute on the lazily-imported module object
  (e.g. ``monkeypatch.setattr(entsoe_mod, "compute_submission_dates", ...)``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

from spotforecast2_safe.submission_cli import (
    AbortReason,
    SubmissionAbort,
    assert_coverage_or_abort,
    compute_dates_or_abort,
    download_combined_or_abort,
    entsoe_predictions,
    load_interim_or_abort,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

NOW = pd.Timestamp("2026-06-23 04:00:00", tz="UTC")
_LOG = logging.getLogger("test_submission_cli")


def _setup_env(monkeypatch, tmp_path: Path) -> Path:
    """Set SPOTFORECAST2_DATA and _CACHE; return data_home."""
    data_home = tmp_path / "data"
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("SPOTFORECAST2_DATA", str(data_home))
    monkeypatch.setenv("SPOTFORECAST2_CACHE", str(cache_home))
    return data_home


class _FakeDates(NamedTuple):
    now: pd.Timestamp
    today: pd.Timestamp
    yesterday: pd.Timestamp
    tomorrow: pd.Timestamp
    last_target: pd.Timestamp
    start_dl: str
    end_dl: str


def _fake_dates() -> _FakeDates:
    return _FakeDates(
        now=NOW,
        today=NOW.floor("D"),
        yesterday=NOW.floor("D") - pd.Timedelta(days=1),
        tomorrow=NOW.floor("D") + pd.Timedelta(days=1),
        last_target=NOW.floor("D") + pd.Timedelta(hours=47),
        start_dl="202301010000",
        end_dl="202601010000",
    )


def _minimal_interim() -> pd.DataFrame:
    idx = pd.date_range("2026-06-10", periods=48, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "Actual Load": [40_000.0] * 48,
            "Forecasted Load": [39_000.0] * 48,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# AbortReason and SubmissionAbort
# ---------------------------------------------------------------------------


def test_abort_reason_members():
    """AbortReason has exactly the five mandated members."""
    expected = {
        "DATE",
        "INTERIM_MISSING",
        "COVERAGE",
        "STALE_CACHE",
        "DOWNLOAD_UNRECOVERABLE",
    }
    assert set(AbortReason.__members__) == expected


def test_submission_abort_carries_reason_and_message():
    """SubmissionAbort exposes .reason and str() is the message."""
    exc = SubmissionAbort(AbortReason.DATE, "bad date")
    assert exc.reason is AbortReason.DATE
    assert str(exc) == "bad date"


def test_submission_abort_is_exception():
    """SubmissionAbort inherits from Exception (no integer exit code)."""
    assert issubclass(SubmissionAbort, Exception)
    # Must NOT have a .code attribute
    exc = SubmissionAbort(AbortReason.DATE, "x")
    assert not hasattr(exc, "code")


# ---------------------------------------------------------------------------
# compute_dates_or_abort
# ---------------------------------------------------------------------------


def test_compute_dates_success(monkeypatch):
    """Success path: returns whatever compute_submission_dates returns."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    expected = _fake_dates()
    monkeypatch.setattr(
        entsoe_mod, "compute_submission_dates", lambda *a, **kw: expected
    )

    result = compute_dates_or_abort(
        NOW, None, train_years=3, start_margin_days=45, logger=_LOG
    )
    assert result is expected


def test_compute_dates_value_error_raises_abort_date(monkeypatch):
    """ValueError from compute_submission_dates -> AbortReason.DATE."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    def _raise(*a, **kw):
        raise ValueError("bad date arithmetic")

    monkeypatch.setattr(entsoe_mod, "compute_submission_dates", _raise)

    with pytest.raises(SubmissionAbort) as exc_info:
        compute_dates_or_abort(
            NOW, None, train_years=3, start_margin_days=45, logger=_LOG
        )
    assert exc_info.value.reason is AbortReason.DATE
    assert "bad date arithmetic" in str(exc_info.value)


def test_compute_dates_type_error_raises_abort_date(monkeypatch):
    """TypeError from compute_submission_dates -> AbortReason.DATE."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    def _raise(*a, **kw):
        raise TypeError("none not allowed")

    monkeypatch.setattr(entsoe_mod, "compute_submission_dates", _raise)

    with pytest.raises(SubmissionAbort) as exc_info:
        compute_dates_or_abort(
            NOW, None, train_years=3, start_margin_days=45, logger=_LOG
        )
    assert exc_info.value.reason is AbortReason.DATE


# ---------------------------------------------------------------------------
# load_interim_or_abort
# ---------------------------------------------------------------------------


def test_load_interim_success(monkeypatch):
    """Success path: returns the DataFrame and logs the INFO line."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    df = _minimal_interim()
    monkeypatch.setattr(entsoe_mod, "load_interim", lambda mode: df)

    records: list = []

    class _Cap(logging.Handler):
        def emit(self, r):
            records.append(r)

    cap = _Cap()
    log = logging.getLogger("_test_li")
    log.addHandler(cap)
    log.setLevel(logging.DEBUG)

    result = load_interim_or_abort(log)
    assert result is df
    assert any("interim" in r.getMessage() for r in records)


def test_load_interim_file_not_found_raises_abort(monkeypatch):
    """FileNotFoundError -> AbortReason.INTERIM_MISSING."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    def _raise(mode):
        raise FileNotFoundError("energy_load.csv not found")

    monkeypatch.setattr(entsoe_mod, "load_interim", _raise)

    with pytest.raises(SubmissionAbort) as exc_info:
        load_interim_or_abort(_LOG)
    assert exc_info.value.reason is AbortReason.INTERIM_MISSING
    assert "energy_load.csv" in str(exc_info.value)


# ---------------------------------------------------------------------------
# assert_coverage_or_abort — success
# ---------------------------------------------------------------------------

_COV_KWARGS = dict(
    fallback=False,
    corruption_mode="preview",
    deviation_ref="Forecasted Load",
    max_actual_lag_hours=36,
    gap_scan_days=28,
    max_actual_gap_hours=12,
    corruption_policy="truncate",
    range_mw=8_000,
    step_mw=6_000,
    qc_window_days=3,
    deviation_mw=11_000,
    deviation_slots=1,
    logger=_LOG,
)


def test_assert_coverage_success(monkeypatch):
    """Success path: returns whatever assert_submission_coverage returns."""
    import spotforecast2_safe.preprocessing.coverage as cov_mod

    fake_cov = MagicMock(n_steps=24)
    monkeypatch.setattr(
        cov_mod, "assert_submission_coverage", lambda *a, **kw: fake_cov
    )

    result = assert_coverage_or_abort(_minimal_interim(), _fake_dates(), **_COV_KWARGS)
    assert result is fake_cov


def test_assert_coverage_forwards_corruption_mode(monkeypatch):
    """corruption_mode is forwarded verbatim to assert_submission_coverage."""
    import spotforecast2_safe.preprocessing.coverage as cov_mod

    received: list[str] = []

    def _spy(*a, corruption_mode, **kw):
        received.append(corruption_mode)
        return MagicMock(n_steps=24)

    monkeypatch.setattr(cov_mod, "assert_submission_coverage", _spy)

    assert_coverage_or_abort(
        _minimal_interim(),
        _fake_dates(),
        **{**_COV_KWARGS, "corruption_mode": "authoritative"},
    )
    assert received == ["authoritative"]


def test_assert_coverage_forwards_preview_mode(monkeypatch):
    """corruption_mode='preview' is forwarded for team4 path."""
    import spotforecast2_safe.preprocessing.coverage as cov_mod

    received: list[str] = []

    def _spy(*a, corruption_mode, **kw):
        received.append(corruption_mode)
        return MagicMock(n_steps=24)

    monkeypatch.setattr(cov_mod, "assert_submission_coverage", _spy)

    assert_coverage_or_abort(
        _minimal_interim(),
        _fake_dates(),
        **{**_COV_KWARGS, "corruption_mode": "preview"},
    )
    assert received == ["preview"]


# ---------------------------------------------------------------------------
# assert_coverage_or_abort — failure split
# ---------------------------------------------------------------------------


def test_assert_coverage_stale_cache_raises_stale_cache(monkeypatch):
    """CoverageError with 'yesterday 23:00 UTC' -> AbortReason.STALE_CACHE."""
    import spotforecast2_safe.preprocessing.coverage as cov_mod
    from spotforecast2_safe.exceptions import CoverageError

    def _raise(*a, **kw):
        raise CoverageError("data older than yesterday 23:00 UTC")

    monkeypatch.setattr(cov_mod, "assert_submission_coverage", _raise)

    with pytest.raises(SubmissionAbort) as exc_info:
        assert_coverage_or_abort(_minimal_interim(), _fake_dates(), **_COV_KWARGS)
    assert exc_info.value.reason is AbortReason.STALE_CACHE


def test_assert_coverage_gap_raises_coverage(monkeypatch):
    """CoverageError without the stale-cache substring -> AbortReason.COVERAGE."""
    import spotforecast2_safe.preprocessing.coverage as cov_mod
    from spotforecast2_safe.exceptions import CoverageError

    def _raise(*a, **kw):
        raise CoverageError("interior gap of 15 hours detected")

    monkeypatch.setattr(cov_mod, "assert_submission_coverage", _raise)

    with pytest.raises(SubmissionAbort) as exc_info:
        assert_coverage_or_abort(_minimal_interim(), _fake_dates(), **_COV_KWARGS)
    assert exc_info.value.reason is AbortReason.COVERAGE


def test_assert_coverage_staleness_raises_coverage(monkeypatch):
    """CoverageError about lag (not the exact fallback-gate phrase) -> COVERAGE."""
    import spotforecast2_safe.preprocessing.coverage as cov_mod
    from spotforecast2_safe.exceptions import CoverageError

    def _raise(*a, **kw):
        raise CoverageError("actual load lag exceeds 36 h")

    monkeypatch.setattr(cov_mod, "assert_submission_coverage", _raise)

    with pytest.raises(SubmissionAbort) as exc_info:
        assert_coverage_or_abort(_minimal_interim(), _fake_dates(), **_COV_KWARGS)
    assert exc_info.value.reason is AbortReason.COVERAGE


# ---------------------------------------------------------------------------
# entsoe_predictions — tolerant path
# ---------------------------------------------------------------------------


def test_entsoe_predictions_success(monkeypatch):
    """Success path: returns the Series from extract_entsoe_hourly_forecast."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    series = pd.Series([1.0, 2.0], name="Forecasted Load")
    monkeypatch.setattr(entsoe_mod, "extract_entsoe_hourly_forecast", lambda df: series)

    result = entsoe_predictions(_minimal_interim(), _LOG)
    assert result is series


def test_entsoe_predictions_value_error_returns_empty(monkeypatch, caplog):
    """ValueError (all-NaN column) -> empty Series + WARNING logged."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    def _raise(df):
        raise ValueError("all-NaN column")

    monkeypatch.setattr(entsoe_mod, "extract_entsoe_hourly_forecast", _raise)

    with caplog.at_level(logging.WARNING):
        result = entsoe_predictions(_minimal_interim(), _LOG)

    assert len(result) == 0
    assert result.name == "Forecasted Load"
    assert result.dtype == "float64"
    assert any("all-NaN" in r.message for r in caplog.records)


def test_entsoe_predictions_key_error_propagates(monkeypatch):
    """KeyError (missing column) propagates unchanged — not caught."""
    import spotforecast2_safe.downloader.entsoe as entsoe_mod

    def _raise(df):
        raise KeyError("Forecasted Load")

    monkeypatch.setattr(entsoe_mod, "extract_entsoe_hourly_forecast", _raise)

    with pytest.raises(KeyError):
        entsoe_predictions(_minimal_interim(), _LOG)


# ---------------------------------------------------------------------------
# download_combined_or_abort
# ---------------------------------------------------------------------------


def _resil_result(mode, fallback_gate):
    """Build a minimal DownloadResult-like mock."""
    r = MagicMock()
    r.mode = mode
    r.fallback_gate = fallback_gate
    return r


def test_download_combined_success_live(monkeypatch, tmp_path):
    """Live download: returns False (no snapshot restored)."""
    _setup_env(monkeypatch, tmp_path)
    import spotforecast2_safe.downloader.resilience as resil_mod

    monkeypatch.setattr(
        resil_mod,
        "download_combined_with_fallback",
        lambda *a, **kw: _resil_result(mode="combined", fallback_gate=False),
    )

    result = download_combined_or_abort(
        "fake_key",
        _fake_dates(),
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=True,
        country_code="DE",
        logger=_LOG,
    )
    assert result is False


def test_download_combined_success_cache(monkeypatch, tmp_path):
    """Cache (snapshot) restore: returns True."""
    _setup_env(monkeypatch, tmp_path)
    import spotforecast2_safe.downloader.resilience as resil_mod

    monkeypatch.setattr(
        resil_mod,
        "download_combined_with_fallback",
        lambda *a, **kw: _resil_result(mode="combined", fallback_gate=True),
    )

    result = download_combined_or_abort(
        "fake_key",
        _fake_dates(),
        max_retries=1,
        backoff=0.0,
        timeout=None,
        fallback_enabled=True,
        country_code="DE",
        logger=_LOG,
    )
    assert result is True


def test_download_combined_mode_none_raises_abort(monkeypatch, tmp_path):
    """mode=None -> AbortReason.DOWNLOAD_UNRECOVERABLE."""
    _setup_env(monkeypatch, tmp_path)
    import spotforecast2_safe.downloader.resilience as resil_mod

    monkeypatch.setattr(
        resil_mod,
        "download_combined_with_fallback",
        lambda *a, **kw: _resil_result(mode=None, fallback_gate=True),
    )

    with pytest.raises(SubmissionAbort) as exc_info:
        download_combined_or_abort(
            "fake_key",
            _fake_dates(),
            max_retries=1,
            backoff=0.0,
            timeout=None,
            fallback_enabled=True,
            country_code="DE",
            logger=_LOG,
        )
    assert exc_info.value.reason is AbortReason.DOWNLOAD_UNRECOVERABLE
    assert "operator intervention" in str(exc_info.value)


def test_download_combined_no_fallback_message(monkeypatch, tmp_path):
    """mode=None + fallback_enabled=False -> message mentions --no-fallback."""
    _setup_env(monkeypatch, tmp_path)
    import spotforecast2_safe.downloader.resilience as resil_mod

    monkeypatch.setattr(
        resil_mod,
        "download_combined_with_fallback",
        lambda *a, **kw: _resil_result(mode=None, fallback_gate=True),
    )

    with pytest.raises(SubmissionAbort) as exc_info:
        download_combined_or_abort(
            "fake_key",
            _fake_dates(),
            max_retries=1,
            backoff=0.0,
            timeout=None,
            fallback_enabled=False,
            country_code="DE",
            logger=_LOG,
        )
    assert exc_info.value.reason is AbortReason.DOWNLOAD_UNRECOVERABLE
    assert "--no-fallback" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Public surface check
# ---------------------------------------------------------------------------


def test_public_names_present():
    """All mandated public names are accessible on the module."""
    import spotforecast2_safe.submission_cli as m

    assert hasattr(m, "AbortReason")
    assert hasattr(m, "SubmissionAbort")
    assert hasattr(m, "compute_dates_or_abort")
    assert hasattr(m, "load_interim_or_abort")
    assert hasattr(m, "assert_coverage_or_abort")
    assert hasattr(m, "entsoe_predictions")
    assert hasattr(m, "download_combined_or_abort")

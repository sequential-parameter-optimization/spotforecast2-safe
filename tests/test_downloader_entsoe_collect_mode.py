# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for collect mode and build_zone_qc_frame in download_zone_loads."""

import sys
import types

import pandas as pd
import pytest

from spotforecast2_safe.downloader.entsoe import (
    GERMAN_TSO_ZONES,
    ZoneResult,
    build_zone_qc_frame,
    download_zone_loads,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
    return tmp_path


class _FakeClient:
    """Stand-in for entsoe.EntsoePandasClient returning per-area load frames."""

    _LOADS = {
        "DE_AMPRION": 100.0,
        "DE_TENNET": 120.0,
        "DE_TRANSNET": 40.0,
        "DE_50HZ": 50.0,
    }

    def __init__(self, api_key, timeout=None):
        self.api_key = api_key
        self.timeout = timeout

    def query_load_and_forecast(self, country_code, start, end):
        idx = pd.date_range(start, end, freq="h", inclusive="left")
        level = self._LOADS.get(country_code, 10.0)
        return pd.DataFrame(
            {"Actual Load": level, "Forecasted Load": level + 1.0}, index=idx
        )


class _FailingClient(_FakeClient):
    """Like _FakeClient but raises for DE_TENNET (zone 2 of 4)."""

    def query_load_and_forecast(self, country_code, start, end):
        if country_code == "DE_TENNET":
            raise RuntimeError("simulated TENNET failure")
        return super().query_load_and_forecast(country_code, start, end)


@pytest.fixture
def fake_entsoe(monkeypatch):
    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = _FakeClient
    monkeypatch.setitem(sys.modules, "entsoe", module)
    return module


@pytest.fixture
def failing_entsoe(monkeypatch):
    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = _FailingClient
    monkeypatch.setitem(sys.modules, "entsoe", module)
    return module


# ---------------------------------------------------------------------------
# collect/raise parity: all zones succeed
# ---------------------------------------------------------------------------


def test_collect_all_ok_returns_dict_of_zone_results(data_home, fake_entsoe):
    """Collect mode with all zones succeeding returns 4 ok=True ZoneResults."""
    results = download_zone_loads(
        api_key="x",
        start="202301010000",
        end="202301050000",
        force=True,
        on_zone_failure="collect",
    )

    assert isinstance(results, dict)
    assert list(results.keys()) == list(GERMAN_TSO_ZONES.keys())

    for col, r in results.items():
        assert isinstance(r, ZoneResult)
        assert r.ok is True
        assert r.error is None
        assert r.column == col
        assert r.area == GERMAN_TSO_ZONES[col]
        assert r.interim_path is not None
        assert (
            r.interim_path.exists()
        ), f"interim path missing for {col}: {r.interim_path}"


def test_raise_mode_returns_none_and_writes_files(data_home, fake_entsoe):
    """Raise mode (default) returns None and writes per-zone interim files."""
    result = download_zone_loads(
        api_key="x",
        start="202301010000",
        end="202301050000",
        force=True,
        # on_zone_failure defaults to "raise"
    )

    assert result is None
    for col in GERMAN_TSO_ZONES:
        path = data_home / "interim" / f"zone_{col}.csv"
        assert path.exists(), f"missing interim file for {col}"


def test_collect_and_raise_write_same_files(
    data_home, fake_entsoe, tmp_path, monkeypatch
):
    """Collect and raise mode produce identical interim file content."""
    # raise-mode run in data_home
    download_zone_loads(
        api_key="x",
        start="202301010000",
        end="202301030000",
        force=True,
    )
    raise_frames = {}
    for col in GERMAN_TSO_ZONES:
        raise_frames[col] = pd.read_csv(
            data_home / "interim" / f"zone_{col}.csv", index_col=0
        )

    # collect-mode run in a separate tmp dir
    collect_home = tmp_path / "collect_home"
    collect_home.mkdir()
    monkeypatch.setenv("SPOTFORECAST2_DATA", str(collect_home))
    download_zone_loads(
        api_key="x",
        start="202301010000",
        end="202301030000",
        force=True,
        on_zone_failure="collect",
    )
    for col in GERMAN_TSO_ZONES:
        collect_frame = pd.read_csv(
            collect_home / "interim" / f"zone_{col}.csv", index_col=0
        )
        pd.testing.assert_frame_equal(raise_frames[col], collect_frame)


# ---------------------------------------------------------------------------
# partial failure: zone 2 (load_tennet) raises
# ---------------------------------------------------------------------------


def test_collect_partial_failure_records_error_continues(data_home, failing_entsoe):
    """Collect mode: DE_TENNET fails; other three zones succeed."""
    results = download_zone_loads(
        api_key="x",
        start="202301010000",
        end="202301050000",
        force=True,
        on_zone_failure="collect",
    )

    assert isinstance(results, dict)
    assert results["load_tennet"].ok is False
    assert isinstance(results["load_tennet"].error, Exception)
    assert results["load_tennet"].interim_path is None
    assert not (data_home / "interim" / "zone_load_tennet.csv").exists()

    for col in ("load_amprion", "load_transnetbw", "load_50hertz"):
        assert results[col].ok is True
        assert results[col].error is None
        assert results[col].interim_path is not None
        assert results[col].interim_path.exists()


def test_raise_mode_partial_failure_raises(data_home, failing_entsoe):
    """Raise mode: DE_TENNET failure propagates immediately."""
    with pytest.raises(RuntimeError):
        download_zone_loads(
            api_key="x",
            start="202301010000",
            end="202301050000",
            force=True,
        )


# ---------------------------------------------------------------------------
# argument validation always raises
# ---------------------------------------------------------------------------


def test_invalid_on_zone_failure_raises_valueerror(data_home, fake_entsoe):
    """Unknown on_zone_failure value raises ValueError before any download."""
    with pytest.raises(ValueError, match="on_zone_failure"):
        download_zone_loads(
            api_key="x",
            start="202301010000",
            end="202301050000",
            force=True,
            on_zone_failure="ignore",  # type: ignore[arg-type]
        )


def test_invalid_on_zone_failure_raises_in_collect_context(data_home, fake_entsoe):
    """Validation error is raised even when a collect-like value is expected."""
    with pytest.raises(ValueError):
        download_zone_loads(
            api_key="x",
            on_zone_failure="skip",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# build_zone_qc_frame
# ---------------------------------------------------------------------------


def _write_zone_csv(interim_dir, col, actual, idx, forecast=None):
    interim_dir.mkdir(parents=True, exist_ok=True)
    data = {col: actual}
    if forecast is not None:
        data[f"{col}_forecast"] = forecast
    pd.DataFrame(data, index=idx).rename_axis("Time (UTC)").to_csv(
        interim_dir / f"zone_{col}.csv"
    )


def test_build_zone_qc_frame_full_four_zones_totals(data_home):
    """Full four zones: Actual Load and Forecasted Load totals are correct."""
    zones = {
        "load_amprion": "DE_AMPRION",
        "load_tennet": "DE_TENNET",
        "load_transnetbw": "DE_TRANSNET",
        "load_50hertz": "DE_50HZ",
    }
    idx = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")
    levels = {
        "load_amprion": 100.0,
        "load_tennet": 120.0,
        "load_transnetbw": 40.0,
        "load_50hertz": 50.0,
    }
    forecasts = {col: lvl + 1.0 for col, lvl in levels.items()}

    for col, lvl in levels.items():
        _write_zone_csv(data_home / "interim", col, lvl, idx, forecast=forecasts[col])

    qc = build_zone_qc_frame(zones=zones)

    expected_actual = sum(levels.values())
    expected_forecast = sum(forecasts.values())

    assert "Actual Load" in qc.columns
    assert "Forecasted Load" in qc.columns
    assert qc["Actual Load"].iloc[0] == pytest.approx(expected_actual)
    assert qc["Forecasted Load"].iloc[0] == pytest.approx(expected_forecast)
    # All zone columns present
    for col in zones:
        assert col in qc.columns


def test_build_zone_qc_frame_min_count_partial_nan(data_home):
    """A missing zone tail row makes Actual Load NaN there (min_count semantics)."""
    zones = {"load_a": "AREA_A", "load_b": "AREA_B"}
    idx = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")

    # load_a has a shorter index (drop last 2 rows) — its tail will be NaN after outer-join
    _write_zone_csv(data_home / "interim", "load_a", 100.0, idx[:4], forecast=105.0)
    _write_zone_csv(data_home / "interim", "load_b", 50.0, idx, forecast=52.0)

    qc = build_zone_qc_frame(zones=zones)

    # First 4 rows: both zones present → total is non-NaN
    assert not qc["Actual Load"].iloc[:4].isna().any()
    # Last 2 rows: load_a is NaN → total must be NaN (min_count enforced)
    assert qc["Actual Load"].iloc[4:].isna().all()


def test_build_zone_qc_frame_missing_forecast_column_all_nan(data_home):
    """If any zone lacks a forecast column, Forecasted Load is all-NaN."""
    zones = {"load_a": "AREA_A", "load_b": "AREA_B"}
    idx = pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC")

    # load_a has no forecast column
    _write_zone_csv(data_home / "interim", "load_a", 100.0, idx)  # no forecast
    _write_zone_csv(data_home / "interim", "load_b", 50.0, idx, forecast=52.0)

    qc = build_zone_qc_frame(zones=zones)

    assert "Forecasted Load" in qc.columns
    assert (
        qc["Forecasted Load"].isna().all()
    ), "Expected all-NaN Forecasted Load when not all zones have a forecast column"


def test_build_zone_qc_frame_missing_interim_file_raises(data_home):
    """Missing interim file raises FileNotFoundError naming the path."""
    zones = {"load_amprion": "DE_AMPRION"}
    with pytest.raises(FileNotFoundError, match="zone_load_amprion"):
        build_zone_qc_frame(zones=zones)


def test_build_zone_qc_frame_index_is_utc_sorted(data_home):
    """Result index is UTC-aware and sorted ascending."""
    zones = {"load_a": "AREA_A"}
    idx = pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC")
    _write_zone_csv(data_home / "interim", "load_a", 100.0, idx[::-1], forecast=105.0)

    qc = build_zone_qc_frame(zones=zones)

    assert qc.index.is_monotonic_increasing
    assert str(qc.index.tz) == "UTC"


def test_build_zone_qc_frame_uses_default_german_zones(data_home):
    """Calling without zones= uses GERMAN_TSO_ZONES (fails fast on missing file)."""
    with pytest.raises(FileNotFoundError):
        build_zone_qc_frame()  # no zones= → uses GERMAN_TSO_ZONES → files missing


def test_build_zone_qc_frame_explicit_data_home_no_env_var(tmp_path, monkeypatch):
    """data_home= kwarg works without SPOTFORECAST2_DATA env var being set."""
    monkeypatch.delenv("SPOTFORECAST2_DATA", raising=False)
    zones = {"load_a": "AREA_A", "load_b": "AREA_B"}
    idx = pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC")
    interim = tmp_path / "interim"
    interim.mkdir(parents=True)
    for col, actual, forecast in [("load_a", 100.0, 105.0), ("load_b", 50.0, 52.0)]:
        pd.DataFrame({col: actual, f"{col}_forecast": forecast}, index=idx).rename_axis(
            "Time (UTC)"
        ).to_csv(interim / f"zone_{col}.csv")

    qc = build_zone_qc_frame(zones=zones, data_home=tmp_path)

    assert qc["Actual Load"].iloc[0] == pytest.approx(150.0)
    assert qc["Forecasted Load"].iloc[0] == pytest.approx(157.0)

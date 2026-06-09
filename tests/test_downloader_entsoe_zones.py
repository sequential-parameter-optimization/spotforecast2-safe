# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the four-German-TSO-zone load downloader and assembler."""

import sys
import types

import pandas as pd
import pytest

from spotforecast2_safe.downloader.entsoe import (
    GERMAN_TSO_ZONES,
    assemble_zone_loads,
    download_zone_loads,
)


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


@pytest.fixture
def fake_entsoe(monkeypatch):
    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = _FakeClient
    monkeypatch.setitem(sys.modules, "entsoe", module)
    return module


def test_zone_constants_are_entsoe_area_codes():
    assert list(GERMAN_TSO_ZONES) == [
        "load_amprion",
        "load_tennet",
        "load_transnetbw",
        "load_50hertz",
    ]
    assert GERMAN_TSO_ZONES["load_amprion"] == "DE_AMPRION"
    assert GERMAN_TSO_ZONES["load_50hertz"] == "DE_50HZ"


def test_download_and_assemble_roundtrip(data_home, fake_entsoe):
    # A wholly-past window so no rows are dropped as "future".
    download_zone_loads(
        api_key="x",
        start="202301010000",
        end="202301050000",
        force=True,
    )

    # Each zone gets its own namespaced per-zone interim file with its column.
    for col in GERMAN_TSO_ZONES:
        path = data_home / "interim" / f"zone_{col}.csv"
        assert path.exists(), f"missing per-zone interim file for {col}"

    frame = assemble_zone_loads()
    assert frame.columns.tolist() == list(GERMAN_TSO_ZONES)
    assert (frame["load_amprion"].dropna() == 100.0).all()
    assert (frame["load_tennet"].dropna() == 120.0).all()
    # Bottom-up total = sum of the four control-zone loads.
    assert frame.sum(axis=1).dropna().iloc[0] == 100.0 + 120.0 + 40.0 + 50.0
    assert (data_home / "interim" / "energy_load_zones.csv").exists()


def test_download_zone_loads_drops_future_by_default(data_home, fake_entsoe):
    # Default window reaches into tomorrow; keep_forecast_future defaults False,
    # so the per-zone interim must not carry rows after the current UTC moment.
    download_zone_loads(api_key="x", force=True)
    interim = pd.read_csv(data_home / "interim" / "zone_load_amprion.csv")
    interim["Time (UTC)"] = pd.to_datetime(interim["Time (UTC)"], utc=True)
    assert interim["Time (UTC)"].max() <= pd.Timestamp.now(tz="UTC")


def _write_zone_interim(data_home, col, level, idx):
    interim = data_home / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({col: level}, index=idx).rename_axis("Time (UTC)").to_csv(
        interim / f"zone_{col}.csv"
    )


def test_assemble_raises_on_gap(data_home):
    zones = {"load_amprion": "DE_AMPRION", "load_tennet": "DE_TENNET"}
    idx = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")
    _write_zone_interim(data_home, "load_amprion", 100.0, idx.delete(2))  # gap
    _write_zone_interim(data_home, "load_tennet", 120.0, idx)
    with pytest.raises(ValueError, match="missing values"):
        assemble_zone_loads(zones=zones)


def test_assemble_passthrough_keeps_nan(data_home):
    zones = {"load_amprion": "DE_AMPRION", "load_tennet": "DE_TENNET"}
    idx = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")
    _write_zone_interim(data_home, "load_amprion", 100.0, idx.delete(2))
    _write_zone_interim(data_home, "load_tennet", 120.0, idx)
    frame = assemble_zone_loads(zones=zones, on_missing="passthrough")
    assert int(frame.isna().sum().sum()) == 1  # the single dropped hour
    assert len(frame) == 6  # reindexed onto the complete hourly range


def test_assemble_unknown_on_missing_raises(data_home):
    with pytest.raises(ValueError, match="on_missing"):
        assemble_zone_loads(on_missing="fill")


def test_assemble_missing_file_raises(data_home):
    with pytest.raises(FileNotFoundError):
        assemble_zone_loads(zones={"load_amprion": "DE_AMPRION"})

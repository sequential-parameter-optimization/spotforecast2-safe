# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ENTSO-E day-ahead side-table downloaders and loaders."""

import sys
import types

import pandas as pd
import pytest

from spotforecast2_safe.data.fetch_data import (
    load_day_ahead_price,
    load_renewable_forecast,
)
from spotforecast2_safe.downloader.entsoe import (
    download_day_ahead_price,
    download_renewable_forecast,
    merge_build_manual,
)


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
    return tmp_path


class _FakeClient:
    """Stand-in for entsoe.EntsoePandasClient returning deterministic frames."""

    def __init__(self, api_key, timeout=None):
        self.api_key = api_key
        self.timeout = timeout

    def query_wind_and_solar_forecast(self, country_code, start, end):
        idx = pd.date_range(start, end, freq="h", inclusive="left")
        return pd.DataFrame({"Solar": 1.0, "Wind Onshore": 2.0}, index=idx)

    def query_day_ahead_prices(self, country_code, start, end):
        idx = pd.date_range(start, end, freq="h", inclusive="left")
        return pd.Series(50.0, index=idx, name="ignored")


@pytest.fixture
def fake_entsoe(monkeypatch):
    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = _FakeClient
    monkeypatch.setitem(sys.modules, "entsoe", module)
    return module


# ---------------------------------------------------------------------------
# merge_build_manual raw_subdir namespacing
# ---------------------------------------------------------------------------


def test_merge_raw_subdir_isolated_from_load(data_home):
    idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
    # a load file directly under raw/ ...
    (data_home / "raw").mkdir(parents=True)
    pd.DataFrame({"Actual Load": 1.0}, index=idx).rename_axis("Time (UTC)").to_csv(
        data_home / "raw" / "entsoe_load_a.csv"
    )
    # ... and a renewable file under raw/renewable/
    (data_home / "raw" / "renewable").mkdir()
    pd.DataFrame({"Solar": 9.0}, index=idx).rename_axis("Time (UTC)").to_csv(
        data_home / "raw" / "renewable" / "r.csv"
    )

    merge_build_manual(output_file="energy_load.csv", keep_forecast_future=True)
    merge_build_manual(
        output_file="renewable_forecast.csv",
        keep_forecast_future=True,
        raw_subdir="renewable",
    )

    load = pd.read_csv(data_home / "interim" / "energy_load.csv")
    ren = pd.read_csv(data_home / "interim" / "renewable_forecast.csv")
    assert "Actual Load" in load.columns and "Solar" not in load.columns
    assert "Solar" in ren.columns and "Actual Load" not in ren.columns


# ---------------------------------------------------------------------------
# download_* (mocked client)
# ---------------------------------------------------------------------------


def test_download_renewable_forecast(data_home, fake_entsoe):
    download_renewable_forecast(
        api_key="x",
        country_code="DE",
        start="202301010000",
        end="202301050000",
        force=True,
    )
    out = load_renewable_forecast()
    assert sorted(out.columns) == ["Solar", "Wind Onshore"]
    assert (out["Wind Onshore"].dropna() == 2.0).all()


def test_download_day_ahead_price(data_home, fake_entsoe):
    download_day_ahead_price(
        api_key="x",
        country_code="DE_LU",
        start="202301010000",
        end="202301050000",
        force=True,
    )
    out = load_day_ahead_price()
    assert (out.dropna() == 50.0).all()
    assert out.name == "Day-ahead Price"


def test_download_renewable_default_timeout_reaches_client(data_home, fake_entsoe):
    """The default timeout=60.0 arrives at _FakeClient.timeout."""
    # We need a capturing version of _FakeClient to inspect the timeout.
    captured = {}

    class CapturingFakeClient(_FakeClient):
        def __init__(self, api_key, timeout=None):
            super().__init__(api_key, timeout=timeout)
            captured["timeout"] = timeout

    import types

    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = CapturingFakeClient
    import sys

    sys.modules["entsoe"] = module

    download_renewable_forecast(
        api_key="x",
        country_code="DE",
        start="202301010000",
        end="202301050000",
        force=True,
    )
    assert captured.get("timeout") == 60.0


def test_download_price_default_timeout_reaches_client(data_home, fake_entsoe):
    """The default timeout=60.0 arrives at _FakeClient.timeout for the price table."""
    captured = {}

    class CapturingFakeClient(_FakeClient):
        def __init__(self, api_key, timeout=None):
            super().__init__(api_key, timeout=timeout)
            captured["timeout"] = timeout

    import types

    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = CapturingFakeClient
    import sys

    sys.modules["entsoe"] = module

    download_day_ahead_price(
        api_key="x",
        country_code="DE_LU",
        start="202301010000",
        end="202301050000",
        force=True,
    )
    assert captured.get("timeout") == 60.0


def test_download_missing_entsoe_raises(data_home, monkeypatch):
    monkeypatch.setitem(sys.modules, "entsoe", None)
    with pytest.raises((ImportError, TypeError)):
        download_renewable_forecast(
            api_key="x", start="202301010000", end="202301050000", force=True
        )


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def test_load_renewable_forecast_missing_file(data_home):
    with pytest.raises(FileNotFoundError):
        load_renewable_forecast()


def test_load_day_ahead_price_missing_column(data_home):
    interim = data_home / "interim"
    interim.mkdir(parents=True)
    idx = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
    pd.DataFrame({"Other": 1.0}, index=idx).rename_axis("Time (UTC)").to_csv(
        interim / "day_ahead_price.csv"
    )
    with pytest.raises(KeyError):
        load_day_ahead_price()

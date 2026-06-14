# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ENTSO-E day-ahead side-table downloaders and loaders."""

import sys
import types

import pandas as pd
import pytest

import spotforecast2_safe.downloader.entsoe as _entsoe_mod
from spotforecast2_safe.data.fetch_data import (
    load_day_ahead_price,
    load_renewable_forecast,
)
from spotforecast2_safe.downloader.entsoe import (
    download_day_ahead_price,
    download_renewable_forecast,
    download_side_tables,
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


class _FailingRenewableClient(_FakeClient):
    """Fake client whose renewable query always raises RuntimeError."""

    def query_wind_and_solar_forecast(self, country_code, start, end):
        raise RuntimeError("synthetic renewable failure")


class _FailingPriceClient(_FakeClient):
    """Fake client whose day-ahead price query always raises RuntimeError."""

    def query_day_ahead_prices(self, country_code, start, end):
        raise RuntimeError("synthetic price failure")


@pytest.fixture
def fake_entsoe(monkeypatch):
    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = _FakeClient
    monkeypatch.setitem(sys.modules, "entsoe", module)
    return module


@pytest.fixture
def failing_entsoe(monkeypatch):
    """entsoe module whose renewable client always raises."""
    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = _FailingRenewableClient
    monkeypatch.setitem(sys.modules, "entsoe", module)
    return module


@pytest.fixture
def failing_price_entsoe(monkeypatch):
    """entsoe module whose day-ahead price client always raises."""
    module = types.ModuleType("entsoe")
    module.EntsoePandasClient = _FailingPriceClient
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


# ---------------------------------------------------------------------------
# download_side_tables
# ---------------------------------------------------------------------------


def test_download_side_tables_raise_mode_both_providers_called(data_home, fake_entsoe):
    """Raise-mode (default): both providers succeed, both interim files written."""
    result = download_side_tables(
        api_key="x",
        start="202301010000",
        end="202301050000",
        force=True,
    )
    assert result is None
    assert (data_home / "interim" / "renewable_forecast.csv").exists()
    assert (data_home / "interim" / "day_ahead_price.csv").exists()


def test_download_side_tables_raise_mode_propagates(
    data_home, failing_entsoe, monkeypatch
):
    """Raise-mode: a failing renewable query propagates as RuntimeError."""
    # Speed up the retry loop so the test doesn't take 25 s.
    monkeypatch.setattr(_entsoe_mod, "_RETRY_BACKOFF_SECONDS", 0)
    with pytest.raises(RuntimeError):
        download_side_tables(
            api_key="x",
            start="202301010000",
            end="202301050000",
            force=True,
        )


def test_download_side_tables_skip_mode_renewable_fails_price_succeeds(
    data_home, failing_entsoe, monkeypatch, caplog
):
    """Skip-mode: renewable failure is logged; price still written; no raise."""
    monkeypatch.setattr(_entsoe_mod, "_RETRY_BACKOFF_SECONDS", 0)
    import logging

    with caplog.at_level(
        logging.WARNING, logger="spotforecast2_safe.downloader.entsoe"
    ):
        result = download_side_tables(
            api_key="x",
            start="202301010000",
            end="202301050000",
            force=True,
            on_provider_failure="skip",
        )
    assert result is None
    # WARNING must mention the failing provider
    assert any("renewable forecast" in r.message for r in caplog.records)
    # Price file should be written despite the renewable failure.
    assert (data_home / "interim" / "day_ahead_price.csv").exists()
    # Renewable interim file should NOT exist.
    assert not (data_home / "interim" / "renewable_forecast.csv").exists()


def test_download_side_tables_skip_mode_price_fails_renewable_succeeds(
    data_home, failing_price_entsoe, monkeypatch, caplog
):
    """Skip-mode (symmetric): price failure is logged; renewable still written."""
    monkeypatch.setattr(_entsoe_mod, "_RETRY_BACKOFF_SECONDS", 0)
    import logging

    with caplog.at_level(
        logging.WARNING, logger="spotforecast2_safe.downloader.entsoe"
    ):
        result = download_side_tables(
            api_key="x",
            start="202301010000",
            end="202301050000",
            force=True,
            on_provider_failure="skip",
        )
    assert result is None
    assert any("day-ahead price" in r.message for r in caplog.records)
    # Renewable file written despite the price failure.
    assert (data_home / "interim" / "renewable_forecast.csv").exists()
    assert not (data_home / "interim" / "day_ahead_price.csv").exists()


def test_download_side_tables_invalid_on_provider_failure_raises_before_download(
    data_home, monkeypatch
):
    """Invalid on_provider_failure raises ValueError before any download."""
    # Use no fake entsoe so that if a download is attempted the import fails —
    # but the ValueError should fire first.
    monkeypatch.setitem(sys.modules, "entsoe", None)
    with pytest.raises(ValueError, match="on_provider_failure"):
        download_side_tables(
            api_key="x",
            start="202301010000",
            end="202301050000",
            force=True,
            on_provider_failure="ignore",
        )
    # Neither interim file should have been created.
    assert not (data_home / "interim" / "renewable_forecast.csv").exists()
    assert not (data_home / "interim" / "day_ahead_price.csv").exists()


def test_download_side_tables_country_code_routing(data_home, fake_entsoe, monkeypatch):
    """Renewable receives country_code='DE'; price receives 'DE_LU'."""
    all_calls: list[tuple[str, str]] = []
    original_renewable = _FakeClient.query_wind_and_solar_forecast
    original_price = _FakeClient.query_day_ahead_prices

    def patched_renewable(self, country_code, start, end):
        all_calls.append(("renewable", country_code))
        return original_renewable(self, country_code, start, end)

    def patched_price(self, country_code, start, end):
        all_calls.append(("price", country_code))
        return original_price(self, country_code, start, end)

    # monkeypatch.setattr auto-reverts after the test (no manual try/finally).
    monkeypatch.setattr(_FakeClient, "query_wind_and_solar_forecast", patched_renewable)
    monkeypatch.setattr(_FakeClient, "query_day_ahead_prices", patched_price)

    download_side_tables(
        api_key="x",
        start="202301010000",
        end="202301050000",
        force=True,
        country_code="DE",
        price_country_code="DE_LU",
    )

    renewable_codes = [cc for (name, cc) in all_calls if name == "renewable"]
    price_codes = [cc for (name, cc) in all_calls if name == "price"]
    assert renewable_codes == ["DE"]
    assert price_codes == ["DE_LU"]

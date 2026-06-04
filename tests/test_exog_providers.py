# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the pluggable exogenous-feature providers."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.exog_providers import (
    EXOG_PROVIDER_REGISTRY,
    CovidInfectionRateProvider,
    EntsoeDayAheadPriceProvider,
    EntsoeForecastLoadProvider,
    EntsoeNetLoadProvider,
    EntsoeRenewableForecastProvider,
    ExogProviderError,
    build_providers,
    build_providers_from_config,
)


def _write_interim(data_home, name, frame):
    interim = data_home / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    frame.rename_axis("Time (UTC)").to_csv(interim / name)


@pytest.fixture
def hourly_index():
    return pd.date_range("2023-01-01", periods=72, freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# CovidInfectionRateProvider
# ---------------------------------------------------------------------------


class TestCovidProvider:
    def test_builds_bundled_series_in_pandemic_window(self):
        idx = pd.date_range("2021-12-01", periods=48, freq="h", tz="UTC")
        out = CovidInfectionRateProvider().build(idx)
        assert out.columns.tolist() == ["covid_infection_rate"]
        assert len(out) == 48
        assert not out.isna().any().any()
        assert (out["covid_infection_rate"] > 0).any()
        assert str(out.dtypes.iloc[0]) == "float32"

    def test_zero_fill_before_pandemic(self):
        idx = pd.date_range("2015-01-01", periods=24, freq="h", tz="UTC")
        out = CovidInfectionRateProvider().build(idx)
        assert (out["covid_infection_rate"] == 0.0).all()

    def test_zero_fill_after_series_end(self):
        idx = pd.date_range("2035-01-01", periods=24, freq="h", tz="UTC")
        out = CovidInfectionRateProvider().build(idx)
        assert (out["covid_infection_rate"] == 0.0).all()

    def test_custom_csv_path_and_ffill(self, tmp_path):
        csv = tmp_path / "covid.csv"
        pd.DataFrame(
            {
                "date": ["2022-01-01", "2022-01-03"],
                "covid_infection_rate": [100.0, 300.0],
            }
        ).to_csv(csv, index=False)
        idx = pd.date_range("2022-01-01", "2022-01-04 23:00", freq="h", tz="UTC")
        out = CovidInfectionRateProvider(csv_path=csv).build(idx)
        # 2022-01-02 forward-fills the 2022-01-01 value
        assert out.loc["2022-01-01 05:00", "covid_infection_rate"] == pytest.approx(
            100.0
        )
        assert out.loc["2022-01-02 05:00", "covid_infection_rate"] == pytest.approx(
            100.0
        )
        assert out.loc["2022-01-03 05:00", "covid_infection_rate"] == pytest.approx(
            300.0
        )
        # outside the [first, last] span -> fill_outside (0.0)
        assert out.loc["2022-01-04 05:00", "covid_infection_rate"] == pytest.approx(0.0)

    def test_custom_fill_outside(self, tmp_path):
        csv = tmp_path / "covid.csv"
        pd.DataFrame({"date": ["2022-01-01"], "covid_infection_rate": [50.0]}).to_csv(
            csv, index=False
        )
        idx = pd.date_range("2010-01-01", periods=5, freq="h", tz="UTC")
        out = CovidInfectionRateProvider(csv_path=csv, fill_outside=-1.0).build(idx)
        assert (out["covid_infection_rate"] == -1.0).all()

    def test_tz_naive_index_supported(self):
        idx = pd.date_range("2021-12-01", periods=12, freq="h")  # tz-naive
        out = CovidInfectionRateProvider().build(idx)
        assert len(out) == 12 and not out.isna().any().any()

    def test_empty_index(self):
        idx = pd.DatetimeIndex([], tz="UTC")
        out = CovidInfectionRateProvider().build(idx)
        assert out.empty and out.columns.tolist() == ["covid_infection_rate"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ExogProviderError):
            CovidInfectionRateProvider(csv_path=tmp_path / "nope.csv").build(
                pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC")
            )


# ---------------------------------------------------------------------------
# EntsoeForecastLoadProvider
# ---------------------------------------------------------------------------


class TestForecastLoadProvider:
    def test_builds_from_interim(self, tmp_path, hourly_index):
        _write_interim(
            tmp_path,
            "energy_load.csv",
            pd.DataFrame(
                {"Actual Load": 1000.0, "Forecasted Load": 1010.0}, index=hourly_index
            ),
        )
        out = EntsoeForecastLoadProvider(data_home=tmp_path).build(hourly_index)
        assert out.columns.tolist() == ["entsoe_forecasted_load"]
        assert (out["entsoe_forecasted_load"] == 1010.0).all()

    def test_missing_file_raises(self, tmp_path, hourly_index):
        with pytest.raises(ExogProviderError):
            EntsoeForecastLoadProvider(data_home=tmp_path).build(hourly_index)

    def test_gap_in_window_raises(self, tmp_path, hourly_index):
        frame = pd.DataFrame(
            {"Actual Load": 1000.0, "Forecasted Load": 1010.0}, index=hourly_index
        )
        frame.iloc[5, 1] = np.nan
        _write_interim(tmp_path, "energy_load.csv", frame)
        with pytest.raises(ExogProviderError):
            EntsoeForecastLoadProvider(data_home=tmp_path).build(hourly_index)


# ---------------------------------------------------------------------------
# EntsoeRenewableForecastProvider / NetLoad / Price
# ---------------------------------------------------------------------------


class TestRenewableNetPriceProviders:
    def _seed(self, data_home, index):
        _write_interim(
            data_home,
            "energy_load.csv",
            pd.DataFrame(
                {"Actual Load": 1000.0, "Forecasted Load": 1010.0}, index=index
            ),
        )
        _write_interim(
            data_home,
            "renewable_forecast.csv",
            pd.DataFrame(
                {"Solar": 10.0, "Wind Onshore": 20.0, "Wind Offshore": 5.0}, index=index
            ),
        )
        _write_interim(
            data_home,
            "day_ahead_price.csv",
            pd.DataFrame({"Day-ahead Price": 90.0}, index=index),
        )

    def test_renewable_sums_wind_and_solar(self, tmp_path, hourly_index):
        self._seed(tmp_path, hourly_index)
        out = EntsoeRenewableForecastProvider(data_home=tmp_path).build(hourly_index)
        assert sorted(out.columns) == ["entsoe_solar_forecast", "entsoe_wind_forecast"]
        assert (out["entsoe_wind_forecast"] == 25.0).all()
        assert (out["entsoe_solar_forecast"] == 10.0).all()

    def test_renewable_missing_columns_raises(self, tmp_path, hourly_index):
        _write_interim(
            tmp_path,
            "renewable_forecast.csv",
            pd.DataFrame({"Nuclear": 1.0}, index=hourly_index),
        )
        with pytest.raises(ExogProviderError):
            EntsoeRenewableForecastProvider(data_home=tmp_path).build(hourly_index)

    def test_net_load_arithmetic(self, tmp_path, hourly_index):
        self._seed(tmp_path, hourly_index)
        out = EntsoeNetLoadProvider(data_home=tmp_path).build(hourly_index)
        # 1010 - (10 + 20 + 5) = 975
        assert (out["entsoe_net_load"] == 975.0).all()

    def test_net_load_missing_renewable_raises(self, tmp_path, hourly_index):
        _write_interim(
            tmp_path,
            "energy_load.csv",
            pd.DataFrame(
                {"Actual Load": 1000.0, "Forecasted Load": 1010.0}, index=hourly_index
            ),
        )
        with pytest.raises(ExogProviderError):
            EntsoeNetLoadProvider(data_home=tmp_path).build(hourly_index)

    def test_price_builds(self, tmp_path, hourly_index):
        self._seed(tmp_path, hourly_index)
        out = EntsoeDayAheadPriceProvider(data_home=tmp_path).build(hourly_index)
        assert out.columns.tolist() == ["entsoe_day_ahead_price"]
        assert (out["entsoe_day_ahead_price"] == 90.0).all()

    def test_price_missing_raises(self, tmp_path, hourly_index):
        with pytest.raises(ExogProviderError):
            EntsoeDayAheadPriceProvider(data_home=tmp_path).build(hourly_index)


# ---------------------------------------------------------------------------
# Registry + builders
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_flag_names(self):
        assert set(EXOG_PROVIDER_REGISTRY) == {
            "include_covid_infection_rate",
            "include_entsoe_forecast_load",
            "include_entsoe_renewable_forecast",
            "include_entsoe_net_load",
            "include_entsoe_day_ahead_price",
        }

    def test_build_providers_respects_flags_and_order(self):
        providers = build_providers(
            {
                "include_entsoe_day_ahead_price": True,
                "include_covid_infection_rate": True,
            }
        )
        # registry order, not call order
        assert [p.name for p in providers] == [
            "covid_infection_rate",
            "entsoe_day_ahead_price",
        ]

    def test_build_providers_empty(self):
        assert build_providers({}) == []

    def test_build_providers_from_config(self):
        from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe

        cfg = ConfigEntsoe(
            include_entsoe_forecast_load=True, include_entsoe_net_load=True
        )
        names = [p.name for p in build_providers_from_config(cfg)]
        assert names == ["entsoe_forecasted_load", "entsoe_net_load"]

    def test_build_providers_forwards_data_home(self, tmp_path):
        providers = build_providers(
            {"include_entsoe_forecast_load": True}, data_home=tmp_path
        )
        assert providers[0].data_home == tmp_path


# ---------------------------------------------------------------------------
# _align_to_index gap-healing (new tests)
# ---------------------------------------------------------------------------


def _make_frame(index: pd.DatetimeIndex, value: float = 100.0) -> pd.DataFrame:
    """Return a DataFrame with one column aligned to *index*."""
    return pd.DataFrame({"col": value}, index=index)


class TestAlignToIndexGapHealing:
    """Tests for max_gap and validate_index in _align_to_index."""

    def test_strict_default_raises_on_gap(self):
        """Default max_gap=0 + validate_index=None raises ExogProviderError."""
        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
        sparse = full[[0, 1, 2, 4, 5, 6, 7, 8, 9]]  # hour 3 missing
        frame = _make_frame(sparse)
        with pytest.raises(ExogProviderError) as exc_info:
            _align_to_index(frame, full, provider="test_provider")
        msg = str(exc_info.value)
        assert "test_provider" in msg
        assert "1 of 10" in msg
        assert "The provider cannot cover the requested range" in msg

    def test_interior_1h_gap_healed_at_max_gap_1(self):
        """A single interior NaN is healed by time-interpolation at max_gap=1."""
        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC")
        # values: 0, 10, [missing], 30, 40  — linear interpolation gives 20
        values = [0.0, 10.0, np.nan, 30.0, 40.0]
        frame = pd.DataFrame({"col": values}, index=full)
        result = _align_to_index(frame, full, provider="p", max_gap=1)
        assert not result.isna().any().any()
        assert result.dtypes.iloc[0] == np.dtype("float32")
        # time-interpolated value at hour 2 between 10 and 30 → 20
        assert result["col"].iloc[2] == pytest.approx(20.0, abs=1e-3)

    def test_25h_block_raises_at_max_gap_24(self):
        """A 25-hour contiguous gap is NOT healed when max_gap=24."""
        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=30, freq="h", tz="UTC")
        frame = pd.DataFrame({"col": 1.0}, index=full)
        # Punch a 25-row hole in the middle
        frame.iloc[2:27] = np.nan
        with pytest.raises(ExogProviderError):
            _align_to_index(frame, full, provider="p", max_gap=24)

    def test_oversize_run_outside_window_does_not_block_in_window_heal(self):
        """Healing is all-or-nothing PER RUN, not per frame.

        Mirrors the day-ahead-price reality: a years-long leading void
        (outside the validated train window) plus a small trailing tail
        (inside it). The oversize run must stay NaN and be tolerated by the
        window, while the tail is still healed.
        """
        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
        frame = pd.DataFrame({"col": 1.0}, index=full)
        frame.iloc[:50] = np.nan  # oversize leading run (50 h >> max_gap)
        frame.iloc[-2:] = np.nan  # 2-hour trailing tail
        window = full[60:]  # validated window excludes the leading void
        result = _align_to_index(
            frame, full, provider="p", max_gap=3, validate_index=window
        )
        # tail healed (ffill), in-window NaN-free, leading void still NaN
        assert not result.reindex(window).isna().any().any()
        assert result["col"].iloc[-1] == pytest.approx(1.0)
        assert result["col"].iloc[:50].isna().all()
        # re-mask boundary is exact: first valid cell is untouched
        assert result["col"].iloc[50] == pytest.approx(1.0)

    def test_trailing_2h_ffilled_at_max_gap_2(self):
        """Two trailing NaN cells are forward-filled at max_gap=2."""
        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")
        values = [1.0, 2.0, 3.0, 4.0, np.nan, np.nan]
        frame = pd.DataFrame({"col": values}, index=full)
        result = _align_to_index(frame, full, provider="p", max_gap=2)
        assert not result.isna().any().any()
        assert result["col"].iloc[4] == pytest.approx(4.0)
        assert result["col"].iloc[5] == pytest.approx(4.0)

    def test_leading_2h_bfilled_at_max_gap_2(self):
        """Two leading NaN cells are back-filled at max_gap=2."""
        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")
        values = [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0]
        frame = pd.DataFrame({"col": values}, index=full)
        result = _align_to_index(frame, full, provider="p", max_gap=2)
        assert not result.isna().any().any()
        assert result["col"].iloc[0] == pytest.approx(3.0)
        assert result["col"].iloc[1] == pytest.approx(3.0)

    def test_validate_index_window_suppresses_out_of_window_nans(self):
        """30 leading NaN outside a tail-24h validate window do not raise."""
        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=54, freq="h", tz="UTC")
        # Only the last 24 hours have data
        tail_start = full[-24]
        frame = pd.DataFrame({"col": np.nan}, index=full)
        frame.loc[tail_start:, "col"] = 99.0
        validate_window = pd.date_range(tail_start, periods=24, freq="h", tz="UTC")
        # max_gap=0 but out-of-window NaN should not trigger the gate
        result = _align_to_index(
            frame, full, provider="p", max_gap=0, validate_index=validate_window
        )
        # Out-of-window NaN cells remain
        assert result["col"].iloc[:30].isna().all()
        # In-window cells are NaN-free
        assert not result.loc[tail_start:, "col"].isna().any()
        assert result.dtypes.iloc[0] == np.dtype("float32")

    def test_caplog_warning_on_heal(self, caplog):
        """WARNING log contains provider name, healed count, and span."""
        import logging

        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC")
        values = [1.0, 2.0, np.nan, 4.0, 5.0]
        frame = pd.DataFrame({"col": values}, index=full)
        with caplog.at_level(
            logging.WARNING, logger="spotforecast2_safe.preprocessing.exog_providers"
        ):
            _align_to_index(frame, full, provider="my_provider", max_gap=1)
        assert any("my_provider" in r.message for r in caplog.records)
        assert any("healed" in r.message for r in caplog.records)
        assert any("max_gap=1" in r.message for r in caplog.records)

    def test_defaults_identical_to_prior_behaviour(self, tmp_path):
        """max_gap=0 + validate_index=None: no healing, strict fail-safe."""
        from spotforecast2_safe.preprocessing.exog_providers import _align_to_index

        full = pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC")
        frame = _make_frame(full, value=42.0)
        result = _align_to_index(frame, full, provider="p")
        assert (result["col"] == 42.0).all()
        assert result.dtypes.iloc[0] == np.dtype("float32")
        assert len(result) == 4

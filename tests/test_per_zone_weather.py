# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the per-zone weather feature (ADR feat/per-zone-weather-entsoe).

All tests are network-free: no Open-Meteo calls are made. The seam unit test
for get_target_data uses a synthetic zone_weather frame.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.manager.features import get_target_data
from spotforecast2_safe.weather import WeatherFetchError
from spotforecast2_safe.weather.locations import (
    GERMAN_LOAD_CENTERS,
    GERMAN_TSO_ZONE_CITIES,
    WeatherLocation,
    locations_for_zone,
)

# ---------------------------------------------------------------------------
# 1. Mapping / partition tests
# ---------------------------------------------------------------------------


class TestZoneCitiesMapping:
    """Verify GERMAN_TSO_ZONE_CITIES covers exactly the four expected zone keys."""

    EXPECTED_ZONES = {
        "load_50hertz",
        "load_amprion",
        "load_tennet",
        "load_transnetbw",
    }

    def test_exactly_four_zone_keys(self):
        assert set(GERMAN_TSO_ZONE_CITIES.keys()) == self.EXPECTED_ZONES

    def test_no_overlap_between_zones(self):
        all_cities: list[str] = []
        for cities in GERMAN_TSO_ZONE_CITIES.values():
            all_cities.extend(cities)
        assert len(all_cities) == len(
            set(all_cities)
        ), "A city appears in more than one TSO zone."

    def test_no_city_used_twice(self):
        seen: set[str] = set()
        for cities in GERMAN_TSO_ZONE_CITIES.values():
            for city in cities:
                assert city not in seen, f"City {city!r} appears in multiple zones."
                seen.add(city)

    def test_hamburg_in_50hertz(self):
        assert "Hamburg" in GERMAN_TSO_ZONE_CITIES["load_50hertz"]

    def test_bremen_in_tennet(self):
        assert "Bremen" in GERMAN_TSO_ZONE_CITIES["load_tennet"]

    def test_all_cities_in_registry(self):
        registry_names = {loc.name for loc in GERMAN_LOAD_CENTERS}
        for zone, cities in GERMAN_TSO_ZONE_CITIES.items():
            for city in cities:
                assert (
                    city in registry_names
                ), f"City {city!r} (zone {zone!r}) not in GERMAN_LOAD_CENTERS."

    def test_all_15_cities_assigned_to_exactly_one_zone(self):
        """Every city in the 15-city registry is assigned to exactly one zone."""
        all_zone_cities = [
            c for cities in GERMAN_TSO_ZONE_CITIES.values() for c in cities
        ]
        registry_names = {loc.name for loc in GERMAN_LOAD_CENTERS}
        # Full cover: all 15 registry cities are partitioned (Dortmund ∈ amprion).
        assert set(all_zone_cities) == registry_names
        assert len(all_zone_cities) == len(registry_names) == 15


class TestLocationsForZone:
    """Tests for the locations_for_zone resolver."""

    def test_50hertz_returns_four_locations(self):
        locs = locations_for_zone("load_50hertz")
        assert len(locs) == 4
        assert all(isinstance(loc, WeatherLocation) for loc in locs)

    def test_registry_weights_preserved(self):
        locs = locations_for_zone("load_50hertz")
        registry_index = {loc.name: loc for loc in GERMAN_LOAD_CENTERS}
        for loc in locs:
            assert loc.weight == registry_index[loc.name].weight

    def test_unknown_zone_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown TSO zone"):
            locations_for_zone("nonsense")

    def test_unknown_zone_error_lists_valid_zones(self):
        with pytest.raises(ValueError) as exc_info:
            locations_for_zone("load_unknown")
        msg = str(exc_info.value)
        for zone in GERMAN_TSO_ZONE_CITIES:
            assert zone in msg

    def test_override_dict_is_honoured(self):
        custom = [WeatherLocation("Custom", 50.0, 8.0, 100.0)]
        result = locations_for_zone("load_50hertz", override={"load_50hertz": custom})
        assert result == (WeatherLocation("Custom", 50.0, 8.0, 100.0),)

    def test_override_dict_missing_key_falls_back_to_registry(self):
        # Override only covers one zone; the other should still resolve.
        custom = [WeatherLocation("Custom", 50.0, 8.0, 100.0)]
        locs = locations_for_zone("load_amprion", override={"load_50hertz": custom})
        names = {loc.name for loc in locs}
        assert "Köln" in names  # expected registry city for amprion

    def test_deterministic_order_across_calls(self):
        locs1 = locations_for_zone("load_tennet")
        locs2 = locations_for_zone("load_tennet")
        assert locs1 == locs2

    def test_all_four_zones_resolve(self):
        for zone in GERMAN_TSO_ZONE_CITIES:
            locs = locations_for_zone(zone)
            assert len(locs) >= 1


# ---------------------------------------------------------------------------
# 2. Config validation guards
# ---------------------------------------------------------------------------


class TestConfigPerZoneWeatherGuards:
    """Verify validate_config raises on incompatible per_zone_weather settings."""

    def test_mutually_exclusive_with_population_weighted(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ConfigMulti(
                per_zone_weather=True,
                use_population_weighted_weather=True,
                use_exogenous_features=True,
            )

    def test_requires_exogenous_features(self):
        with pytest.raises(ValueError, match="requires exogenous features"):
            ConfigMulti(
                per_zone_weather=True,
                use_exogenous_features=False,
            )

    def test_incompatible_with_poly_degree_2(self):
        with pytest.raises(ValueError, match="polynomial interaction"):
            ConfigMulti(
                per_zone_weather=True,
                use_exogenous_features=True,
                poly_features_degree=2,
            )

    def test_zone_weather_locations_non_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="zone_weather_locations must be a dict"):
            ConfigMulti(
                per_zone_weather=True,
                use_exogenous_features=True,
                zone_weather_locations=123,  # type: ignore[arg-type]
            )

    def test_valid_construction_succeeds(self):
        cfg = ConfigMulti(
            per_zone_weather=True,
            use_exogenous_features=True,
        )
        assert cfg.per_zone_weather is True
        assert cfg.zone_weather_locations is None

    def test_valid_construction_with_override_dict(self):
        custom = [WeatherLocation("Custom", 50.0, 8.0, 100.0)]
        cfg = ConfigMulti(
            per_zone_weather=True,
            use_exogenous_features=True,
            zone_weather_locations={"load_50hertz": custom},
        )
        assert isinstance(cfg.zone_weather_locations, dict)

    def test_default_per_zone_weather_is_false(self):
        cfg = ConfigMulti()
        assert cfg.per_zone_weather is False
        assert cfg.zone_weather_locations is None


# ---------------------------------------------------------------------------
# 3. Seam unit test: get_target_data with zone_weather overwrite
# ---------------------------------------------------------------------------


def _make_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


class TestGetTargetDataZoneWeather:
    """Verify that zone_weather overwrites shared weather columns in-place."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.n_train = 168
        self.n_future = 24
        self.idx_train = _make_index(self.n_train)
        self.idx_future = _make_index(self.n_future, start="2024-01-08")

        # Shared weather values (shared baseline)
        shared_temp = rng.uniform(5.0, 20.0, self.n_train).astype("float32")
        hour_sin = np.sin(2 * np.pi * self.idx_train.hour / 24).astype("float32")

        self.df_pipeline = pd.DataFrame(
            {"load": rng.normal(100, 10, self.n_train)}, index=self.idx_train
        )

        self.data_with_exog = pd.DataFrame(
            {
                "load": self.df_pipeline["load"],
                "temperature_2m": shared_temp,
                "hour_sin": hour_sin,
            },
            index=self.idx_train,
        )

        shared_temp_future = rng.uniform(5.0, 20.0, self.n_future).astype("float32")
        hour_sin_future = np.sin(2 * np.pi * self.idx_future.hour / 24).astype(
            "float32"
        )
        self.exo_pred = pd.DataFrame(
            {
                "temperature_2m": shared_temp_future,
                "hour_sin": hour_sin_future,
            },
            index=self.idx_future,
        )

        # Per-zone weather: different temperature values
        zone_temp_train = rng.uniform(0.0, 5.0, self.n_train).astype("float32")
        zone_temp_future = rng.uniform(0.0, 5.0, self.n_future).astype("float32")
        # Deliberately different from the shared values above
        self.zone_weather_train = pd.DataFrame(
            {"temperature_2m": zone_temp_train}, index=self.idx_train
        )
        self.zone_weather_future = pd.DataFrame(
            {"temperature_2m": zone_temp_future}, index=self.idx_future
        )
        # Combined zone_weather covering both train and future
        self.zone_weather = pd.concat(
            [self.zone_weather_train, self.zone_weather_future]
        )

        self.config = ConfigMulti(
            targets=["load"],
            use_exogenous_features=True,
        )
        self.start_ts = self.idx_train[0]
        self.end_ts = self.idx_train[-1]

    def test_zone_temperature_overwrites_shared(self):
        y_train, exog_train, exog_future = get_target_data(
            target="load",
            df_pipeline=self.df_pipeline,
            config=self.config,
            data_with_exog=self.data_with_exog,
            exog_feature_names=["temperature_2m", "hour_sin"],
            exo_pred=self.exo_pred,
            zone_weather=self.zone_weather,
            start_train_ts=self.start_ts,
            end_train_ts=self.end_ts,
        )

        expected_train = (
            self.zone_weather["temperature_2m"]
            .reindex(self.idx_train)
            .astype("float32")
        )
        expected_future = (
            self.zone_weather["temperature_2m"]
            .reindex(self.idx_future)
            .astype("float32")
        )

        pd.testing.assert_series_equal(
            exog_train["temperature_2m"].rename(None),
            expected_train.rename(None),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            exog_future["temperature_2m"].rename(None),
            expected_future.rename(None),
            check_names=False,
        )

    def test_non_weather_column_unchanged(self):
        _, exog_train, exog_future = get_target_data(
            target="load",
            df_pipeline=self.df_pipeline,
            config=self.config,
            data_with_exog=self.data_with_exog,
            exog_feature_names=["temperature_2m", "hour_sin"],
            exo_pred=self.exo_pred,
            zone_weather=self.zone_weather,
            start_train_ts=self.start_ts,
            end_train_ts=self.end_ts,
        )

        expected_sin = (
            self.data_with_exog["hour_sin"].reindex(self.idx_train).astype("float32")
        )
        pd.testing.assert_series_equal(
            exog_train["hour_sin"].rename(None),
            expected_sin.rename(None),
            check_names=False,
        )

    def test_column_not_in_zone_weather_is_left_as_is(self):
        """A column in exog_feature_names but NOT in zone_weather stays unchanged."""
        zone_weather_partial = self.zone_weather[[]].copy()  # empty columns
        _, exog_train, _ = get_target_data(
            target="load",
            df_pipeline=self.df_pipeline,
            config=self.config,
            data_with_exog=self.data_with_exog,
            exog_feature_names=["temperature_2m", "hour_sin"],
            exo_pred=self.exo_pred,
            zone_weather=zone_weather_partial,
            start_train_ts=self.start_ts,
            end_train_ts=self.end_ts,
        )
        expected_temp = (
            self.data_with_exog["temperature_2m"]
            .reindex(self.idx_train)
            .astype("float32")
        )
        pd.testing.assert_series_equal(
            exog_train["temperature_2m"].rename(None),
            expected_temp.rename(None),
            check_names=False,
        )

    def test_shape_and_columns_unchanged(self):
        feature_names = ["temperature_2m", "hour_sin"]
        _, exog_train, exog_future = get_target_data(
            target="load",
            df_pipeline=self.df_pipeline,
            config=self.config,
            data_with_exog=self.data_with_exog,
            exog_feature_names=feature_names,
            exo_pred=self.exo_pred,
            zone_weather=self.zone_weather,
            start_train_ts=self.start_ts,
            end_train_ts=self.end_ts,
        )
        assert list(exog_train.columns) == feature_names
        assert list(exog_future.columns) == feature_names
        assert exog_train.shape == (self.n_train, 2)
        assert exog_future.shape == (self.n_future, 2)

    def test_dtype_float32(self):
        _, exog_train, exog_future = get_target_data(
            target="load",
            df_pipeline=self.df_pipeline,
            config=self.config,
            data_with_exog=self.data_with_exog,
            exog_feature_names=["temperature_2m", "hour_sin"],
            exo_pred=self.exo_pred,
            zone_weather=self.zone_weather,
            start_train_ts=self.start_ts,
            end_train_ts=self.end_ts,
        )
        assert exog_train["temperature_2m"].dtype == np.float32
        assert exog_future["temperature_2m"].dtype == np.float32

    def test_zone_weather_none_uses_shared(self):
        """Without zone_weather, the shared values are returned unchanged."""
        _, exog_train_z, _ = get_target_data(
            target="load",
            df_pipeline=self.df_pipeline,
            config=self.config,
            data_with_exog=self.data_with_exog,
            exog_feature_names=["temperature_2m", "hour_sin"],
            exo_pred=self.exo_pred,
            zone_weather=None,
            start_train_ts=self.start_ts,
            end_train_ts=self.end_ts,
        )
        expected_temp = (
            self.data_with_exog["temperature_2m"]
            .reindex(self.idx_train)
            .astype("float32")
        )
        pd.testing.assert_series_equal(
            exog_train_z["temperature_2m"].rename(None),
            expected_temp.rename(None),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


class TestLocationsForZoneDeterminism:
    def test_two_calls_identical(self):
        locs_a = locations_for_zone("load_transnetbw")
        locs_b = locations_for_zone("load_transnetbw")
        assert locs_a == locs_b

    def test_all_zones_deterministic(self):
        for zone in GERMAN_TSO_ZONE_CITIES:
            assert locations_for_zone(zone) == locations_for_zone(zone)


# ---------------------------------------------------------------------------
# 5. Regression-when-off: zone_weather_aligned empty without exog features
# ---------------------------------------------------------------------------


class TestRegressionPerZoneOff:
    """Without per_zone_weather, zone_weather_aligned stays empty."""

    def test_zone_weather_aligned_empty_without_exog(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(
                targets=["load"],
                use_exogenous_features=False,
                cache_home=tmp,
            )
            from spotforecast2_safe.multitask import LazyTask

            rng = np.random.default_rng(0)
            idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
            df = pd.DataFrame({"load": rng.normal(100, 5, 200)}, index=idx)
            task = LazyTask(cfg, dataframe=df)
            task.prepare_data().detect_outliers().impute().build_exogenous_features()
            assert task.zone_weather_aligned == {}

    def test_per_zone_weather_false_default(self):
        cfg = ConfigMulti(use_exogenous_features=False)
        assert cfg.per_zone_weather is False


# ---------------------------------------------------------------------------
# 6. Pipeline integration: per-zone frames cover the forecast horizon and
#    distinct zone weather reaches each target's exog_future (network-free).
#    This guards the regression where the per-zone frame was truncated to
#    df_pipeline.index (historical only), NaN-ing out the future weather.
# ---------------------------------------------------------------------------


def _zone_load_df(periods: int = 24 * 21) -> pd.DataFrame:
    """Two-zone hourly load frame whose columns are valid TSO zone keys."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=periods, freq="h", tz="UTC")
    idx.name = "DateTime"
    return pd.DataFrame(
        {
            "load_50hertz": rng.normal(11000, 300, periods),
            "load_tennet": rng.normal(25000, 600, periods),
        },
        index=idx,
    )


def _fake_weather_factory():
    """Return a get_weather_features stand-in: zone-distinct, time-varying temp.

    Indexed over the requested ``[start, cov_end]`` (which spans the forecast
    horizon), exactly like the real fetch. The temperature offset is derived
    from the mean latitude of ``locations`` so each zone gets a different
    series; the global baseline call (``locations is None``) gets its own.
    """

    def fake_weather(*, data, start, cov_end, locations=None, **kwargs):
        idx = pd.date_range(start, cov_end, freq="h")
        if locations is None:
            offset = 50.0  # global baseline sentinel
        else:
            offset = float(sum(lat for lat, _ in locations) / len(locations))
        temp = offset + np.sin(2 * np.pi * idx.hour / 24)
        weather_aligned = pd.DataFrame({"temperature_2m": temp}, index=idx)
        # Real WindowFeatures returns the raw columns PLUS window columns; with
        # weather windows off only the raw column is selected, so the raw column
        # alone reproduces the relevant shape (weather_features == raw here).
        weather_features = weather_aligned.copy()
        return weather_features, weather_aligned

    return fake_weather


class TestPerZonePipelineIntegration:
    """build_exogenous_features wires distinct per-zone weather over the horizon."""

    def _build_task(self, monkeypatch, tmp_path):
        import spotforecast2_safe.multitask.base as mt_base
        from spotforecast2_safe.multitask import LazyTask

        monkeypatch.setattr(mt_base, "get_weather_features", _fake_weather_factory())
        cfg = ConfigMulti(
            targets=["load_50hertz", "load_tennet"],
            predict_size=6,
            use_exogenous_features=True,
            per_zone_weather=True,
            on_weather_failure="raise",
            use_outlier_detection=False,
            auto_save_models=False,
            number_folds=2,
            verbose=False,
            cache_home=str(tmp_path),
        )
        task = LazyTask(cfg, dataframe=_zone_load_df())
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        return task

    def test_zone_frames_cover_forecast_horizon(self, monkeypatch, tmp_path):
        task = self._build_task(monkeypatch, tmp_path)
        assert set(task.zone_weather_aligned) == {"load_50hertz", "load_tennet"}
        cov_end = task.run_state.cov_end
        data_end = task.run_state.data_end
        assert cov_end > data_end  # horizon really extends past history
        for zone, frame in task.zone_weather_aligned.items():
            # The frame must span the horizon, not just the training history.
            assert frame.index.max() == cov_end, (
                f"{zone}: per-zone weather truncated to {frame.index.max()} "
                f"(history end {data_end}); must reach cov_end {cov_end}."
            )

    def test_future_exog_is_per_zone_and_not_nan(self, monkeypatch, tmp_path):
        task = self._build_task(monkeypatch, tmp_path)
        assert "temperature_2m" in task.exog_feature_names
        future_means = {}
        for zone in ("load_50hertz", "load_tennet"):
            _, _, exog_future = task._get_target_data(zone)
            temp = exog_future["temperature_2m"]
            assert temp.notna().all(), f"{zone}: future weather has NaN"
            future_means[zone] = float(temp.mean())
        # Distinct zones -> distinct regional temperature (different offsets).
        assert abs(future_means["load_50hertz"] - future_means["load_tennet"]) > 0.5

    def test_skip_partial_zone_failure_degrades_to_no_weather(
        self, monkeypatch, tmp_path
    ):
        """One zone failing under 'skip' degrades the WHOLE pipeline to no-weather."""
        import spotforecast2_safe.multitask.base as mt_base
        from spotforecast2_safe.multitask import LazyTask

        base_fake = _fake_weather_factory()

        def failing_for_tennet(*, data, start, cov_end, locations=None, **kwargs):
            # load_tennet includes München (lat 48.1351); fail just that zone.
            if locations is not None and any(
                abs(lat - 48.1351) < 0.01 for lat, _ in locations
            ):
                raise WeatherFetchError("simulated Open-Meteo outage for tennet")
            return base_fake(
                data=data, start=start, cov_end=cov_end, locations=locations, **kwargs
            )

        monkeypatch.setattr(mt_base, "get_weather_features", failing_for_tennet)
        cfg = ConfigMulti(
            targets=["load_50hertz", "load_tennet"],
            predict_size=6,
            use_exogenous_features=True,
            per_zone_weather=True,
            on_weather_failure="skip",
            use_outlier_detection=False,
            auto_save_models=False,
            number_folds=2,
            verbose=False,
            cache_home=str(tmp_path),
        )
        task = LazyTask(cfg, dataframe=_zone_load_df())
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        # Fail-safe degradation: no per-zone frames, no weather columns selected.
        assert task.zone_weather_aligned == {}
        assert "temperature_2m" not in task.exog_feature_names

    def test_multicity_combine_mismatch_raises_weatherfetcherror(self, monkeypatch):
        """A transient per-city fetch (mismatched index) must surface as a
        CATCHABLE WeatherFetchError, not a bare ValueError that escapes
        on_weather_failure='skip' (the 22.4.0 per-zone crash, fixed in 22.4.1)."""
        from spotforecast2_safe.weather import WeatherFetchError, get_weather_features

        start = pd.Timestamp("2024-01-01", tz="UTC")
        cov_end = pd.Timestamp("2024-01-06", tz="UTC")
        full = pd.date_range(start, cov_end, freq="h")

        def fake_fetch(*, cov_start, cov_end, latitude, longitude, **kwargs):
            # The northern city (lat ~53) returns a SHORTER range, mimicking a
            # 429 fallback that doesn't cover the full window -> mismatched index.
            idx = full[:-12] if latitude > 53.0 else full
            return pd.DataFrame(
                {"temperature_2m": np.arange(len(idx), dtype="float64")}, index=idx
            )

        # get_weather_features imports fetch_weather_data lazily from this module.
        # The data package re-exports a `fetch_data` function that shadows the
        # submodule name, so fetch the real module object from sys.modules.
        import sys

        fd_mod = sys.modules["spotforecast2_safe.data.fetch_data"]
        monkeypatch.setattr(fd_mod, "fetch_weather_data", fake_fetch)
        ref = pd.DataFrame({"load": np.zeros(len(full))}, index=full)
        with pytest.raises(WeatherFetchError, match="Multi-city"):
            get_weather_features(
                data=ref,
                start=start,
                cov_end=cov_end,
                forecast_horizon=24,
                locations=[(52.52, 13.40), (53.55, 9.99)],
                location_weights=[1.0, 1.0],
            )

    def test_throttle_spaces_open_meteo_requests(self, monkeypatch):
        """_throttle_open_meteo sleeps to maintain the minimum request spacing."""
        import spotforecast2_safe.weather.client as wc

        monkeypatch.setattr(wc, "_MIN_REQUEST_INTERVAL_S", 0.5)
        monkeypatch.setattr(wc, "_LAST_REQUEST_MONOTONIC", 0.0)
        clock = [1000.0]
        sleeps: list[float] = []
        monkeypatch.setattr(wc, "monotonic", lambda: clock[0])
        monkeypatch.setattr(wc, "sleep", lambda s: sleeps.append(s))

        wc._throttle_open_meteo()  # first call after a long idle -> no sleep
        assert sleeps == []
        wc._throttle_open_meteo()  # immediate second call -> must wait ~0.5 s
        assert len(sleeps) == 1 and abs(sleeps[0] - 0.5) < 1e-9

    def test_throttle_noop_when_disabled(self, monkeypatch):
        import spotforecast2_safe.weather.client as wc

        sleeps: list[float] = []
        monkeypatch.setattr(wc, "_MIN_REQUEST_INTERVAL_S", 0.0)
        monkeypatch.setattr(wc, "sleep", lambda s: sleeps.append(s))
        wc._throttle_open_meteo()
        wc._throttle_open_meteo()
        assert sleeps == []

    def test_non_zone_target_raises_value_error(self, monkeypatch, tmp_path):
        """per_zone_weather=True with a target that is not a TSO zone must raise."""
        import spotforecast2_safe.multitask.base as mt_base
        from spotforecast2_safe.multitask import LazyTask

        monkeypatch.setattr(mt_base, "get_weather_features", _fake_weather_factory())
        rng = np.random.default_rng(3)
        idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
        idx.name = "DateTime"
        df = pd.DataFrame({"load": rng.normal(100, 5, len(idx))}, index=idx)
        cfg = ConfigMulti(
            targets=["load"],  # not a GERMAN_TSO_ZONE_CITIES key
            predict_size=6,
            use_exogenous_features=True,
            per_zone_weather=True,
            use_outlier_detection=False,
            auto_save_models=False,
            number_folds=2,
            verbose=False,
            cache_home=str(tmp_path),
        )
        task = LazyTask(cfg, dataframe=df)
        with pytest.raises(ValueError, match="Unknown TSO zone"):
            task.prepare_data().detect_outliers().impute().build_exogenous_features()

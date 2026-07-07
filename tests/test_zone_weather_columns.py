# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``zone_weather_columns`` opt-in knob (ADR feat/zone-weather-columns).

All tests are network-free: no Open-Meteo calls are made. ``get_weather_features``
is monkeypatched, mirroring the pattern in ``tests/test_per_zone_weather.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spotforecast2_safe.weather.zone_columns as zc
from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.manager.features import select_exogenous_features
from spotforecast2_safe.weather import WeatherFetchError
from spotforecast2_safe.weather.locations import GERMAN_TSO_ZONE_CITIES

ZONE_ORDER = tuple(GERMAN_TSO_ZONE_CITIES)  # deterministic dict order
ZONE_SHORTS = tuple(z.removeprefix("load_") for z in ZONE_ORDER)


def _fake_weather_factory():
    """Return a ``get_weather_features`` stand-in: zone-distinct, time-varying.

    Mirrors ``tests/test_per_zone_weather.py::_fake_weather_factory``: the
    frame is indexed over the requested ``[start, cov_end]`` window (spans the
    forecast horizon) and the offset is derived from the mean latitude of
    ``locations`` so each zone gets a distinguishable series. Two raw columns
    are returned (``temperature_2m`` and ``relative_humidity_2m``) so the
    "N raw cols per zone" shape is exercised with N > 1.
    """

    def fake_weather(*, data, start, cov_end, locations=None, **kwargs):
        idx = pd.date_range(start, cov_end, freq="h")
        offset = float(sum(lat for lat, _ in locations) / len(locations))
        temp = offset + np.sin(2 * np.pi * idx.hour / 24)
        humidity = offset * 0.5 + np.cos(2 * np.pi * idx.hour / 24)
        weather_aligned = pd.DataFrame(
            {"temperature_2m": temp, "relative_humidity_2m": humidity}, index=idx
        )
        weather_features = weather_aligned.copy()
        return weather_features, weather_aligned

    return fake_weather


# ---------------------------------------------------------------------------
# (a) suffix/concat: 4 zones x N raw cols, deterministic order, all present
# ---------------------------------------------------------------------------


class TestBuildZoneWeatherColumns:
    def _call(self, monkeypatch):
        monkeypatch.setattr(zc, "get_weather_features", _fake_weather_factory())
        idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
        ref = pd.DataFrame({"load": range(len(idx))}, index=idx)
        return zc.build_zone_weather_columns(
            data=ref,
            start=idx[0],
            cov_end=idx[-1],
            forecast_horizon=6,
            latitude=51.5,
            longitude=7.5,
            timezone="UTC",
        )

    def test_four_zones_times_two_raw_columns(self, monkeypatch):
        wf, wa = self._call(monkeypatch)
        assert wa.shape[1] == 4 * 2
        assert wf.shape[1] == 4 * 2

    def test_all_suffixed_columns_present_and_ordered(self, monkeypatch):
        _, wa = self._call(monkeypatch)
        expected = [
            f"{col}__{short}"
            for short in ZONE_SHORTS
            for col in ("temperature_2m", "relative_humidity_2m")
        ]
        assert list(wa.columns) == expected

    def test_deterministic_zone_order(self, monkeypatch):
        wf1, wa1 = self._call(monkeypatch)
        wf2, wa2 = self._call(monkeypatch)
        assert list(wa1.columns) == list(wa2.columns)
        pd.testing.assert_frame_equal(wa1, wa2)

    def test_distinct_values_per_zone(self, monkeypatch):
        _, wa = self._call(monkeypatch)
        # Different zones have different mean latitudes -> distinct series.
        temp_50hertz = wa["temperature_2m__50hertz"]
        temp_transnetbw = wa["temperature_2m__transnetbw"]
        assert not temp_50hertz.equals(temp_transnetbw)

    def test_index_spans_forecast_horizon(self, monkeypatch):
        monkeypatch.setattr(zc, "get_weather_features", _fake_weather_factory())
        idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
        ref = pd.DataFrame({"load": range(len(idx))}, index=idx)
        data_end = idx[65]
        cov_end = idx[-1]
        wf, wa = zc.build_zone_weather_columns(
            data=ref,
            start=idx[0],
            cov_end=cov_end,
            forecast_horizon=6,
            latitude=51.5,
            longitude=7.5,
        )
        assert wa.index.max() == cov_end
        assert cov_end > data_end

    def test_any_zone_failure_raises_weatherfetcherror(self, monkeypatch):
        base_fake = _fake_weather_factory()

        def failing_for_tennet(*, data, start, cov_end, locations=None, **kwargs):
            # tennet includes München (lat 48.1351); fail just that zone.
            if any(abs(lat - 48.1351) < 0.01 for lat, _ in locations):
                raise WeatherFetchError("simulated Open-Meteo outage for tennet")
            return base_fake(
                data=data, start=start, cov_end=cov_end, locations=locations, **kwargs
            )

        monkeypatch.setattr(zc, "get_weather_features", failing_for_tennet)
        idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
        ref = pd.DataFrame({"load": range(len(idx))}, index=idx)
        with pytest.raises(WeatherFetchError, match="tennet"):
            zc.build_zone_weather_columns(
                data=ref,
                start=idx[0],
                cov_end=idx[-1],
                forecast_horizon=6,
                latitude=51.5,
                longitude=7.5,
            )

    def test_override_locations_honoured(self, monkeypatch):
        from spotforecast2_safe.weather.locations import WeatherLocation

        seen_lats: list[float] = []

        def recording_fake(*, data, start, cov_end, locations=None, **kwargs):
            seen_lats.append(locations[0][0])
            return base_fake(
                data=data, start=start, cov_end=cov_end, locations=locations, **kwargs
            )

        base_fake = _fake_weather_factory()
        monkeypatch.setattr(zc, "get_weather_features", recording_fake)
        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        ref = pd.DataFrame({"load": range(len(idx))}, index=idx)
        custom = [WeatherLocation("Custom", 12.34, 5.67, 100.0)]
        zc.build_zone_weather_columns(
            data=ref,
            start=idx[0],
            cov_end=idx[-1],
            forecast_horizon=6,
            latitude=51.5,
            longitude=7.5,
            zone_locations_override={"load_50hertz": custom},
        )
        assert 12.34 in seen_lats


# ---------------------------------------------------------------------------
# (b) selection: every *__<zone> passes select_exogenous_features, cyclical
# encoding / windows do not mangle the suffixed names
# ---------------------------------------------------------------------------


class TestSelectionCompatibility:
    def test_all_suffixed_raw_columns_selected(self, monkeypatch):
        monkeypatch.setattr(zc, "get_weather_features", _fake_weather_factory())
        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        ref = pd.DataFrame({"load": range(len(idx))}, index=idx)
        _, weather_aligned = zc.build_zone_weather_columns(
            data=ref,
            start=idx[0],
            cov_end=idx[-1],
            forecast_horizon=6,
            latitude=51.5,
            longitude=7.5,
        )
        # exogenous_features == weather_aligned here (no calendar/holiday cols)
        selected = select_exogenous_features(
            exogenous_features=weather_aligned,
            weather_aligned=weather_aligned,
        )
        assert set(selected) == set(weather_aligned.columns)
        assert len(selected) == 8

    def test_suffixed_names_never_match_cyclical_regex(self):
        for short in ZONE_SHORTS:
            for col in ("temperature_2m", "relative_humidity_2m"):
                name = f"{col}__{short}"
                assert not name.endswith("_sin") and not name.endswith("_cos")

    def test_weather_window_columns_still_selected_when_suffixed(self):
        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        weather_aligned = pd.DataFrame(
            {"temperature_2m__50hertz": range(24)}, index=idx
        )
        exogenous = weather_aligned.copy()
        exogenous["temperature_2m__50hertz_window_7D_mean"] = 1.0
        selected = select_exogenous_features(
            exogenous_features=exogenous,
            weather_aligned=weather_aligned,
            include_weather_windows=True,
        )
        assert "temperature_2m__50hertz" in selected
        assert "temperature_2m__50hertz_window_7D_mean" in selected


# ---------------------------------------------------------------------------
# (c) mutual-exclusion validation
# ---------------------------------------------------------------------------


class TestConfigZoneWeatherColumnsGuards:
    def test_mutually_exclusive_with_per_zone_weather(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ConfigMulti(zone_weather_columns=True, per_zone_weather=True)

    def test_mutually_exclusive_with_population_weighted(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ConfigMulti(zone_weather_columns=True, use_population_weighted_weather=True)

    def test_requires_exogenous_features(self):
        with pytest.raises(ValueError, match="requires exogenous features"):
            ConfigMulti(zone_weather_columns=True, use_exogenous_features=False)

    def test_incompatible_with_poly_degree_2(self):
        with pytest.raises(ValueError, match="polynomial interaction"):
            ConfigMulti(zone_weather_columns=True, poly_features_degree=2)

    def test_zone_weather_locations_non_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="zone_weather_locations must be a dict"):
            ConfigMulti(
                zone_weather_columns=True,
                zone_weather_locations=123,  # type: ignore[arg-type]
            )

    def test_valid_construction_succeeds(self):
        cfg = ConfigMulti(zone_weather_columns=True)
        assert cfg.zone_weather_columns is True
        assert cfg.poly_features_degree == 1

    def test_default_is_false(self):
        cfg = ConfigMulti()
        assert cfg.zone_weather_columns is False

    def test_raises_via_set_params_too(self):
        cfg = ConfigMulti()
        with pytest.raises(ValueError, match="mutually exclusive"):
            cfg.set_params(zone_weather_columns=True, per_zone_weather=True)


# ---------------------------------------------------------------------------
# (d) skip-degradation: one zone failure under on_weather_failure="skip"
# degrades the WHOLE weather block, never mixing a partial zone set.
# (e) poly=1 (the only value compatible with zone_weather_columns) honoured:
# no poly_ columns are created from the 8+ suffixed weather columns.
# ---------------------------------------------------------------------------


def _single_target_df(periods: int = 24 * 10) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-01", periods=periods, freq="h", tz="UTC")
    idx.name = "DateTime"
    return pd.DataFrame({"load": rng.normal(50000, 2000, periods)}, index=idx)


def _failing_for_tennet_factory():
    """Like ``_fake_weather_factory`` but raises for the tennet zone.

    Identifies the tennet fetch by München's latitude (48.1351), which is
    unique to that zone's city list.
    """
    base_fake = _fake_weather_factory()

    def failing_for_tennet(*, data, start, cov_end, locations=None, **kwargs):
        if any(abs(lat - 48.1351) < 0.01 for lat, _ in locations):
            raise WeatherFetchError("simulated Open-Meteo outage for tennet")
        return base_fake(
            data=data, start=start, cov_end=cov_end, locations=locations, **kwargs
        )

    return failing_for_tennet


class TestZoneWeatherColumnsPipelineIntegration:
    def _build_task(
        self, monkeypatch, tmp_path, *, on_weather_failure="raise", fetch_fn=None
    ):
        from spotforecast2_safe.multitask import LazyTask

        monkeypatch.setattr(
            zc, "get_weather_features", fetch_fn or _fake_weather_factory()
        )
        cfg = ConfigMulti(
            targets=["load"],
            predict_size=6,
            use_exogenous_features=True,
            zone_weather_columns=True,
            on_weather_failure=on_weather_failure,
            use_outlier_detection=False,
            auto_save_models=False,
            number_folds=2,
            verbose=False,
            cache_home=str(tmp_path),
        )
        task = LazyTask(cfg, dataframe=_single_target_df())
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        return task

    def test_suffixed_columns_reach_exog_feature_names(self, monkeypatch, tmp_path):
        task = self._build_task(monkeypatch, tmp_path)
        for short in ZONE_SHORTS:
            assert f"temperature_2m__{short}" in task.exog_feature_names
            assert f"relative_humidity_2m__{short}" in task.exog_feature_names

    def test_no_poly_columns_created_with_default_poly_degree(
        self, monkeypatch, tmp_path
    ):
        task = self._build_task(monkeypatch, tmp_path)
        assert task.config.poly_features_degree == 1
        poly_cols = [
            c for c in task.exogenous_features.columns if c.startswith("poly_")
        ]
        assert poly_cols == []

    def test_skip_partial_zone_failure_degrades_whole_block(
        self, monkeypatch, tmp_path
    ):
        """One zone failing under 'skip' degrades the WHOLE block, not a
        partial mix of 3 real zones + 1 missing zone."""
        task = self._build_task(
            monkeypatch,
            tmp_path,
            on_weather_failure="skip",
            fetch_fn=_failing_for_tennet_factory(),
        )

        for short in ZONE_SHORTS:
            assert f"temperature_2m__{short}" not in task.exog_feature_names
        assert task.weather_aligned.shape[1] == 0

    def test_skip_partial_zone_failure_raise_mode_propagates(
        self, monkeypatch, tmp_path
    ):
        with pytest.raises(WeatherFetchError):
            self._build_task(
                monkeypatch,
                tmp_path,
                on_weather_failure="raise",
                fetch_fn=_failing_for_tennet_factory(),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

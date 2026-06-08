# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for derived weather features and population-weighted aggregation.

Covers the pure-function contract of
``spotforecast2_safe.weather.derived``: numerical correctness, determinism
(identical input → identical output), and the fail-safe rule (an input that
cannot be transformed raises ``ValueError`` rather than being silently
repaired).
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.weather.derived import (
    DERIVED_FEATURE_KEYS,
    add_derived_weather_features,
    apparent_temperature,
    cooling_degree_hours,
    dew_point,
    heating_degree_hours,
    population_weighted_average,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")


class TestDegreeHours:
    """Heating/cooling degree-hours split the U-shaped response into two arms."""

    def test_heating_degree_hours_values_and_base(self):
        t = pd.Series([5.0, 15.0, 25.0])
        assert heating_degree_hours(t, base=15.0).tolist() == [10.0, 0.0, 0.0]
        # A higher base lifts the heating demand everywhere below it.
        assert heating_degree_hours(t, base=18.0).tolist() == [13.0, 3.0, 0.0]

    def test_cooling_degree_hours_values_and_base(self):
        t = pd.Series([5.0, 22.0, 30.0])
        assert cooling_degree_hours(t, base=22.0).tolist() == [0.0, 0.0, 8.0]

    def test_names(self):
        t = pd.Series([1.0])
        assert heating_degree_hours(t).name == "hdh"
        assert cooling_degree_hours(t).name == "cdh"

    def test_nan_input_raises(self):
        t = pd.Series([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            heating_degree_hours(t)
        with pytest.raises(ValueError, match="NaN"):
            cooling_degree_hours(t)

    def test_non_series_raises(self):
        with pytest.raises(ValueError, match="Series"):
            heating_degree_hours([1.0, 2.0])  # type: ignore[arg-type]

    def test_deterministic(self):
        t = pd.Series([3.0, 17.0, 28.0])
        pd.testing.assert_series_equal(heating_degree_hours(t), heating_degree_hours(t))


class TestDewPoint:
    """Magnus-Tetens dew point folds humidity into a temperature."""

    def test_saturation_equals_temperature(self):
        t = pd.Series([20.0, 0.0])
        rh = pd.Series([100.0, 100.0])
        out = dew_point(t, rh)
        # At 100% RH, dew point ≈ air temperature.
        assert abs(out.iloc[0] - 20.0) < 0.05
        assert abs(out.iloc[1] - 0.0) < 0.05

    def test_half_humidity_known_value(self):
        out = dew_point(pd.Series([20.0]), pd.Series([50.0]))
        assert abs(out.iloc[0] - 9.26) < 0.1

    def test_index_mismatch_raises(self):
        t = pd.Series([20.0], index=[pd.Timestamp("2023-01-01", tz="UTC")])
        rh = pd.Series([50.0], index=[pd.Timestamp("2023-01-02", tz="UTC")])
        with pytest.raises(ValueError, match="share an index"):
            dew_point(t, rh)

    def test_humidity_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            dew_point(pd.Series([20.0]), pd.Series([150.0]))

    def test_zero_humidity_is_finite(self):
        out = dew_point(pd.Series([20.0]), pd.Series([0.0]))
        assert np.isfinite(out.iloc[0])


class TestApparentTemperature:
    """Steadman apparent temperature captures the humidity load driver."""

    def test_known_value_ms(self):
        out = apparent_temperature(
            pd.Series([30.0]), pd.Series([70.0]), pd.Series([2.0])
        )
        assert abs(out.iloc[0] - 34.37) < 0.1

    def test_wind_unit_conversion(self):
        # 7.2 km/h == 2.0 m/s, so the two calls must agree.
        ms = apparent_temperature(
            pd.Series([30.0]), pd.Series([70.0]), pd.Series([2.0]), wind_speed_unit="ms"
        )
        kmh = apparent_temperature(
            pd.Series([30.0]),
            pd.Series([70.0]),
            pd.Series([7.2]),
            wind_speed_unit="kmh",
        )
        assert abs(ms.iloc[0] - kmh.iloc[0]) < 1e-9

    def test_bad_unit_raises(self):
        with pytest.raises(ValueError, match="wind_speed_unit"):
            apparent_temperature(
                pd.Series([30.0]),
                pd.Series([70.0]),
                pd.Series([2.0]),
                wind_speed_unit="mph",
            )

    def test_index_mismatch_raises(self):
        t = pd.Series([30.0], index=[0])
        rh = pd.Series([70.0], index=[0])
        w = pd.Series([2.0], index=[1])
        with pytest.raises(ValueError, match="share an index"):
            apparent_temperature(t, rh, w)


class TestPopulationWeightedAverage:
    """Combine per-city frames into one demand-weighted national index."""

    def test_weighted_value(self):
        idx = _idx(3)
        a = pd.DataFrame({"temperature_2m": [10.0] * 3}, index=idx)
        b = pd.DataFrame({"temperature_2m": [20.0] * 3}, index=idx)
        out = population_weighted_average([a, b], [3.0, 1.0])
        # (3*10 + 1*20) / 4 == 12.5
        assert out["temperature_2m"].tolist() == [12.5, 12.5, 12.5]

    def test_scale_invariance(self):
        idx = _idx(2)
        a = pd.DataFrame({"x": [10.0, 10.0]}, index=idx)
        b = pd.DataFrame({"x": [20.0, 20.0]}, index=idx)
        out1 = population_weighted_average([a, b], [3.0, 1.0])
        out2 = population_weighted_average([a, b], [300.0, 100.0])
        pd.testing.assert_frame_equal(out1, out2)

    def test_preserves_column_order(self):
        idx = _idx(2)
        cols = ["temperature_2m", "wind_speed_10m", "relative_humidity_2m"]
        a = pd.DataFrame({c: [1.0, 1.0] for c in cols}, index=idx)[cols]
        b = pd.DataFrame({c: [3.0, 3.0] for c in cols}, index=idx)[cols]
        out = population_weighted_average([a, b], [1.0, 1.0])
        assert list(out.columns) == cols

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one frame"):
            population_weighted_average([], [])

    def test_length_mismatch_raises(self):
        idx = _idx(2)
        a = pd.DataFrame({"x": [1.0, 1.0]}, index=idx)
        with pytest.raises(ValueError, match="length mismatch"):
            population_weighted_average([a], [1.0, 2.0])

    def test_negative_weight_raises(self):
        idx = _idx(2)
        a = pd.DataFrame({"x": [1.0, 1.0]}, index=idx)
        b = pd.DataFrame({"x": [2.0, 2.0]}, index=idx)
        with pytest.raises(ValueError, match="non-negative"):
            population_weighted_average([a, b], [1.0, -1.0])

    def test_zero_sum_raises(self):
        idx = _idx(2)
        a = pd.DataFrame({"x": [1.0, 1.0]}, index=idx)
        with pytest.raises(ValueError, match="positive value"):
            population_weighted_average([a], [0.0])

    def test_index_mismatch_raises(self):
        a = pd.DataFrame({"x": [1.0]}, index=_idx(1))
        b = pd.DataFrame(
            {"x": [2.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"),
        )
        with pytest.raises(ValueError, match="different index"):
            population_weighted_average([a, b], [1.0, 1.0])

    def test_column_mismatch_raises(self):
        idx = _idx(1)
        a = pd.DataFrame({"x": [1.0]}, index=idx)
        b = pd.DataFrame({"y": [2.0]}, index=idx)
        with pytest.raises(ValueError, match="different columns"):
            population_weighted_average([a, b], [1.0, 1.0])


class TestAddDerivedWeatherFeatures:
    """The orchestrator appends derived columns deterministically, fail-safe."""

    def _weather(self) -> pd.DataFrame:
        idx = _idx(4)
        return pd.DataFrame(
            {
                "temperature_2m": [28.0, 30.0, 12.0, 8.0],
                "relative_humidity_2m": [60.0, 55.0, 80.0, 90.0],
                "wind_speed_10m": [7.2, 10.8, 3.6, 1.8],  # km/h
            },
            index=idx,
        )

    def test_adds_all_four_columns(self):
        out = add_derived_weather_features(
            self._weather(), ["hdh", "cdh", "apparent_temperature", "dew_point"]
        )
        added = set(out.columns) - set(self._weather().columns)
        assert added == {"hdh", "cdh", "apparent_temperature", "dew_point"}

    def test_original_columns_preserved(self):
        w = self._weather()
        out = add_derived_weather_features(w, ["hdh"])
        for c in w.columns:
            assert c in out.columns
        pd.testing.assert_series_equal(out["temperature_2m"], w["temperature_2m"])

    def test_canonical_column_order_independent_of_request_order(self):
        w = self._weather()
        out_a = add_derived_weather_features(w, ["dew_point", "hdh"])
        out_b = add_derived_weather_features(w, ["hdh", "dew_point"])
        # Derived columns always appear in DERIVED_FEATURE_KEYS order.
        derived_a = [c for c in out_a.columns if c in DERIVED_FEATURE_KEYS]
        derived_b = [c for c in out_b.columns if c in DERIVED_FEATURE_KEYS]
        assert derived_a == derived_b == ["hdh", "dew_point"]

    def test_empty_features_returns_copy(self):
        w = self._weather()
        out = add_derived_weather_features(w, [])
        pd.testing.assert_frame_equal(out, w)
        assert out is not w  # a copy, not the same object

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="unknown derived feature"):
            add_derived_weather_features(self._weather(), ["humidex"])

    def test_missing_source_column_raises(self):
        only_temp = self._weather()[["temperature_2m"]]
        with pytest.raises(ValueError, match="apparent_temperature.*needs column"):
            add_derived_weather_features(only_temp, ["apparent_temperature"])

    def test_deterministic(self):
        w = self._weather()
        pd.testing.assert_frame_equal(
            add_derived_weather_features(w, ["hdh", "cdh"]),
            add_derived_weather_features(w, ["hdh", "cdh"]),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

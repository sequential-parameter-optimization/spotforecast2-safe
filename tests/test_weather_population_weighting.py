# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Wired-path tests for population-weighted and derived weather features.

The Open-Meteo fetch is monkeypatched (mirroring
``tests/test_weather_features_trim.py``) so these tests are fully offline and
deterministic. They check that ``get_weather_features`` combines multiple
locations by weight and derives the requested feature columns.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from spotforecast2_safe.weather import get_weather_features

START = pd.Timestamp("2024-01-01 00:00", tz="UTC")
COV_END = pd.Timestamp("2024-01-03 23:00", tz="UTC")
FORECAST_HORIZON = 24


def _data() -> pd.DataFrame:
    idx = pd.date_range(START, periods=48, freq="h", tz="UTC")
    return pd.DataFrame({"load": range(48)}, index=idx)


def _raw(temperature: float, humidity: float = 50.0, wind: float = 5.0) -> pd.DataFrame:
    idx = pd.date_range(START, COV_END, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "temperature_2m": [temperature] * len(idx),
            "relative_humidity_2m": [humidity] * len(idx),
            "wind_speed_10m": [wind] * len(idx),
        },
        index=idx,
    )


class TestPopulationWeightedPath:
    def test_two_cities_combined_by_weight(self):
        lat_a, lat_b = 52.52, 48.135

        def fake_fetch(*, latitude, **kwargs):
            return _raw(10.0 if latitude == lat_a else 20.0)

        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            side_effect=fake_fetch,
        ):
            _, aligned = get_weather_features(
                data=_data(),
                start=START,
                cov_end=COV_END,
                forecast_horizon=FORECAST_HORIZON,
                locations=[(lat_a, 13.405), (lat_b, 11.582)],
                location_weights=[3.0, 1.0],
            )

        # (3*10 + 1*20) / 4 == 12.5 at every hour.
        assert (aligned["temperature_2m"].round(6) == 12.5).all()

    def test_locations_without_weights_raises(self):
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            side_effect=lambda **kw: _raw(10.0),
        ):
            with pytest.raises(ValueError, match="location_weights is required"):
                get_weather_features(
                    data=_data(),
                    start=START,
                    cov_end=COV_END,
                    forecast_horizon=FORECAST_HORIZON,
                    locations=[(52.52, 13.405)],
                )

    def test_length_mismatch_raises(self):
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            side_effect=lambda **kw: _raw(10.0),
        ):
            with pytest.raises(ValueError, match="equal length"):
                get_weather_features(
                    data=_data(),
                    start=START,
                    cov_end=COV_END,
                    forecast_horizon=FORECAST_HORIZON,
                    locations=[(52.52, 13.405), (48.135, 11.582)],
                    location_weights=[1.0],
                )


class TestDerivedFeaturePath:
    def test_derived_columns_present(self):
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=_raw(temperature=25.0, humidity=60.0, wind=7.2),
        ):
            features, _ = get_weather_features(
                data=_data(),
                start=START,
                cov_end=COV_END,
                forecast_horizon=FORECAST_HORIZON,
                derived_features=["hdh", "cdh", "apparent_temperature", "dew_point"],
            )

        for col in ("hdh", "cdh", "apparent_temperature", "dew_point"):
            assert col in features.columns
        assert (
            not features[["hdh", "cdh", "apparent_temperature", "dew_point"]]
            .isna()
            .any()
            .any()
        )

    def test_no_derived_by_default(self):
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=_raw(temperature=25.0),
        ):
            features, _ = get_weather_features(
                data=_data(),
                start=START,
                cov_end=COV_END,
                forecast_horizon=FORECAST_HORIZON,
            )
        for col in ("hdh", "cdh", "apparent_temperature", "dew_point"):
            assert col not in features.columns

    def test_degree_hours_respects_base(self):
        # temperature 25 °C, heating base 30 → hdh = 5 everywhere.
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=_raw(temperature=25.0),
        ):
            features, _ = get_weather_features(
                data=_data(),
                start=START,
                cov_end=COV_END,
                forecast_horizon=FORECAST_HORIZON,
                derived_features=["hdh"],
                hdh_base=30.0,
            )
        assert (features["hdh"].round(6) == 5.0).all()


class TestCombinedPath:
    def test_population_weighted_plus_derived(self):
        lat_a, lat_b = 52.52, 48.135

        def fake_fetch(*, latitude, **kwargs):
            return _raw(10.0 if latitude == lat_a else 30.0, humidity=70.0, wind=3.6)

        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            side_effect=fake_fetch,
        ):
            features, aligned = get_weather_features(
                data=_data(),
                start=START,
                cov_end=COV_END,
                forecast_horizon=FORECAST_HORIZON,
                locations=[(lat_a, 13.405), (lat_b, 11.582)],
                location_weights=[1.0, 1.0],
                derived_features=["cdh"],
                cdh_base=18.0,
            )
        # weighted temp = (10 + 30) / 2 == 20; cdh with base 18 == 2.
        assert (aligned["temperature_2m"].round(6) == 20.0).all()
        assert (features["cdh"].round(6) == 2.0).all()
        # Regression: the derived column must also survive in the raw-aligned
        # frame (weather_aligned), not just the rolling-window frame
        # (weather_features) -- see TestDerivedColumnsSurviveAlignedFrame.
        assert (aligned["cdh"].round(6) == 2.0).all()


class TestDerivedColumnsSurviveAlignedFrame:
    """Regression: derived weather columns must reach ``weather_aligned``.

    ``manager.features.select_exogenous_features`` selects raw weather
    columns purely by membership in ``weather_aligned.columns``
    (``col in weather_aligned.columns``). Before this fix, the derived
    columns (hdh/cdh/apparent_temperature/dew_point) were appended only to
    the internal ``weather_aligned_filled`` copy used for the rolling-window
    transform, so they reached ``weather_features`` but NOT the returned
    ``weather_aligned`` -- silently dropping them from the final exog matrix
    even with ``include_degree_hours=True`` / ``include_apparent_temperature=True``.
    """

    def test_hdh_cdh_present_in_both_returned_frames(self):
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=_raw(temperature=25.0),
        ):
            features, aligned = get_weather_features(
                data=_data(),
                start=START,
                cov_end=COV_END,
                forecast_horizon=FORECAST_HORIZON,
                derived_features=["hdh", "cdh"],
            )
        for col in ("hdh", "cdh"):
            assert col in features.columns, f"{col} missing from weather_features"
            assert col in aligned.columns, f"{col} missing from weather_aligned"
        assert not aligned[["hdh", "cdh"]].isna().any().any()

    def test_raw_columns_and_index_unchanged_by_the_join(self):
        """The fix must not disturb the raw (non-derived) aligned columns."""
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=_raw(temperature=12.0, humidity=55.0, wind=4.0),
        ):
            _, aligned_plain = get_weather_features(
                data=_data(),
                start=START,
                cov_end=COV_END,
                forecast_horizon=FORECAST_HORIZON,
            )
            _, aligned_derived = get_weather_features(
                data=_data(),
                start=START,
                cov_end=COV_END,
                forecast_horizon=FORECAST_HORIZON,
                derived_features=["hdh", "cdh"],
            )
        raw_cols = list(aligned_plain.columns)
        assert raw_cols == [c for c in aligned_derived.columns if c in raw_cols]
        pd.testing.assert_frame_equal(aligned_derived[raw_cols], aligned_plain)
        assert aligned_derived.index.equals(aligned_plain.index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

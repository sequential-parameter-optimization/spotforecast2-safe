# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for continuous ephemeris (solar-geometry) features."""

import pandas as pd
import pytest
from astral import LocationInfo

from spotforecast2_safe.calendar import get_ephemeris_features

DORTMUND = LocationInfo(latitude=51.5136, longitude=7.4653, timezone="UTC")


def _day(date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date, tz="UTC")
    return start, start + pd.Timedelta(hours=23)


class TestEphemerisFeatures:
    def test_columns_and_shape(self):
        start, cov_end = _day("2024-06-21")
        feats = get_ephemeris_features(start, cov_end, DORTMUND)
        assert feats.columns.tolist() == [
            "solar_elevation",
            "daylight_duration_h",
            "hours_since_sunrise",
            "hours_to_sunset",
        ]
        assert feats.shape == (24, 4)
        assert not feats.isna().any().any()

    def test_summer_longer_than_winter(self):
        s_start, s_end = _day("2024-06-21")
        w_start, w_end = _day("2024-12-21")
        summer = get_ephemeris_features(s_start, s_end, DORTMUND)
        winter = get_ephemeris_features(w_start, w_end, DORTMUND)
        assert summer["daylight_duration_h"].iloc[0] > 15.0
        assert winter["daylight_duration_h"].iloc[0] < 9.0
        assert summer["solar_elevation"].max() > winter["solar_elevation"].max()

    def test_elevation_negative_at_night_positive_at_noon(self):
        start, cov_end = _day("2024-06-21")
        feats = get_ephemeris_features(start, cov_end, DORTMUND)
        # Midnight/01:00 UTC: sun below horizon. Midday: well above.
        assert feats["solar_elevation"].iloc[1] < 0.0
        assert feats["solar_elevation"].iloc[12] > 0.0

    def test_sunrise_relative_sign(self):
        start, cov_end = _day("2024-06-21")
        feats = get_ephemeris_features(start, cov_end, DORTMUND)
        # Before sunrise hours_since_sunrise is negative; late day positive.
        assert feats["hours_since_sunrise"].iloc[0] < 0.0
        assert feats["hours_since_sunrise"].iloc[-1] > 0.0
        # hours_to_sunset is positive early, negative after sunset.
        assert feats["hours_to_sunset"].iloc[0] > 0.0
        assert feats["hours_to_sunset"].iloc[-1] < 0.0

    def test_deterministic(self):
        start, cov_end = _day("2024-03-10")
        pd.testing.assert_frame_equal(
            get_ephemeris_features(start, cov_end, DORTMUND),
            get_ephemeris_features(start, cov_end, DORTMUND),
        )

    def test_empty_range(self):
        start = pd.Timestamp("2024-06-21", tz="UTC")
        feats = get_ephemeris_features(start, start - pd.Timedelta(hours=1), DORTMUND)
        assert feats.shape == (0, 4)
        assert feats.columns.tolist() == [
            "solar_elevation",
            "daylight_duration_h",
            "hours_since_sunrise",
            "hours_to_sunset",
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

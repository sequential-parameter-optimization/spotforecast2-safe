# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the German population-weighted weather-location registry."""

import pytest

from spotforecast2_safe.weather.locations import (
    GERMAN_LOAD_CENTERS,
    WeatherLocation,
    coordinates,
    default_german_locations,
    weights,
)


class TestRegistry:
    def test_default_returns_registry(self):
        assert default_german_locations() is GERMAN_LOAD_CENTERS
        assert len(GERMAN_LOAD_CENTERS) == 13

    def test_all_weights_positive(self):
        assert all(loc.weight > 0 for loc in GERMAN_LOAD_CENTERS)

    def test_names_unique(self):
        names = [loc.name for loc in GERMAN_LOAD_CENTERS]
        assert len(names) == len(set(names))

    def test_coordinates_within_germany(self):
        # Germany spans roughly 47–55 °N and 5–15 °E.
        for loc in GERMAN_LOAD_CENTERS:
            assert 47.0 <= loc.latitude <= 55.5, loc.name
            assert 5.0 <= loc.longitude <= 15.5, loc.name

    def test_geographic_spread(self):
        # The set must not collapse onto one climate zone: it spans north
        # (>53 °N) and south (<49 °N), east (>12 °E) and west (<8 °E).
        lats = [loc.latitude for loc in GERMAN_LOAD_CENTERS]
        lons = [loc.longitude for loc in GERMAN_LOAD_CENTERS]
        assert max(lats) > 53.0 and min(lats) < 49.0
        assert max(lons) > 12.0 and min(lons) < 8.0

    def test_coordinates_and_weights_align(self):
        coords = coordinates(GERMAN_LOAD_CENTERS)
        w = weights(GERMAN_LOAD_CENTERS)
        assert len(coords) == len(w) == len(GERMAN_LOAD_CENTERS)
        assert coords[0] == (
            GERMAN_LOAD_CENTERS[0].latitude,
            GERMAN_LOAD_CENTERS[0].longitude,
        )
        assert w[0] == GERMAN_LOAD_CENTERS[0].weight

    def test_deterministic_order(self):
        assert coordinates(default_german_locations()) == coordinates(
            default_german_locations()
        )

    def test_frozen_dataclass(self):
        loc = WeatherLocation("X", 50.0, 8.0, 1.0)
        with pytest.raises(Exception):
            loc.weight = 2.0  # type: ignore[misc]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

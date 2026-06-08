# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Population-weighted weather-location registry for Germany.

A single weather station (the historical default, Dortmund) cannot represent
national electricity load: demand is concentrated in the large cities, while the
climate signal varies across the country. The remedy recommended in the
forecasting literature is a **population-weighted multi-city temperature index**
(Zimmermann & Ziel 2025, ``zimm25a``): sample several load centres and combine
them with fixed weights proportional to population.

This module supplies that fixed, deterministic registry. The default set
:data:`GERMAN_LOAD_CENTERS` is chosen "smartly" along two axes:

- **Population weighting** — each centre's weight is its approximate city
  population (in thousands), so the index leans toward where the load actually
  is (cities) rather than treating every point equally.
- **Geographic spread** — the centres deliberately span the north (Hamburg,
  Bremen, Hannover), south (München, Stuttgart, Nürnberg), east (Berlin,
  Leipzig, Dresden) and west (Köln, Düsseldorf, Dortmund, Frankfurt), so the
  rural/peripheral climate diversity between the big cities is still captured
  instead of collapsing onto one urban heat profile.

The data are static published figures, so the registry is fully deterministic
and key-free, and the weights flow into
:func:`spotforecast2_safe.weather.derived.population_weighted_average`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class WeatherLocation:
    """A single weather sampling location with a population weight.

    Attributes:
        name: Human-readable city name (used for logs/inspection only).
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        weight: Non-negative relative weight (here, approximate population in
            thousands). Absolute scale is irrelevant — weights are normalised
            by the consumer.

    Examples:
        ```{python}
        from spotforecast2_safe.weather.locations import WeatherLocation

        loc = WeatherLocation("Köln", 50.9375, 6.9603, 1073.0)
        print(loc.name, loc.latitude, loc.weight)
        ```
    """

    name: str
    latitude: float
    longitude: float
    weight: float


# Thirteen German load centres spanning all regions, weighted by approximate
# city population (thousands, rounded published figures). Geographic spread is
# intentional: the set is not simply "the N largest cities" but a regionally
# balanced sample so the national index is not dominated by one climate zone.
GERMAN_LOAD_CENTERS: Tuple[WeatherLocation, ...] = (
    WeatherLocation("Berlin", 52.5200, 13.4050, 3677.0),
    WeatherLocation("Hamburg", 53.5511, 9.9937, 1906.0),
    WeatherLocation("München", 48.1351, 11.5820, 1512.0),
    WeatherLocation("Köln", 50.9375, 6.9603, 1073.0),
    WeatherLocation("Frankfurt", 50.1109, 8.6821, 773.0),
    WeatherLocation("Stuttgart", 48.7758, 9.1829, 632.0),
    WeatherLocation("Düsseldorf", 51.2277, 6.7735, 620.0),
    WeatherLocation("Leipzig", 51.3397, 12.3731, 617.0),
    WeatherLocation("Dortmund", 51.5136, 7.4653, 588.0),
    WeatherLocation("Bremen", 53.0793, 8.8017, 567.0),
    WeatherLocation("Dresden", 51.0504, 13.7373, 556.0),
    WeatherLocation("Hannover", 52.3759, 9.7320, 535.0),
    WeatherLocation("Nürnberg", 49.4521, 11.0767, 523.0),
)


def default_german_locations() -> Tuple[WeatherLocation, ...]:
    """Return the default population-weighted German load-centre registry.

    Returns:
        Tuple[WeatherLocation, ...]: :data:`GERMAN_LOAD_CENTERS`, in a fixed,
        deterministic order.

    Examples:
        ```{python}
        from spotforecast2_safe.weather.locations import default_german_locations

        locs = default_german_locations()
        print(len(locs), locs[0].name)
        ```
    """
    return GERMAN_LOAD_CENTERS


def coordinates(
    locations: Sequence[WeatherLocation],
) -> List[Tuple[float, float]]:
    """Extract ``(latitude, longitude)`` pairs in order.

    Args:
        locations: The locations to read.

    Returns:
        List[Tuple[float, float]]: One ``(lat, lon)`` pair per location, in the
        same order.

    Examples:
        ```{python}
        from spotforecast2_safe.weather.locations import (
            coordinates,
            default_german_locations,
        )

        print(coordinates(default_german_locations())[:2])
        ```
    """
    return [(loc.latitude, loc.longitude) for loc in locations]


def weights(locations: Sequence[WeatherLocation]) -> List[float]:
    """Extract the raw (un-normalised) weights in order.

    Args:
        locations: The locations to read.

    Returns:
        List[float]: One weight per location, in the same order. Consumers
        normalise these (e.g.
        :func:`spotforecast2_safe.weather.derived.population_weighted_average`).

    Examples:
        ```{python}
        from spotforecast2_safe.weather.locations import (
            default_german_locations,
            weights,
        )

        w = weights(default_german_locations())
        print(len(w), w[0])
        ```
    """
    return [loc.weight for loc in locations]

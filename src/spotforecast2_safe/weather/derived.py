# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derived weather features and population-weighted spatial aggregation.

This module turns the raw Open-Meteo fields (``temperature_2m``,
``relative_humidity_2m``, ``wind_speed_10m``, …) into the cheapest, most
reliable accuracy levers in load forecasting — the "feature-matrix" gains that
live in :math:`x_t`, not in the model :math:`f(\\cdot)`:

- **Heating / cooling degree-hours** (``hdh`` / ``cdh``) split the U-shaped
  temperature–load response into its two arms so a tree model need not
  rediscover the kink (Rubattu et al. 2023, ``ruba23a``).
- **Apparent temperature** and **dew point** fold in the humidity effect that
  dry-bulb temperature misses; ignoring it under-counts summer cooling load by
  ten to fifteen percent (Maia-Silva et al. 2020, ``maia20a``).
- **Population-weighted spatial aggregation** combines several German load
  centres into one demand-weighted weather index that tracks national load far
  better than a single station (Zimmermann & Ziel 2025, ``zimm25a``); see
  :mod:`spotforecast2_safe.weather.locations` for the default centres.

Every function here is **pure, deterministic, and fail-safe** in line with the
safety-critical contract: identical inputs produce identical outputs (no RNG, no
wall-clock, stable column order), and an input that cannot be transformed (a
missing source column, an out-of-range humidity, mismatched indices) raises an
explicit ``ValueError`` rather than being silently repaired or imputed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Default base temperatures (°C). 15 °C is the long-standing European heating
# threshold; 22 °C is a common cooling threshold. Both are overridable.
DEFAULT_HDH_BASE_C = 15.0
DEFAULT_CDH_BASE_C = 22.0

# Open-Meteo default field names this module consumes.
TEMP_COL = "temperature_2m"
HUMIDITY_COL = "relative_humidity_2m"
WIND_COL = "wind_speed_10m"

# Recognised derived-feature keys for :func:`add_derived_weather_features`.
DERIVED_FEATURE_KEYS = ("hdh", "cdh", "apparent_temperature", "dew_point")


def _require_series(values: pd.Series, name: str) -> pd.Series:
    """Validate *values* is a numeric, NaN-free ``Series`` (fail-safe)."""
    if not isinstance(values, pd.Series):
        raise ValueError(
            f"{name} must be a pandas Series; got {type(values).__name__}."
        )
    if values.isna().any():
        raise ValueError(f"{name} contains NaN; refusing to fabricate values.")
    return values.astype("float64")


def heating_degree_hours(
    temperature: pd.Series, base: float = DEFAULT_HDH_BASE_C
) -> pd.Series:
    """Heating degree-hours :math:`\\max(base - T, 0)`.

    Args:
        temperature: Hourly air temperature in °C.
        base: Heating base temperature in °C. Defaults to ``15.0``.

    Returns:
        pd.Series: ``hdh`` per timestamp, ``float64``, same index as
        *temperature*. Zero whenever it is warmer than *base*.

    Raises:
        ValueError: If *temperature* is not a NaN-free numeric Series.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.weather.derived import heating_degree_hours

        t = pd.Series([5.0, 15.0, 25.0])
        print(heating_degree_hours(t, base=15.0).tolist())
        ```
    """
    t = _require_series(temperature, "temperature")
    return (base - t).clip(lower=0.0).rename("hdh")


def cooling_degree_hours(
    temperature: pd.Series, base: float = DEFAULT_CDH_BASE_C
) -> pd.Series:
    """Cooling degree-hours :math:`\\max(T - base, 0)`.

    Args:
        temperature: Hourly air temperature in °C.
        base: Cooling base temperature in °C. Defaults to ``22.0``.

    Returns:
        pd.Series: ``cdh`` per timestamp, ``float64``, same index as
        *temperature*. Zero whenever it is cooler than *base*.

    Raises:
        ValueError: If *temperature* is not a NaN-free numeric Series.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.weather.derived import cooling_degree_hours

        t = pd.Series([5.0, 22.0, 30.0])
        print(cooling_degree_hours(t, base=22.0).tolist())
        ```
    """
    t = _require_series(temperature, "temperature")
    return (t - base).clip(lower=0.0).rename("cdh")


def dew_point(temperature: pd.Series, relative_humidity: pd.Series) -> pd.Series:
    """Dew-point temperature via the Magnus-Tetens approximation.

    Uses :math:`\\gamma = \\ln(rh/100) + aT/(b+T)` and
    :math:`T_d = b\\gamma/(a-\\gamma)` with ``a = 17.625``, ``b = 243.04`` °C,
    valid for ``-40 °C ≤ T ≤ 60 °C``. The humidity ratio is floored at a tiny
    epsilon purely to keep the logarithm finite at ``rh = 0`` (a numerical
    domain guard, not measurement imputation).

    Args:
        temperature: Hourly air temperature in °C.
        relative_humidity: Relative humidity in percent, ``0 ≤ rh ≤ 100``.

    Returns:
        pd.Series: ``dew_point`` in °C, ``float64``, same index as the inputs.

    Raises:
        ValueError: If either input is not a NaN-free numeric Series, their
            indices differ, or any humidity value lies outside ``[0, 100]``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.weather.derived import dew_point

        t = pd.Series([20.0, 20.0])
        rh = pd.Series([100.0, 50.0])
        print([round(v, 2) for v in dew_point(t, rh).tolist()])
        ```
    """
    t = _require_series(temperature, "temperature")
    rh = _require_series(relative_humidity, "relative_humidity")
    if not t.index.equals(rh.index):
        raise ValueError("temperature and relative_humidity must share an index.")
    if (rh < 0).any() or (rh > 100).any():
        raise ValueError("relative_humidity must lie within [0, 100] percent.")

    a, b = 17.625, 243.04
    ratio = np.clip(rh.to_numpy() / 100.0, 1e-6, 1.0)
    gamma = np.log(ratio) + (a * t.to_numpy()) / (b + t.to_numpy())
    td = (b * gamma) / (a - gamma)
    return pd.Series(td, index=t.index, name="dew_point").astype("float64")


def apparent_temperature(
    temperature: pd.Series,
    relative_humidity: pd.Series,
    wind_speed: pd.Series,
    *,
    wind_speed_unit: str = "ms",
) -> pd.Series:
    """Steadman apparent ("feels-like") temperature.

    :math:`AT = T + 0.33 e - 0.70 w - 4.00`, where the water-vapour pressure
    :math:`e = (rh/100)\\,6.105\\,\\exp(17.27 T/(237.7+T))` in hPa and ``w`` is
    the 10 m wind speed in m/s. This is the Australian Bureau of Meteorology
    formulation and captures the humidity load-driver that dry-bulb temperature
    misses (Maia-Silva et al. 2020, ``maia20a``).

    Args:
        temperature: Hourly air temperature in °C.
        relative_humidity: Relative humidity in percent, ``0 ≤ rh ≤ 100``.
        wind_speed: 10 m wind speed; unit set by *wind_speed_unit*.
        wind_speed_unit: ``"ms"`` (metres per second, the formula's native unit,
            default) or ``"kmh"`` (kilometres per hour — the Open-Meteo
            default, converted internally).

    Returns:
        pd.Series: ``apparent_temperature`` in °C, ``float64``, same index as
        the inputs.

    Raises:
        ValueError: If any input is not a NaN-free numeric Series, the indices
            differ, humidity is outside ``[0, 100]``, or *wind_speed_unit* is
            not ``"ms"`` or ``"kmh"``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.weather.derived import apparent_temperature

        t = pd.Series([30.0])
        rh = pd.Series([70.0])
        w = pd.Series([2.0])  # m/s
        print(round(apparent_temperature(t, rh, w).iloc[0], 2))
        ```
    """
    if wind_speed_unit not in ("ms", "kmh"):
        raise ValueError(
            f"wind_speed_unit must be 'ms' or 'kmh'; got {wind_speed_unit!r}."
        )
    t = _require_series(temperature, "temperature")
    rh = _require_series(relative_humidity, "relative_humidity")
    w = _require_series(wind_speed, "wind_speed")
    if not (t.index.equals(rh.index) and t.index.equals(w.index)):
        raise ValueError(
            "temperature, relative_humidity and wind_speed must share an index."
        )
    if (rh < 0).any() or (rh > 100).any():
        raise ValueError("relative_humidity must lie within [0, 100] percent.")

    w_ms = w / 3.6 if wind_speed_unit == "kmh" else w
    e = (rh / 100.0) * 6.105 * np.exp(17.27 * t / (237.7 + t))
    at = t + 0.33 * e - 0.70 * w_ms - 4.00
    return at.rename("apparent_temperature").astype("float64")


def population_weighted_average(
    frames: Sequence[pd.DataFrame], weights: Sequence[float]
) -> pd.DataFrame:
    """Combine per-location weather frames into one demand-weighted index.

    Forms :math:`\\sum_i \\tilde{w}_i X_i` where :math:`\\tilde{w}` are the
    *weights* normalised to sum to one. All frames must share the same index and
    the same columns (the identical Open-Meteo schema fetched per location), so
    the result is a single frame with that schema representing a national,
    population-weighted weather signal (Zimmermann & Ziel 2025, ``zimm25a``).

    Args:
        frames: One weather frame per location, all with the same
            ``DatetimeIndex`` and the same columns.
        weights: One non-negative weight per frame (e.g. city population). They
            are normalised internally; their absolute scale is irrelevant.

    Returns:
        pd.DataFrame: The weighted-average frame, same index and columns as the
        inputs (column order taken from ``frames[0]``).

    Raises:
        ValueError: If *frames* is empty, lengths mismatch, weights are
            negative or sum to zero, or the frames disagree on index or columns.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.weather.derived import population_weighted_average

        idx = pd.date_range("2023-06-01", periods=3, freq="h", tz="UTC")
        a = pd.DataFrame({"temperature_2m": [10.0, 10.0, 10.0]}, index=idx)
        b = pd.DataFrame({"temperature_2m": [20.0, 20.0, 20.0]}, index=idx)
        out = population_weighted_average([a, b], [3.0, 1.0])
        print(out["temperature_2m"].tolist())  # (3*10 + 1*20)/4 == 12.5
        ```
    """
    frames = list(frames)
    weights = list(weights)
    if not frames:
        raise ValueError("population_weighted_average requires at least one frame.")
    if len(frames) != len(weights):
        raise ValueError(
            f"frames ({len(frames)}) and weights ({len(weights)}) length mismatch."
        )
    w = np.asarray(weights, dtype="float64")
    if (w < 0).any():
        raise ValueError("weights must be non-negative.")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value.")

    base_index = frames[0].index
    base_columns = list(frames[0].columns)
    for i, f in enumerate(frames[1:], start=1):
        if not f.index.equals(base_index):
            raise ValueError(f"frame {i} has a different index from frame 0.")
        if list(f.columns) != base_columns:
            raise ValueError(f"frame {i} has different columns from frame 0.")

    norm = w / total
    acc = frames[0][base_columns].astype("float64") * norm[0]
    for i in range(1, len(frames)):
        acc = acc + frames[i][base_columns].astype("float64") * norm[i]
    return acc


def add_derived_weather_features(
    weather: pd.DataFrame,
    features: Sequence[str],
    *,
    hdh_base: float = DEFAULT_HDH_BASE_C,
    cdh_base: float = DEFAULT_CDH_BASE_C,
    temp_col: str = TEMP_COL,
    humidity_col: str = HUMIDITY_COL,
    wind_col: str = WIND_COL,
    wind_speed_unit: str = "kmh",
) -> pd.DataFrame:
    """Append the requested derived columns to a raw weather frame (fail-safe).

    A thin, deterministic orchestrator over :func:`heating_degree_hours`,
    :func:`cooling_degree_hours`, :func:`dew_point`, and
    :func:`apparent_temperature`. The original columns are preserved; the
    requested derived columns are appended in the fixed order of
    :data:`DERIVED_FEATURE_KEYS` (stable column ordering). The default
    *wind_speed_unit* is ``"kmh"`` to match the Open-Meteo payload consumed by
    :func:`spotforecast2_safe.weather.get_weather_features`.

    Args:
        weather: Raw weather frame containing the source columns the requested
            features need.
        features: Any subset of :data:`DERIVED_FEATURE_KEYS`. An empty sequence
            returns *weather* unchanged (a copy).
        hdh_base: Heating base temperature in °C.
        cdh_base: Cooling base temperature in °C.
        temp_col: Name of the temperature column.
        humidity_col: Name of the relative-humidity column.
        wind_col: Name of the wind-speed column.
        wind_speed_unit: Unit of *wind_col*, ``"ms"`` or ``"kmh"`` (default).

    Returns:
        pd.DataFrame: A copy of *weather* with the requested derived columns
        appended.

    Raises:
        ValueError: If a requested feature is unknown, or a source column it
            needs is absent from *weather*.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.weather.derived import add_derived_weather_features

        idx = pd.date_range("2023-07-01", periods=2, freq="h", tz="UTC")
        weather = pd.DataFrame(
            {
                "temperature_2m": [28.0, 30.0],
                "relative_humidity_2m": [60.0, 55.0],
                "wind_speed_10m": [7.2, 10.8],  # km/h
            },
            index=idx,
        )
        out = add_derived_weather_features(
            weather, ["hdh", "cdh", "apparent_temperature", "dew_point"]
        )
        print(sorted(set(out.columns) - set(weather.columns)))
        ```
    """
    unknown = [f for f in features if f not in DERIVED_FEATURE_KEYS]
    if unknown:
        raise ValueError(
            f"unknown derived feature(s) {unknown}; "
            f"valid keys are {list(DERIVED_FEATURE_KEYS)}."
        )

    def _col(name: str, needed_for: str) -> pd.Series:
        if name not in weather.columns:
            raise ValueError(
                f"derived feature '{needed_for}' needs column '{name}', "
                f"absent from weather frame (has {list(weather.columns)})."
            )
        return weather[name]

    out = weather.copy()
    # Iterate in the canonical order so column order is deterministic regardless
    # of the order the caller listed the features in.
    for key in DERIVED_FEATURE_KEYS:
        if key not in features:
            continue
        if key == "hdh":
            out["hdh"] = heating_degree_hours(_col(temp_col, "hdh"), base=hdh_base)
        elif key == "cdh":
            out["cdh"] = cooling_degree_hours(_col(temp_col, "cdh"), base=cdh_base)
        elif key == "dew_point":
            out["dew_point"] = dew_point(
                _col(temp_col, "dew_point"), _col(humidity_col, "dew_point")
            )
        elif key == "apparent_temperature":
            out["apparent_temperature"] = apparent_temperature(
                _col(temp_col, "apparent_temperature"),
                _col(humidity_col, "apparent_temperature"),
                _col(wind_col, "apparent_temperature"),
                wind_speed_unit=wind_speed_unit,
            )
    return out

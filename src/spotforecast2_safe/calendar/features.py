# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Calendar and day/night feature engineering for forecasting pipelines.

Public helpers:

- `get_calendar_features()` — extract month, week, day-of-week, and hour
  from a time index.
- `get_day_night_features()` — derive sunrise hour, sunset hour, daylight
  hours, and an ``is_daylight`` indicator using the
  `astral <https://astral.readthedocs.io>`_ library.
- `get_ephemeris_features()` — per-hour solar elevation plus daylight
  duration and signed sunrise/sunset-relative time. Purely deterministic
  (date + fixed coordinates), leakage-free, no extra dependency; sharpens
  the lighting-load timing and midday PV offset the hour-of-day RBFs only
  encode implicitly.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import elevation, sun
from feature_engine.datetime import DatetimeFeatures

from spotforecast2_safe.utils.convert_to_utc import to_utc_timestamp


def get_calendar_features(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    freq: str = "h",
    timezone: str = "UTC",
    features_to_extract: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create calendar-based features for a contiguous time range.

    Uses `DatetimeFeatures` to extract
    temporal components from a regularly spaced ``DatetimeIndex``.  The
    resulting DataFrame has the same index as the generated time grid and
    one integer column per requested feature.

    Args:
        start: Start of the time range.  String values are parsed with
            ``utc=True``.
        cov_end: Inclusive end of the time range.  String values are
            parsed with ``utc=True``.
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        timezone: Timezone label applied to the generated index.
            Defaults to ``"UTC"``.
        features_to_extract: Calendar components to extract.  Defaults
            to ``["month", "week", "day_of_week", "hour"]``.

    Returns:
        pd.DataFrame: DataFrame with integer columns for each extracted
        calendar feature.  The index is a tz-aware
        `DatetimeIndex` with the requested ``freq``.

    Raises:
        ValueError: If ``start`` is later than ``cov_end``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.calendar import get_calendar_features

        start = pd.Timestamp("2024-01-01", tz="UTC")
        cov_end = pd.Timestamp("2024-01-02 23:00", tz="UTC")  # 48 hours

        features = get_calendar_features(
            start=start,
            cov_end=cov_end,
            freq="h",
            timezone="UTC",
        )
        print("shape:", features.shape)
        print("columns:", features.columns.tolist())
        print(features.head(3))
        assert features.shape == (48, 4)
        assert features.columns.tolist() == ["month", "week", "day_of_week", "hour"]
        ```
    """
    if features_to_extract is None:
        features_to_extract = ["month", "week", "day_of_week", "hour"]

    start = to_utc_timestamp(start)
    cov_end = to_utc_timestamp(cov_end)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=timezone)
    extended_data = pd.DataFrame({"dummy": 0}, index=extended_index)

    transformer = DatetimeFeatures(
        variables="index",
        features_to_extract=features_to_extract,
        drop_original=True,
    )
    return transformer.fit_transform(extended_data)[features_to_extract]


def get_day_night_features(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    location: LocationInfo,
    freq: str = "h",
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Create day/night features using astronomical sunrise and sunset times.

    Sunrise and sunset times are computed once per unique calendar date
    (using `astral.sun.sun()`) and then broadcast to all timestamps
    in the requested hourly grid, which avoids redundant computation for
    large date ranges.

    The returned DataFrame contains four columns:

    - ``sunrise_hour`` — rounded sunrise hour (0–23).
    - ``sunset_hour`` — rounded sunset hour (0–23).
    - ``daylight_hours`` — ``sunset_hour - sunrise_hour``.
    - ``is_daylight`` — ``1`` if the timestamp is between sunrise and
      sunset, else ``0``.

    Args:
        start: Start of the time range.  String values are parsed with
            ``utc=True``.
        cov_end: Inclusive end of the time range.  String values are
            parsed with ``utc=True``.
        location: `LocationInfo` instance describing the
            geographic location (latitude, longitude, timezone).
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        timezone: Timezone label applied to the generated index.
            Defaults to ``"UTC"``.

    Returns:
        pd.DataFrame: DataFrame with columns ``sunrise_hour``,
        ``sunset_hour``, ``daylight_hours``, ``is_daylight``.  The index
        is a tz-aware `DatetimeIndex` with the requested
        ``freq``.

    Examples:
        ```{python}
        import pandas as pd
        from astral import LocationInfo
        from spotforecast2_safe.calendar import get_day_night_features

        # Use a two-week window around the autumn equinox so that
        # is_daylight.mean() is close to 0.5 (daylight ~12 h / 24 h).
        start = pd.Timestamp("2024-09-15", tz="UTC")
        cov_end = pd.Timestamp("2024-09-28 23:00", tz="UTC")

        location = LocationInfo(
            latitude=51.5136,
            longitude=7.4653,
            timezone="UTC",
        )
        features = get_day_night_features(
            start=start,
            cov_end=cov_end,
            location=location,
            freq="h",
            timezone="UTC",
        )
        print("shape:", features.shape)
        print("columns:", features.columns.tolist())
        print("is_daylight mean:", round(features["is_daylight"].mean(), 3))
        print(features.head(3))
        assert features.shape == (336, 4)
        assert features.columns.tolist() == [
            "sunrise_hour",
            "sunset_hour",
            "daylight_hours",
            "is_daylight",
        ]
        assert 0.4 < features["is_daylight"].mean() < 0.65
        ```
    """
    start = to_utc_timestamp(start)
    cov_end = to_utc_timestamp(cov_end)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=timezone)
    normalized_dates = extended_index.normalize()
    unique_dates = normalized_dates.unique()

    # Cache sunrise / sunset per unique date so astral.sun is called once
    # per day instead of once per timestamp in the grid.
    sun_by_date = {
        d: sun(location.observer, date=d, tzinfo=location.timezone)
        for d in unique_dates
    }
    sunrise_series = pd.Series(
        [sun_by_date[d]["sunrise"] for d in normalized_dates],
        index=extended_index,
    )
    sunset_series = pd.Series(
        [sun_by_date[d]["sunset"] for d in normalized_dates],
        index=extended_index,
    )

    sunrise_hour = sunrise_series.dt.round("h").dt.hour
    sunset_hour = sunset_series.dt.round("h").dt.hour

    features = pd.DataFrame(
        {
            "sunrise_hour": sunrise_hour,
            "sunset_hour": sunset_hour,
        }
    )
    features["daylight_hours"] = features["sunset_hour"] - features["sunrise_hour"]
    features["is_daylight"] = np.where(
        (extended_index.hour >= features["sunrise_hour"])
        & (extended_index.hour < features["sunset_hour"]),
        1,
        0,
    )
    return features


def get_ephemeris_features(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    location: LocationInfo,
    freq: str = "h",
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Create continuous solar-geometry features from the ephemeris.

    Unlike `get_day_night_features` (which rounds sunrise/sunset to whole
    hours and emits a binary daylight flag), this builder exposes the
    *continuous* solar geometry the hour-of-day RBFs only encode implicitly:
    the per-hour solar elevation, the exact daylight duration, and the signed
    time relative to sunrise and sunset. These linearise lighting-load timing
    and the midday PV offset, are purely deterministic from the date and the
    fixed coordinates, add no dependency, and leak nothing for any forecast
    hour (Xie & Hong 2018, ``xieh18a``; López 2020, ``lope20a``).

    The returned DataFrame contains four ``float64`` columns:

    - ``solar_elevation`` — solar elevation angle in degrees (negative at
      night, peaking at solar noon).
    - ``daylight_duration_h`` — exact sunset−sunrise span for the date, hours.
    - ``hours_since_sunrise`` — signed hours since that date's sunrise
      (negative before sunrise).
    - ``hours_to_sunset`` — signed hours until that date's sunset
      (negative after sunset).

    Args:
        start: Start of the time range. String values are parsed with
            ``utc=True``.
        cov_end: Inclusive end of the time range. String values are parsed
            with ``utc=True``.
        location: `LocationInfo` describing the geographic location.
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        timezone: Timezone label applied to the generated index. Defaults to
            ``"UTC"``.

    Returns:
        pd.DataFrame: Columns ``solar_elevation``, ``daylight_duration_h``,
        ``hours_since_sunrise``, ``hours_to_sunset``; tz-aware
        `DatetimeIndex` with the requested ``freq``.

    Examples:
        ```{python}
        import pandas as pd
        from astral import LocationInfo
        from spotforecast2_safe.calendar import get_ephemeris_features

        start = pd.Timestamp("2024-06-21", tz="UTC")
        cov_end = pd.Timestamp("2024-06-21 23:00", tz="UTC")
        location = LocationInfo(latitude=51.5136, longitude=7.4653, timezone="UTC")

        feats = get_ephemeris_features(start, cov_end, location)
        print("columns:", feats.columns.tolist())
        print("shape:", feats.shape)
        # Summer solstice: long day and a high midday sun in Dortmund.
        print("max elevation:", round(feats["solar_elevation"].max(), 1))
        assert feats.shape == (24, 4)
        assert feats["solar_elevation"].max() > 50.0
        assert feats["daylight_duration_h"].iloc[0] > 14.0
        ```
    """
    start = to_utc_timestamp(start)
    cov_end = to_utc_timestamp(cov_end)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=timezone)
    if len(extended_index) == 0:
        return pd.DataFrame(
            columns=[
                "solar_elevation",
                "daylight_duration_h",
                "hours_since_sunrise",
                "hours_to_sunset",
            ],
            index=extended_index,
            dtype="float64",
        )

    normalized_dates = extended_index.normalize()
    unique_dates = normalized_dates.unique()

    # Cache sunrise/sunset per unique date so astral.sun runs once per day.
    sun_by_date = {
        d: sun(location.observer, date=d, tzinfo=location.timezone)
        for d in unique_dates
    }
    sunrise = pd.DatetimeIndex(
        [sun_by_date[d]["sunrise"] for d in normalized_dates]
    ).tz_convert("UTC")
    sunset = pd.DatetimeIndex(
        [sun_by_date[d]["sunset"] for d in normalized_dates]
    ).tz_convert("UTC")

    idx_utc = extended_index.tz_convert("UTC")
    hour = pd.Timedelta(hours=1)
    hours_since_sunrise = (idx_utc - sunrise) / hour
    hours_to_sunset = (sunset - idx_utc) / hour
    daylight_duration_h = (sunset - sunrise) / hour

    # Per-hour solar elevation; astral computes it analytically from the
    # observer and the exact timestamp (deterministic, no network).
    solar_elevation = [
        elevation(location.observer, ts.to_pydatetime()) for ts in extended_index
    ]

    return pd.DataFrame(
        {
            "solar_elevation": np.asarray(solar_elevation, dtype="float64"),
            "daylight_duration_h": np.asarray(daylight_duration_h, dtype="float64"),
            "hours_since_sunrise": np.asarray(hours_since_sunrise, dtype="float64"),
            "hours_to_sunset": np.asarray(hours_to_sunset, dtype="float64"),
        },
        index=extended_index,
    )

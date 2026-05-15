# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Calendar and day/night feature engineering for forecasting pipelines.

Public helpers:

- :func:`get_calendar_features` — extract month, week, day-of-week, and hour
  from a time index.
- :func:`get_day_night_features` — derive sunrise hour, sunset hour, daylight
  hours, and an ``is_daylight`` indicator using the
  `astral <https://astral.readthedocs.io>`_ library.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from feature_engine.datetime import DatetimeFeatures

from spotforecast2_safe.calendar._common import to_utc_timestamp


def get_calendar_features(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    freq: str = "h",
    timezone: str = "UTC",
    features_to_extract: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create calendar-based features for a contiguous time range.

    Uses :class:`~feature_engine.datetime.DatetimeFeatures` to extract
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
        :class:`~pandas.DatetimeIndex` with the requested ``freq``.

    Raises:
        ValueError: If ``start`` is later than ``cov_end``.

    Examples:

        ```{python}
        import pandas as pd
        from spotforecast2_safe.calendar import get_calendar_features

        start = pd.Timestamp("2024-01-01", tz="UTC")
        cov_end = pd.Timestamp("2024-01-07 23:00", tz="UTC")

        features = get_calendar_features(
            start=start,
            cov_end=cov_end,
            freq="h",
            timezone="UTC",
        )
        print("shape:", features.shape)
        print("columns:", features.columns.tolist())
        print(features.head(3))
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
    (using :func:`astral.sun.sun`) and then broadcast to all timestamps
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
        location: :class:`~astral.LocationInfo` instance describing the
            geographic location (latitude, longitude, timezone).
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        timezone: Timezone label applied to the generated index.
            Defaults to ``"UTC"``.

    Returns:
        pd.DataFrame: DataFrame with columns ``sunrise_hour``,
        ``sunset_hour``, ``daylight_hours``, ``is_daylight``.  The index
        is a tz-aware :class:`~pandas.DatetimeIndex` with the requested
        ``freq``.

    Examples:

        ```{python}
        import pandas as pd
        from astral import LocationInfo
        from spotforecast2_safe.calendar import get_day_night_features

        start = pd.Timestamp("2024-06-01", tz="UTC")
        cov_end = pd.Timestamp("2024-06-07 23:00", tz="UTC")

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
        print(features.head(3))
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

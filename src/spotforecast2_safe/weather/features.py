# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Weather feature engineering for time series forecasting pipelines.

This module provides the `get_weather_features()` function, which fetches
historical and forecast weather data via `fetch_weather_data()`
and transforms it into rolling-window features suitable for use as exogenous
variables in recursive forecasting models.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from feature_engine.timeseries.forecasting import WindowFeatures

from spotforecast2_safe.preprocessing.curate_data import curate_weather
from spotforecast2_safe.utils.convert_to_utc import to_utc_timestamp


def get_weather_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    latitude: float = 51.5136,
    longitude: float = 7.4653,
    timezone: str = "UTC",
    freq: str = "h",
    window_periods: Optional[List[str]] = None,
    window_functions: Optional[List[str]] = None,
    fallback_on_failure: bool = True,
    cache_home: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch weather data and compute rolling-window features.

    Downloads weather observations/forecasts for the requested period,
    aligns them to a regular ``freq`` grid, and applies
    `WindowFeatures` to
    produce rolling-mean, -max, and -min features over configurable
    windows.

    Args:
        data: Reference time series DataFrame used only for validation
            (shape / temporal coverage checks via
            `curate_weather()`).
        start: Start of the feature window.  String values are parsed
            with ``utc=True``.
        cov_end: Inclusive end of the feature window (must cover the
            full forecast horizon beyond ``end``).  String values are
            parsed with ``utc=True``.
        forecast_horizon: Number of forecast steps; passed to
            `curate_weather()`
            for validation.
        latitude: Latitude of the target location in decimal degrees.
            Defaults to ``51.5136`` (Dortmund, Germany).
        longitude: Longitude of the target location in decimal degrees.
            Defaults to ``7.4653`` (Dortmund, Germany).
        timezone: Timezone label applied to the generated index.
            Defaults to ``"UTC"``.
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        window_periods: Rolling window sizes passed to
            `WindowFeatures`.
            Defaults to ``["1D", "7D"]``.
        window_functions: Aggregation functions applied over each window.
            Defaults to ``["mean", "max", "min"]``.
        fallback_on_failure: If ``True``, use locally cached fallback
            data when the weather API is unavailable.  Defaults to
            ``True``.
        cache_home: Optional path to cache directory.  When provided,
            fetched weather data is cached in
            ``<cache_home>/weather_cache.parquet``.  When None (default),
            no caching is performed.
        verbose: If ``True``, print progress messages to stdout.
            Defaults to ``False``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A two-element tuple:

        - **weather_features** – DataFrame with rolling-window weather
            features aligned to the ``[start, cov_end]`` index.
        - **weather_aligned** – Raw weather DataFrame reindexed to the
            same ``[start, cov_end]`` hourly grid (forward-filled).

    Raises:
        ValueError: If no numeric weather columns are found, or if
            missing values cannot be filled after fetching.

    Examples:
        ```{python}
        import tempfile

        import pandas as pd

        from spotforecast2_safe.weather import get_weather_features

        # Build a minimal synthetic reference DataFrame whose row count is
        # consistent with the requested weather window so curate_weather
        # validation passes without warnings.
        forecast_horizon = 2
        start = pd.Timestamp("2020-06-01", tz="UTC")
        cov_end = pd.Timestamp("2020-06-03", tz="UTC")
        data_end = cov_end - pd.Timedelta(hours=forecast_horizon)
        data_idx = pd.date_range(start=start, end=data_end, freq="h", tz="UTC")
        data = pd.DataFrame({"load": range(len(data_idx))}, index=data_idx)

        cache_home = tempfile.mkdtemp()
        weather_features, weather_aligned = get_weather_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            cache_home=cache_home,
            verbose=False,
        )
        print("weather_features shape:", weather_features.shape)
        print("weather_aligned shape:", weather_aligned.shape)
        print("weather_aligned columns (first 3):", list(weather_aligned.columns)[:3])
        assert weather_features.shape[0] > 0
        assert weather_aligned.shape[0] > 0
        assert "temperature_2m" in weather_aligned.columns
        # Rolling-window transformer adds more columns than raw aligned data
        assert weather_features.shape[1] > weather_aligned.shape[1]
        ```
    """
    if window_periods is None:
        window_periods = ["1D", "7D"]
    if window_functions is None:
        window_functions = ["mean", "max", "min"]

    # Deferred import to break the weather → data.fetch_data → weather circular
    # dependency that arises because fetch_data imports WeatherService from
    # weather.__init__, which in turn re-exports get_weather_features.
    from spotforecast2_safe.data.fetch_data import fetch_weather_data  # noqa: PLC0415

    start = to_utc_timestamp(start)
    cov_end = to_utc_timestamp(cov_end)

    if verbose:
        print("Fetching weather data...")

    weather_df = fetch_weather_data(
        cov_start=start,
        cov_end=cov_end,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        freq=freq,
        fallback_on_failure=fallback_on_failure,
        cache_home=cache_home,
    )

    # Open-Meteo returns whole-day forecast blocks aligned to its own clock,
    # so the raw payload can extend a few hours past ``cov_end``. Trim to the
    # exact ``[start, cov_end]`` window before ``curate_weather`` so the
    # row-count assertion compares against the window actually used by the
    # downstream reindex; otherwise it prints a spurious "wrong shape"
    # warning even though alignment will repair the count.
    weather_df = weather_df.loc[start:cov_end]

    curate_weather(weather_df, data, forecast_horizon=forecast_horizon)

    if verbose:
        print("Processing weather features...")

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=timezone)
    weather_aligned = weather_df.reindex(extended_index, method="ffill")

    weather_columns = weather_aligned.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(weather_columns) == 0:
        raise ValueError("No numeric weather columns found")

    weather_aligned_filled = weather_aligned[weather_columns].copy()
    if weather_aligned_filled.isnull().any().any():
        weather_aligned_filled = weather_aligned_filled.bfill()
        if weather_aligned_filled.isnull().any().any():
            raise ValueError("Missing values in weather data could not be filled")

    wf_transformer = WindowFeatures(
        variables=weather_columns,
        window=window_periods,
        functions=window_functions,
        freq=freq,
    )

    weather_features = wf_transformer.fit_transform(weather_aligned_filled)

    if weather_features.isnull().any().any():
        weather_features = weather_features.bfill()
        if weather_features.isnull().any().any():
            raise ValueError("Missing values in weather features could not be filled")

    if verbose:
        print(f"Weather features shape: {weather_features.shape}")

    return weather_features, weather_aligned

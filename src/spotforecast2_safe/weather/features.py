# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Weather feature engineering for time series forecasting pipelines.

This module provides the `get_weather_features()` function, which fetches
historical and forecast weather data via `fetch_weather_data()`
and transforms it into rolling-window features suitable for use as exogenous
variables in recursive forecasting models.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from feature_engine.timeseries.forecasting import WindowFeatures

from spotforecast2_safe.preprocessing.curate_data import curate_weather
from spotforecast2_safe.utils.convert_to_utc import to_utc_timestamp
from spotforecast2_safe.weather.client import WeatherFetchError
from spotforecast2_safe.weather.derived import (
    DEFAULT_CDH_BASE_C,
    DEFAULT_HDH_BASE_C,
    add_derived_weather_features,
    population_weighted_average,
)

# Longest run of consecutive missing ``freq`` steps that ``get_weather_features``
# will forward-fill during alignment. A longer gap (e.g. the recent window the
# Open-Meteo archive endpoint has not yet published, with the forecast leg also
# missing it) is refused rather than silently carried forward as a flat line.
MAX_FFILL_GAP_PERIODS = 24


def _longest_consecutive_run(index: pd.DatetimeIndex, freq: str) -> int:
    """Length of the longest run of consecutive ``freq``-spaced timestamps."""
    if len(index) == 0:
        return 0
    index = index.sort_values()
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    best = run = 1
    prev = index[0]
    for ts in index[1:]:
        run = run + 1 if (ts - prev) == step else 1
        best = max(best, run)
        prev = ts
    return best


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
    locations: Optional[Sequence[Tuple[float, float]]] = None,
    location_weights: Optional[Sequence[float]] = None,
    derived_features: Optional[Sequence[str]] = None,
    hdh_base: float = DEFAULT_HDH_BASE_C,
    cdh_base: float = DEFAULT_CDH_BASE_C,
    wind_speed_unit: str = "kmh",
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
        locations: Optional sequence of ``(latitude, longitude)`` pairs for a
            **population-weighted multi-city** weather index. When ``None``
            (default) the single ``latitude``/``longitude`` point is used,
            preserving prior behaviour exactly. When given, each location is
            fetched and the raw frames are combined via
            `population_weighted_average`
            using *location_weights*. See
            `spotforecast2_safe.weather.locations`.
        location_weights: Non-negative weight per entry in *locations* (e.g.
            city population). Required when *locations* is given; normalised
            internally.
        derived_features: Optional subset of ``{"hdh", "cdh",
            "apparent_temperature", "dew_point"}``. When given, those columns
            are derived from the (weighted) weather and rolled up alongside the
            raw fields. ``None`` (default) adds nothing. See
            `add_derived_weather_features`.
        hdh_base: Heating base temperature (°C) for ``hdh``. Defaults to
            ``15.0``.
        cdh_base: Cooling base temperature (°C) for ``cdh``. Defaults to
            ``22.0``.
        wind_speed_unit: Unit of the fetched ``wind_speed_10m`` column for
            apparent-temperature, ``"ms"`` or ``"kmh"``. Defaults to ``"kmh"``
            (the Open-Meteo default).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A two-element tuple:

        - **weather_features** – DataFrame with rolling-window weather
            features aligned to the ``[start, cov_end]`` index.
        - **weather_aligned** – Raw weather DataFrame reindexed to the
            same ``[start, cov_end]`` hourly grid (forward-filled). Also
            carries any columns requested via *derived_features* (e.g.
            ``hdh``, ``cdh``, ``apparent_temperature``, ``dew_point``), so
            they are selectable downstream via raw-weather membership in
            ``weather_aligned.columns``.

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

    def _fetch_one(lat: float, lon: float) -> pd.DataFrame:
        """Fetch one location and trim to the exact ``[start, cov_end]`` window.

        Open-Meteo returns whole-day forecast blocks aligned to its own clock,
        so the raw payload can extend a few hours past ``cov_end``. Trimming
        here keeps the row-count assertion in ``curate_weather`` honest and
        guarantees every location shares one index before any weighted combine.
        """
        frame = fetch_weather_data(
            cov_start=start,
            cov_end=cov_end,
            latitude=lat,
            longitude=lon,
            timezone=timezone,
            freq=freq,
            fallback_on_failure=fallback_on_failure,
            cache_home=cache_home,
        )
        return frame.loc[start:cov_end]

    if locations is not None:
        if location_weights is None:
            raise ValueError("location_weights is required when locations is supplied.")
        if len(locations) != len(location_weights):
            raise ValueError(
                f"locations ({len(locations)}) and location_weights "
                f"({len(location_weights)}) must have equal length."
            )
        frames = [_fetch_one(lat, lon) for (lat, lon) in locations]
        # population_weighted_average is fail-safe: it raises a plain ValueError
        # if the per-city frames disagree on index or columns rather than
        # silently aligning. That disagreement is, in practice, a transient
        # per-city fetch failure (e.g. a 429 sending one city to differently
        # ranged fallback data). Re-raise it as WeatherFetchError so the
        # `on_weather_failure` policy can govern it — a bare ValueError would
        # otherwise escape "skip" and crash the whole run (per-zone weather).
        try:
            weather_df = population_weighted_average(frames, list(location_weights))
        except WeatherFetchError:
            raise
        except ValueError as exc:
            raise WeatherFetchError(
                "Multi-city weather frames could not be combined "
                f"({len(frames)} locations); this usually means a transient "
                f"per-city fetch failure or fallback range mismatch: {exc}"
            ) from exc
    else:
        weather_df = _fetch_one(latitude, longitude)

    curate_weather(weather_df, data, forecast_horizon=forecast_horizon)

    if verbose:
        print("Processing weather features...")

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=timezone)

    # Refuse to silently carry the last observation forward across a long gap.
    # If the fetch returned no data for a stretch of the requested window (e.g.
    # the archive endpoint lags and the forecast leg failed), masking it as a
    # flat line of repeated values would inject synthetic "measurements". Short
    # benign gaps (<= MAX_FFILL_GAP_PERIODS, e.g. whole-day-block boundaries)
    # are still forward-filled below.
    missing = extended_index.difference(weather_df.index)
    if len(missing) > 0:
        longest_gap = _longest_consecutive_run(missing, freq)
        if longest_gap > MAX_FFILL_GAP_PERIODS:
            raise WeatherFetchError(
                f"Weather coverage gap of {longest_gap} consecutive '{freq}' "
                f"steps within [{start}, {cov_end}] exceeds the "
                f"{MAX_FFILL_GAP_PERIODS}-step fill tolerance; refusing to carry "
                "the last observation forward. The Open-Meteo fetch likely did "
                "not return recent data for this window."
            )

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

    # Derive degree-hours / apparent-temperature / dew-point on the NaN-free
    # frame (fail-safe: raises if a source column is missing) so they are rolled
    # up alongside the raw fields by WindowFeatures below.
    if derived_features:
        weather_aligned_filled = add_derived_weather_features(
            weather_aligned_filled,
            list(derived_features),
            hdh_base=hdh_base,
            cdh_base=cdh_base,
            wind_speed_unit=wind_speed_unit,
        )
        weather_columns = weather_aligned_filled.columns.tolist()
        # Join the newly-derived columns onto the RETURNED weather_aligned (not
        # just the internal *_filled copy used for the window transform) so
        # they are also selectable as raw-weather columns downstream: the
        # raw-column membership test in
        # manager.features.select_exogenous_features matches
        # ``col in weather_aligned.columns``. Without this, hdh/cdh/apparent
        # temperature/dew point were silently dropped from the final exog
        # matrix even though the caller opted in via
        # ``include_degree_hours`` / ``include_apparent_temperature``.
        derived_cols = [
            c
            for c in weather_aligned_filled.columns
            if c not in weather_aligned.columns
        ]
        weather_aligned = weather_aligned.join(weather_aligned_filled[derived_cols])

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

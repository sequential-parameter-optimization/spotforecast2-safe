# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-TSO-zone weather, concatenated as suffixed columns for one target.

This module supports the ``zone_weather_columns`` opt-in knob
(`spotforecast2_safe.configurator.config_multi.ConfigMulti`): for a *single*
target, fetch population-weighted weather for each of the four German TSO
zones (`spotforecast2_safe.weather.locations.GERMAN_TSO_ZONE_CITIES`) and
concatenate every zone's raw (plus derived/window) columns with a
``__<zone_short>`` suffix, instead of overwriting the shared weather block per
target as ``per_zone_weather`` does. The result is a wider single-target
feature frame — e.g. ``temperature_2m__50hertz``, ``temperature_2m__amprion``,
... — so downstream models can learn zone-specific climate effects for a
single (typically national) target series.

`build_zone_weather_columns` is the single entry point. It is fail-safe: any
per-zone `WeatherFetchError` re-raises for the *whole* block (mirroring the
``per_zone_weather`` degrade-to-empty behaviour under
``on_weather_failure="skip"`` at the ``multitask.base`` call site), never
mixing a partial set of zones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import pandas as pd

from spotforecast2_safe.weather.client import WeatherFetchError
from spotforecast2_safe.weather.features import get_weather_features
from spotforecast2_safe.weather.locations import GERMAN_TSO_ZONE_CITIES
from spotforecast2_safe.weather.locations import coordinates as _coordinates
from spotforecast2_safe.weather.locations import locations_for_zone
from spotforecast2_safe.weather.locations import weights as _weights


def _zone_short(zone: str) -> str:
    """Strip the ``load_`` prefix from a TSO zone key (``"load_50hertz"`` -> ``"50hertz"``)."""
    return zone.removeprefix("load_")


def build_zone_weather_columns(
    *,
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    latitude: float,
    longitude: float,
    timezone: str = "UTC",
    freq: str = "h",
    cache_home: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    derived_features: Optional[Sequence[str]] = None,
    hdh_base: float = 15.0,
    cdh_base: float = 22.0,
    zone_locations_override: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and concatenate population-weighted weather for all four TSO zones.

    Loops the four German TSO zones in
    `spotforecast2_safe.weather.locations.GERMAN_TSO_ZONE_CITIES` order.  For
    each zone, resolves its regional cities via
    `spotforecast2_safe.weather.locations.locations_for_zone` and fetches a
    population-weighted weather frame via
    `spotforecast2_safe.weather.features.get_weather_features`.  Every zone's
    columns are then suffixed ``__<zone_short>`` (the zone key with its
    ``load_`` prefix stripped, e.g. ``50hertz``) and concatenated along the
    columns axis, in zone-registry order, giving a deterministic column
    layout.

    Args:
        data: Reference DataFrame forwarded to `get_weather_features` for
            shape/coverage validation.
        start: Start of the feature window (native window, not truncated to
            *data*'s extent — it must span the forecast horizon).
        cov_end: Inclusive end of the feature window (covers the forecast
            horizon beyond the last observed target row).
        forecast_horizon: Number of forecast steps, forwarded to
            `get_weather_features`.
        latitude: Unused by the per-zone fetch itself (each zone supplies its
            own coordinates); kept for API symmetry with the single-point
            path and forwarded to `get_weather_features` as the fallback
            point (never exercised because ``locations`` is always given).
        longitude: See *latitude*.
        timezone: IANA timezone label for the generated index.
        freq: Pandas-compatible frequency string. Defaults to ``"h"``.
        cache_home: Optional weather-cache directory, forwarded to
            `get_weather_features`.
        verbose: If ``True``, forward progress messages to stdout.
        derived_features: Optional derived-feature subset (``hdh``, ``cdh``,
            ``apparent_temperature``, ``dew_point``), forwarded verbatim to
            every zone's fetch.
        hdh_base: Heating base temperature (°C) for ``hdh``.
        cdh_base: Cooling base temperature (°C) for ``cdh``.
        zone_locations_override: Optional override mapping zone key -> a
            sequence of `spotforecast2_safe.weather.locations.WeatherLocation`
            objects, forwarded to `locations_for_zone` for each zone.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(weather_features_concat,
        weather_aligned_concat)`` — the rolling-window feature frame and the
        raw aligned frame, both column-suffixed per zone and concatenated in
        `GERMAN_TSO_ZONE_CITIES` order. Both keep the native ``[start,
        cov_end]`` index (spans the forecast horizon).

    Raises:
        WeatherFetchError: If any single zone's fetch fails; the whole block
            is refused rather than silently mixing a partial zone set.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.weather import zone_columns as zc

        def fake_weather(*, data, start, cov_end, locations=None, **kwargs):
            idx = pd.date_range(start, cov_end, freq="h")
            offset = sum(lat for lat, _ in locations) / len(locations)
            frame = pd.DataFrame({"temperature_2m": offset + idx.hour * 0.0}, index=idx)
            return frame, frame

        zc.get_weather_features = fake_weather  # monkeypatch for this example

        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        ref = pd.DataFrame({"load": range(len(idx))}, index=idx)
        wf, wa = zc.build_zone_weather_columns(
            data=ref,
            start=idx[0],
            cov_end=idx[-1],
            forecast_horizon=6,
            latitude=51.5,
            longitude=7.5,
        )
        print(sorted(wa.columns))
        assert wa.shape[1] == 4  # one temperature_2m__<zone> column per zone
        ```
    """
    weather_frames = []
    aligned_frames = []
    for zone in GERMAN_TSO_ZONE_CITIES:
        locs = locations_for_zone(zone, override=zone_locations_override)
        short = _zone_short(zone)
        try:
            wf_z, wa_z = get_weather_features(
                data=data,
                start=start,
                cov_end=cov_end,
                forecast_horizon=forecast_horizon,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                freq=freq,
                cache_home=cache_home,
                verbose=verbose,
                locations=_coordinates(locs),
                location_weights=_weights(locs),
                derived_features=derived_features,
                hdh_base=hdh_base,
                cdh_base=cdh_base,
            )
        except WeatherFetchError as exc:
            raise WeatherFetchError(
                f"zone_weather_columns fetch failed for zone {zone!r}: {exc}"
            ) from exc
        wf_z = wf_z.add_suffix(f"__{short}")
        wa_z = wa_z.add_suffix(f"__{short}")
        weather_frames.append(wf_z)
        aligned_frames.append(wa_z)

    weather_features_concat = pd.concat(weather_frames, axis=1)
    weather_aligned_concat = pd.concat(aligned_frames, axis=1)
    return weather_features_concat, weather_aligned_concat

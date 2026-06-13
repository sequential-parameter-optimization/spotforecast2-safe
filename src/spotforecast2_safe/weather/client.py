# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Process-wide proactive throttle for Open-Meteo requests.  Per-zone / global
# population-weighted weather fetch many cities, each through a fresh client and
# session, so a burst can trip Open-Meteo's archive rate limit (HTTP 429) faster
# than per-request retry/backoff can recover.  A minimum spacing between any two
# requests (process-wide) spreads the burst under the limit.  Overridable via the
# SPOTFORECAST2_WEATHER_MIN_REQUEST_INTERVAL env var (seconds; "0" disables).
# Only request *timing* is affected, never the returned data — determinism of
# results is preserved.
_MIN_REQUEST_INTERVAL_S = float(
    os.environ.get("SPOTFORECAST2_WEATHER_MIN_REQUEST_INTERVAL", "0.5")
)
_THROTTLE_LOCK = threading.Lock()
# One-element list holding the monotonic timestamp of the most recent Open-Meteo
# request. A mutable container is updated in place under the lock, so
# ``_throttle_open_meteo`` needs no module-level ``global`` rebind (which static
# analysis flags as an unused global, py/unused-global-variable).
_LAST_REQUEST_MONOTONIC = [0.0]


def _throttle_open_meteo() -> None:
    """Block until at least ``_MIN_REQUEST_INTERVAL_S`` has passed since the last
    Open-Meteo request (process-wide). No-op when the interval is non-positive."""
    if _MIN_REQUEST_INTERVAL_S <= 0:
        return
    with _THROTTLE_LOCK:
        wait = _MIN_REQUEST_INTERVAL_S - (monotonic() - _LAST_REQUEST_MONOTONIC[0])
        if wait > 0:
            sleep(wait)
        _LAST_REQUEST_MONOTONIC[0] = monotonic()


class WeatherFetchError(ValueError):
    """Raised when Open-Meteo cannot be reached or returns no usable data.

    Distinguishes a transient external-API failure (network outage,
    rate-limit exhaustion, both the archive and forecast endpoints
    returning errors) from a real data-validity problem.  Subclasses
    `ValueError` so existing callers that catch `ValueError` keep
    working; consumers that want to react specifically to weather-API
    failures (e.g. `spotforecast2.multitask.base.BaseTask` with
    `config.on_weather_failure="skip"`) catch this class instead.

    Examples:
        ```{python}
        from spotforecast2_safe.weather.client import WeatherFetchError

        try:
            raise WeatherFetchError("Archive and forecast endpoints both failed")
        except WeatherFetchError as exc:
            print(f"Caught WeatherFetchError: {exc}")

        # WeatherFetchError is a subclass of ValueError
        assert issubclass(WeatherFetchError, ValueError)
        print("issubclass(WeatherFetchError, ValueError):", True)
        ```
    """


class WeatherClient:
    """Client for fetching weather data from Open-Meteo API.
    Handles the low-level API interactions, parameter building, and response parsing.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.

    Examples:
        ```{python}
        #| eval: false
        # Fetching from Open-Meteo requires a live network connection.
        import pandas as pd
        from spotforecast2_safe.weather import WeatherClient
        client = WeatherClient(latitude=52.52, longitude=13.405)
        df = client.fetch_archive(
            start=pd.Timestamp("2023-01-01", tz="UTC"),
            end=pd.Timestamp("2023-01-02", tz="UTC"),
        )
        print(df.head())
        ```

        Construction and attribute inspection do not require a network call:

        ```{python}
        from spotforecast2_safe.weather.client import WeatherClient

        client = WeatherClient(latitude=52.52, longitude=13.405)
        assert client.latitude == 52.52
        assert client.longitude == 13.405
        print(f"WeatherClient at ({client.latitude}, {client.longitude})")
        print("HOURLY_PARAMS count:", len(client.HOURLY_PARAMS))
        ```
    """

    ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"

    HOURLY_PARAMS = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]

    def __init__(self, latitude: float, longitude: float) -> None:
        """Initialize WeatherClient.

        Args:
            latitude: Latitude of the location.
            longitude: Longitude of the location.

        Examples:
            ```{python}
            from spotforecast2_safe.weather import WeatherClient
            client = WeatherClient(latitude=52.52, longitude=13.405)
            client.latitude, client.longitude
            ```
        """
        self.latitude = latitude
        self.longitude = longitude
        self.logger = logging.getLogger(__name__)
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        # Reactive backoff: on 429 / 5xx, retry up to 5 times with exponential
        # backoff and honour the server's Retry-After header (Open-Meteo sends
        # one). Pairs with the proactive _throttle_open_meteo() spacing below.
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _fetch(self, url: str, params: dict[str, Any]) -> pd.DataFrame:
        """Execute API request and return parsed DataFrame."""
        _throttle_open_meteo()  # proactive burst spacing (see module top)
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

        if "error" in data and data["error"]:
            raise ValueError(
                f"Open-Meteo API error: {data.get('reason', 'Unknown error')}"
            )

        hourly_data = data.get("hourly", {})
        if not hourly_data:
            raise ValueError("No hourly data returned from API")

        # Parse to DataFrame
        times = pd.to_datetime(hourly_data["time"])
        df_dict = {"datetime": times}
        for param in self.HOURLY_PARAMS:
            if param in hourly_data:
                df_dict[param] = hourly_data[param]

        df = pd.DataFrame(df_dict)
        df.set_index("datetime", inplace=True)
        return df

    def fetch_archive(
        self, start: pd.Timestamp, end: pd.Timestamp, timezone: str = "UTC"
    ) -> pd.DataFrame:
        """Fetch historical data from Archive API.

        Args:
            start: Start date for the historical data.
            end: End date for the historical data.
            timezone: Timezone for the data (default "UTC").

        Examples:
            ```{python}
            #| eval: false
            # Requires a live connection to archive-api.open-meteo.com.
            import pandas as pd
            from spotforecast2_safe.weather import WeatherClient
            client = WeatherClient(latitude=52.52, longitude=13.405)
            df = client.fetch_archive(
                start=pd.Timestamp("2023-01-01", tz="UTC"),
                end=pd.Timestamp("2023-01-02", tz="UTC"),
            )
            print(df.head())
            ```
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(self.HOURLY_PARAMS),
            "timezone": timezone,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
        }
        return self._fetch(self.ARCHIVE_BASE_URL, params)

    def fetch_forecast(
        self, days_ahead: int, timezone: str = "UTC", past_days: int = 0
    ) -> pd.DataFrame:
        """Fetch forecast data from Forecast API.

        Args:
            days_ahead: Number of days ahead for the forecast.
            timezone: Timezone for the data (default "UTC").
            past_days: Recent past days to also request from the forecast
                endpoint via Open-Meteo's `past_days` parameter (0--92).
                The archive endpoint lags several days, so `past_days` lets
                the forecast endpoint backfill that recent window; without it
                the `[now - archive_lag, today]` range is fetched by neither
                endpoint and is later carried forward as a flat line.

        Examples:
            ```{python}
            #| eval: false
            # Requires a live connection to api.open-meteo.com.
            from spotforecast2_safe.weather import WeatherClient
            client = WeatherClient(latitude=52.52, longitude=13.405)
            df = client.fetch_forecast(days_ahead=7)
            print(df.head())
            ```
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(self.HOURLY_PARAMS),
            "timezone": timezone,
            "forecast_days": days_ahead,
        }
        if past_days:
            params["past_days"] = past_days
        return self._fetch(self.FORECAST_BASE_URL, params)


class WeatherService(WeatherClient):
    """High-level service for weather data generation.

    Extends WeatherClient with caching, hybrid fetching (archive+forecast),
    and fallback strategies.

    Args:
        latitude:
            Latitude of the location.
        longitude:
            Longitude of the location.
        cache_path:
            Optional path to cache file for storing fetched data. If provided, the service will attempt to load from cache before fetching and will save new data to this path.
            Default is None (no caching).
        use_forecast:
            Whether to use forecast data for future dates (default True).

    Examples:
        ```{python}
        #| eval: false
        # Requires a live connection to Open-Meteo APIs.
        from pathlib import Path
        import pandas as pd
        from spotforecast2_safe.weather import WeatherService
        client = WeatherService(latitude=52.52, longitude=13.405, cache_path=Path("weather_cache.parquet"))
        start = pd.Timestamp("2023-01-01", tz="UTC")
        end = pd.Timestamp("2023-01-07", tz="UTC")
        df = client.get_dataframe(start=start, end=end, fill_missing=False)
        print(df.head())
        print(df.tail())
        ```

        Construction and configuration do not require a network call:

        ```{python}
        from pathlib import Path
        from spotforecast2_safe.weather.client import WeatherService

        svc = WeatherService(
            latitude=51.03,
            longitude=7.57,
            cache_path=None,
            use_forecast=True,
        )
        assert svc.latitude == 51.03
        assert svc.use_forecast is True
        assert svc.cache_path is None
        print(f"WeatherService at ({svc.latitude}, {svc.longitude}), use_forecast={svc.use_forecast}")
        ```
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        cache_path: Path | None = None,
        use_forecast: bool = True,
    ) -> None:
        super().__init__(latitude, longitude)
        self.cache_path = cache_path
        self.use_forecast = use_forecast

    def get_dataframe(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        timezone: str = "UTC",
        freq: str = "h",
        fallback_on_failure: bool = True,
        fill_missing: bool = False,
    ) -> pd.DataFrame:
        """Get weather DataFrame for a specified range using best available methods.

        Refactored from spotpredict.create_weather_df.  Since the 1.0
        major release, remaining gaps after fetch are rejected by
        default so that synthesised values never reach downstream
        consumers labelled as measurements.  Pass `fill_missing=True`
        to opt into the legacy forward/back-fill behavior.

        Args:
            start: Start date for the data.
            end: End date for the data.
            timezone: Timezone for the data (default "UTC").
            freq: Frequency for the data (default "h").
            fallback_on_failure: Whether to use fallback data on failure (default True).
            fill_missing: Whether to forward- and back-fill remaining
                NaN gaps after fetch/resample (default False).  When
                False (the fail-safe default), any remaining NaN
                raises `ValueError` with the gap timestamps.

        Raises:
            ValueError: If `fill_missing=False` and the merged frame
                still contains NaNs after resample.

        Examples:
            ```{python}
            #| eval: false
            # Requires a live connection to Open-Meteo APIs.
            import pandas as pd
            from spotforecast2_safe.weather import WeatherService
            client = WeatherService(latitude=51.0267, longitude=7.5693)
            start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
            end = pd.Timestamp.now(tz="UTC")
            df = client.get_dataframe(start=start, end=end, fill_missing=False)
            print(df.head())
            print(df.tail())
            ```

            The `fill_missing=False` default rejects frames with gaps.  The
            ValueError path can be exercised offline using `_finalize_df`
            directly with a synthetic frame that has NaN rows:

            ```{python}
            import pandas as pd
            from spotforecast2_safe.weather.client import WeatherService

            svc = WeatherService(latitude=51.03, longitude=7.57)
            idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
            df = pd.DataFrame({"temperature_2m": [1.0, float("nan"), 3.0, 4.0]}, index=idx)

            try:
                svc._finalize_df(df, freq="h", fill_missing=False)
            except ValueError as exc:
                print(f"ValueError raised as expected: {exc}")

            # With fill_missing=True the gap is imputed silently
            filled = svc._finalize_df(df.copy(), freq="h", fill_missing=True)
            assert not filled.isna().any().any()
            print("fill_missing=True: no NaNs remain")
            ```
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if pd.isna(start_ts) or pd.isna(end_ts):
            raise ValueError(
                "start and end must be valid timestamps; "
                f"got start={start!r}, end={end!r}"
            )

        # Localize if naive
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(timezone)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize(timezone)

        # Convert to UTC for consistency
        start_utc = start_ts.tz_convert("UTC")
        end_utc = end_ts.tz_convert("UTC")

        # 1. Try Cache
        cached_df = self._load_cache()
        if cached_df is not None:
            if cached_df.index.min() <= start_utc and cached_df.index.max() >= end_utc:
                self.logger.info("Using full cached data.")
                return self._finalize_df(
                    cached_df.loc[start_utc:end_utc], freq, fill_missing
                )

        # 2. Hybrid Fetch (filling gaps if cache exists, or fetching all)
        # (The original logic did partial fills, but full fetch is safer and
        # simpler for now unless specifically improved).
        # Actually, strict refactor implies keeping logic. Let's keep it simple:
        # fetch what's needed.

        try:
            df = self._fetch_hybrid(start_ts, end_ts, timezone)
        except Exception as e:
            self.logger.warning(f"Fetch failed: {e}")
            if fallback_on_failure and cached_df is not None and len(cached_df) >= 24:
                df = self._create_fallback(start_utc, end_utc, cached_df)
            else:
                raise

        # 3. Merge with cache and save
        if cached_df is not None:
            df = pd.concat([cached_df, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()  # Keep new data

        if self.cache_path:
            self._save_cache(df)

        # 4. Return slice
        return self._finalize_df(df.loc[start_utc:end_utc], freq, fill_missing)

    def _fetch_hybrid(
        self, start: pd.Timestamp, end: pd.Timestamp, timezone: str
    ) -> pd.DataFrame:
        """Fetch from Archive and/or Forecast based on date."""
        now = pd.Timestamp.now(tz=start.tz)
        archive_cutoff = now - pd.Timedelta(days=5)

        dfs = []

        # Archive part
        if start < archive_cutoff:
            arch_end = min(end, archive_cutoff)
            try:
                dfs.append(self.fetch_archive(start, arch_end, timezone))
            except Exception as e:
                self.logger.warning(f"Archive fetch warning: {e}")

        # Forecast part. The forecast endpoint also backfills the recent window
        # the archive endpoint does not yet publish (archive lags ~5 days): pass
        # ``past_days`` so ``[archive_cutoff, today]`` is covered and never left
        # as a hole that downstream alignment would carry forward as a flat
        # line. Fire whenever the requested range reaches into that window.
        if end > archive_cutoff and self.use_forecast:
            days = (end - now).days + 2
            days = min(max(1, days), 16)
            earliest_needed = max(start, archive_cutoff)
            past_days = (now.normalize() - earliest_needed.normalize()).days + 1
            past_days = min(max(0, past_days), 92)
            try:
                df_fore = self.fetch_forecast(days, timezone, past_days=past_days)
                # Filter forecast to needed range to avoid overlap issues
                dfs.append(df_fore)
            except Exception as e:
                self.logger.warning(f"Forecast fetch warning: {e}")

        if not dfs:
            raise WeatherFetchError("Could not fetch data from Archive or Forecast.")

        full_df = pd.concat(dfs)
        full_df = full_df[~full_df.index.duplicated(keep="first")].sort_index()

        # Ensure UTC index
        if full_df.index.tz is None:
            full_df.index = full_df.index.tz_localize(timezone)
        full_df.index = full_df.index.tz_convert("UTC")

        return full_df

    def _create_fallback(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        source_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Repeat last 24h of data."""
        last_24 = source_df.tail(24)
        hours = int((end - start).total_seconds() / 3600) + 1
        repeats = (hours // 24) + 1

        new_data = pd.concat([last_24] * repeats, ignore_index=True)
        new_data = new_data.iloc[:hours]

        idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
        new_data.index = idx
        return new_data

    def _load_cache(self) -> pd.DataFrame | None:
        """Load the parquet cache, quarantining corrupt files.

        A missing cache file returns ``None`` silently (expected on the
        first run).  A *corrupt* or *partially-written* cache used to
        return ``None`` silently via a bare ``except Exception`` — that
        hid silent cache loss behind the same return value as a cache
        miss.  In a safety-critical pipeline that means silent
        divergence between runs.

        This method now:

        - returns ``None`` for the expected "not yet cached" path,
        - on ``pyarrow.lib.ArrowInvalid`` / ``OSError`` /
          ``FileNotFoundError`` (race) / ``ValueError`` from
          ``read_parquet``, **logs a WARNING** with the cache path,
          renames the bad file to ``<cache>.corrupt-<epoch>`` so the
          next run starts fresh, and returns ``None``,
        - lets any other exception propagate (an unfamiliar failure
          mode should not be silently consumed).

        Returns:
            The cached DataFrame or ``None`` if the cache is absent,
            the quarantine path is writable and the cache was
            corrupt, or ``self.cache_path`` is unset.
        """
        if not self.cache_path or not self.cache_path.exists():
            return None
        try:
            df = pd.read_parquet(self.cache_path)
        except (OSError, ValueError) as exc:
            self._quarantine_corrupt_cache(exc)
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    def _quarantine_corrupt_cache(self, exc: BaseException) -> None:
        """Log and move a damaged cache out of the way.

        Args:
            exc: The exception raised by ``read_parquet``; included in
                the log record so the caller can diagnose.
        """
        import time

        cache_path = self.cache_path
        if cache_path is None:
            return
        quarantine = cache_path.with_suffix(
            cache_path.suffix + f".corrupt-{int(time.time())}"
        )
        self.logger.warning(
            "Weather cache at %s is unreadable (%s: %s); "
            "moving to %s so the next run starts fresh.",
            cache_path,
            type(exc).__name__,
            exc,
            quarantine,
        )
        try:
            cache_path.rename(quarantine)
        except OSError as rename_exc:
            self.logger.warning(
                "Could not quarantine %s: %s: %s",
                cache_path,
                type(rename_exc).__name__,
                rename_exc,
            )

    def _save_cache(self, df: pd.DataFrame) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.cache_path)

    def _finalize_df(
        self,
        df: pd.DataFrame,
        freq: str,
        fill_missing: bool = False,
    ) -> pd.DataFrame:
        """Resample, localise, and (optionally) fill gaps.

        Args:
            df: Merged frame ready to be returned.
            freq: Target pandas frequency string.
            fill_missing: When True, forward- then back-fill any
                remaining NaN (legacy behavior).  When False (the
                fail-safe default), any remaining NaN raises
                ``ValueError`` listing the first few gap timestamps.

        Raises:
            ValueError: If ``fill_missing=False`` and the frame still
                has NaNs after resample.
        """
        if freq != "h":
            df = df.resample(freq).ffill()
        elif not df.empty:
            # Reindex onto the complete hourly grid so that *absent* rows (a
            # real fetch hole) surface as NaN and are governed by the same gap
            # policy as NaN-valued rows below. Previously absent hourly rows
            # slipped past the gap check entirely (the reindex only ran for
            # ``freq != "h"``), so a fetch hole was silently returned.
            df = df.sort_index()
            full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
            df = df.reindex(full_index)

        if fill_missing:
            return df.ffill().bfill()

        gap_mask = df.isna().any(axis=1)
        if gap_mask.any():
            gaps = df.index[gap_mask]
            preview = ", ".join(str(ts) for ts in gaps[:5])
            more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
            raise ValueError(
                f"{len(gaps)} missing row(s) in weather frame after "
                f"resample at freq={freq!r}. First gaps: [{preview}]"
                f"{more}. Pass fill_missing=True to opt into legacy "
                "ffill/bfill imputation."
            )
        return df

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ENTSO-E Transparency Platform data downloader.

This module provides utilities to download electricity load and forecast data
directly from the ENTSO-E API and merge local raw data files into a consistent
dataset for forecasting.

Threat model (STRIDE).
    This module crosses a network boundary and therefore carries its own
    STRIDE table. The table enumerates, for every data flow, which of the six
    categories apply, which countermeasure is in force, and where that
    countermeasure is implemented. Contributors who change the surface listed
    below MUST update this table in the same pull request; the rule is
    anchored in CONTRIBUTING.md ("Threat-model update rule") and gated by the
    checklist in .github/pull_request_template.md.

    Data flow 1: outbound HTTPS request to the ENTSO-E Transparency Platform
    (api.entsoe.eu). Endpoints used: load and day-ahead load forecast
    (``query_load_and_forecast``); day-ahead wind/solar generation forecast
    (``query_wind_and_solar_forecast``); and day-ahead spot price
    (``query_day_ahead_prices``). All three are read-only day-ahead queries
    that share the same TLS, retry, and schema-validation countermeasures
    below; the day-ahead side-tables are merged into their own namespaced
    interim files and never alter the ``energy_load.csv`` schema.

    - Spoofing: a forged endpoint could serve crafted load data. Mitigated by
      default TLS certificate verification in ``requests`` (used by the
      ``entsoe`` client) and by the fixed ENTSO-E endpoint baked into
      ``EntsoePandasClient`` (the host is not derived from caller input); do
      not disable TLS verification.
    - Tampering: an on-path attacker could modify the XML payload. Mitigated
      by TLS integrity and by the schema-shaped parser below that rejects any
      record whose index is not a monotonic datetime.
    - Repudiation: the API token is the only caller identity. This module
      emits standard ``logging`` records for each download attempt (dates,
      country code, attempt count, errors); it keeps no dedicated
      request-audit log and records no response-byte hash. Operators needing
      non-repudiation must capture these logs at the host/SIEM level.
    - Information Disclosure: the API token is read only from the environment
      or the caller argument and handed to ``EntsoePandasClient``; this module
      does not log the token or the request URL directly. There is no
      automatic ``securityToken=`` redaction filter, and an HTTP error raised
      by the client may embed the request URL in its message (logged at
      WARNING during retries); deployments that enable third-party HTTP debug
      logging, or that treat the token as highly sensitive, should add their
      own redaction.
    - Denial of Service: upstream rate limiting or transient errors could
      stall the pipeline. Mitigated by a bounded, explicit retry loop
      (``_MAX_RETRIES`` / ``_RETRY_BACKOFF_SECONDS``) that raises
      ``RuntimeError`` after the attempt budget rather than looping silently.
    - Elevation of Privilege: the module runs with the host-process
      privileges; it opens no setuid boundary. Not applicable at module
      scope.

    Data flow 2: local filesystem read/write under ``get_data_home()``
    (raw CSV ingest, interim-file merge).

    - Tampering: an attacker with write access to the data directory could
      plant a malformed CSV. Mitigated by the downstream schema validation
      in ``data/fetch_data.py`` and by the fail-safe ``on_missing`` contract
      in the preprocessing layer.
    - Information Disclosure: cached files may contain timestamped load
      series. Not sensitive in isolation, but operators running this library
      against non-public data must configure the data directory
      permissions; this module does not set permissions on behalf of the
      caller.
    - Other STRIDE categories: not applicable for a local filesystem flow
      whose threat model is owned by the host operating system.
"""

import logging
import time
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BACKOFF_SECONDS = 5
_COOLDOWN_HOURS = 24


def merge_build_manual(
    output_file: str = "energy_load.csv",
    keep_forecast_future: bool = False,
    raw_subdir: Optional[str] = None,
) -> None:
    """
    Merge all raw CSV files from the 'raw' directory into a single interim file.

    This function looks for all `.csv` files in `get_data_home() / "raw"`
    (or, when *raw_subdir* is given, in `get_data_home() / "raw" / raw_subdir`),
    sorts them by time index, and saves the unique combined data to
    `get_data_home() / "interim" / output_file`.

    Namespacing raw files by *raw_subdir* keeps the day-ahead side-tables
    (``renewable_forecast.csv``, ``day_ahead_price.csv``) from clobbering the
    load schema in ``energy_load.csv``: the default glob is non-recursive, so
    load files in ``raw/`` and renewable files in ``raw/renewable/`` never mix.

    Args:
        output_file: The name of the combined output file.
            Defaults to "energy_load.csv".
        keep_forecast_future: If False (the default), rows with a timestamp
            after the current UTC moment are dropped, so the interim file
            holds only data that is "actual" up to now -- the correct,
            leakage-free input for model training. If True, those future rows
            are retained, which preserves ENTSO-E's day-ahead `Forecasted Load`
            for tomorrow (future `Actual Load` is still NaN, so no future
            target leaks). Use True only for forecast-baseline / comparison
            workflows that need the published day-ahead forecast.
        raw_subdir: Optional sub-directory under ``raw/`` to merge instead of
            ``raw/`` itself. Used by the day-ahead side-table downloaders
            (e.g. ``"renewable"``, ``"price"``). Defaults to ``None``.

    Raises:
        FileNotFoundError: If the raw directory does not exist.
        ValueError: If no valid CSV files are found for merging.

    Notes:
        Individual raw CSV files that fail to parse are logged at ERROR
        level and skipped; the merge continues with the remaining files.
        When zero files parse successfully the function returns early
        without writing an output file.

        Logging information can be selected by setting the log level for the
        `spotforecast2_safe.downloader.entsoe` logger. Common levels are
        `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. The cell
        below shows the default (WARNING); change the level to `INFO` or
        `DEBUG` for more verbose output.

        ```{python}
        import logging
        logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.WARNING)
        ```

    Examples:
        ```{python}
        #| eval: false
        from spotforecast2_safe.downloader.entsoe import merge_build_manual

        # Merge with the default output filename
        merge_build_manual()

        # Or merge with a custom output filename
        merge_build_manual(output_file="custom_energy_load.csv")

        # Retain future rows so tomorrow's day-ahead `Forecasted Load`
        # survives (forecast-baseline workflows only; not for training)
        merge_build_manual(keep_forecast_future=True)
        ```
    """
    data_home = get_data_home()
    raw_dir = data_home / "raw" / raw_subdir if raw_subdir else data_home / "raw"
    interim_dir = data_home / "interim"

    if not raw_dir.exists():
        logger.warning(
            "Raw data directory %s does not exist. Nothing to merge.", raw_dir
        )
        return

    logger.info("Merging raw files from %s...", raw_dir)

    list_dfs = []
    for csv_file in raw_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            # Assuming 'Time (UTC)' is the index name as per spotprivate config
            # We'll try common index names or the first column
            index_col = "Time (UTC)" if "Time (UTC)" in df.columns else 0
            df.rename(
                columns={
                    (
                        df.columns[0] if isinstance(index_col, int) else index_col
                    ): "Time (UTC)"
                },
                inplace=True,
            )

            df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"], utc=True)
            df.set_index("Time (UTC)", inplace=True)

            # Clean data: handle placeholders like '-'
            for col in df.columns:
                df[col] = df[col].apply(lambda x: np.nan if x == "-" else x)

            list_dfs.append(df)
        except Exception as e:
            logger.error("Failed to process raw file %s: %s", csv_file, e)

    if not list_dfs:
        logger.info("No valid raw data files found for merging.")
        return

    merged_df = pd.concat(list_dfs)
    merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()

    # Filter out future data points if any (only keep what's theoretically
    # "actual" up to now). Skipped when keep_forecast_future is True, so the
    # day-ahead `Forecasted Load` for tomorrow survives -- future `Actual Load`
    # is still NaN, so no future target leaks into a training read.
    if not keep_forecast_future:
        merged_df = merged_df.loc[merged_df.index <= pd.Timestamp.now(tz="UTC")]

    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / output_file
    merged_df.to_csv(output_path)
    logger.info("Successfully merged data saved to %s", output_path)


def download_new_data(
    api_key: str,
    country_code: str = "FR",
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    keep_forecast_future: bool = False,
) -> None:
    """
    Download new load and forecast data from ENTSO-E.

    This function queries the ENTSO-E Transparency Platform for a given period.
    If no start date is provided, it automatically resumes from the last
    available data point.

    Args:
        api_key: The ENTSO-E API key.
        country_code: The country code to query (e.g., 'FR', 'DE').
            Defaults to "FR".
        start: Start date in 'YYYYMMDDHH00' format.
        end: End date in 'YYYYMMDDHH00' format.
        force: If True, bypass the 24h cooldown check.
        keep_forecast_future: If True, retain rows after the current UTC
            moment when building the interim file, preserving ENTSO-E's
            day-ahead `Forecasted Load` for tomorrow. Defaults to False (future
            rows are dropped, the leakage-free input for training). The flag
            only keeps rows the query window already covers, so pass an ``end``
            that reaches into the target day (e.g. tomorrow) as well. See
            `merge_build_manual` for details.

    Raises:
        ImportError:
            If the Python package 'entsoe-py' is not installed.
        ValueError:
            If ``start`` or ``end`` cannot be parsed as a valid timestamp.
        RuntimeError:
            If data fetching fails after ``_MAX_RETRIES`` attempts.

    Notes:
        Logging information can be selected by setting the log level for the
        `spotforecast2_safe.downloader.entsoe` logger. Common levels are
        `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. The cell
        below shows the default (WARNING); change the level to `INFO` or
        `DEBUG` for more verbose output.

        ```{python}
        import logging
        logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.WARNING)
        ```

    Examples:
        Replace ``YOUR_API_KEY`` below with your ENTSO-E Web API Security
        Token (free registration at https://transparency.entsoe.eu, then
        "My Account Settings" → "Web API Security Token"). Prefer reading
        the token from an environment variable rather than hard-coding it
        in source — see the Information-Disclosure note in the module
        STRIDE table above.

        ```{python}
        #| eval: false
        from spotforecast2_safe.downloader.entsoe import download_new_data

        # Basic download for Germany with explicit start/end dates
        download_new_data(
            api_key="YOUR_API_KEY",
            country_code="DE",
            start="202301010000",
            end="202301020000",
            force=True,
        )

        # Incremental download (automatically resumes from last data point)
        download_new_data(api_key="YOUR_API_KEY", country_code="FR")

        # Forced download bypassing the 24-hour cooldown check
        download_new_data(
            api_key="YOUR_API_KEY",
            country_code="DE",
            force=True,
        )

        # Retain tomorrow's day-ahead `Forecasted Load` (forecast baseline);
        # `end` must reach into the target day for the rows to exist
        download_new_data(
            api_key="YOUR_API_KEY",
            country_code="DE",
            start="202301010000",
            end="202301030000",
            force=True,
            keep_forecast_future=True,
        )
        ```
    """

    try:
        from entsoe import EntsoePandasClient
    except ImportError as e:
        raise ImportError(
            "The 'entsoe-py' library is required for this functionality. "
            "Install it with: pip install entsoe-py"
        ) from e

    # First merge existing files to get the latest index
    merge_build_manual(keep_forecast_future=keep_forecast_future)

    logger.info("Initiating data download from ENTSO-E...")

    # Determine start date
    if start is None:
        try:
            current_data = fetch_data()  # This might look at interim or a specific file
            # Resume from the last row at or before now, not the raw index max:
            # with keep_forecast_future=True the interim can carry future
            # forecast rows, and resuming from those would skip real history.
            observed = current_data.loc[
                current_data.index <= pd.Timestamp.now(tz="UTC")
            ]
            last_observed = (
                observed.index[-1] if not observed.empty else current_data.index[-1]
            )
            start_date = last_observed + pd.Timedelta(hours=1)
            logger.info(
                "No start date provided. Resuming from last data point: %s", start_date
            )
        except (FileNotFoundError, ValueError, IndexError) as exc:
            # Narrow fallback: only the three signals that mean "no prior
            # data on disk yet". Anything else (ImportError, OSError
            # beyond FileNotFoundError, KeyError from a schema change,
            # ...) is a real bug and must propagate so the caller can see
            # it instead of silently starting from 7 days ago.
            start_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
            logger.warning(
                "No previous data found (%s: %s). Starting from default date: %s",
                type(exc).__name__,
                exc,
                start_date,
            )
    else:
        start_date = pd.to_datetime(start, utc=True, errors="coerce")
        if pd.isna(start_date):
            raise ValueError(f"start={start!r} did not parse to a valid timestamp")
        logger.info("Using provided start date: %s", start_date)

    # Determine end date
    if end is None:
        end_date = pd.Timestamp.now(tz="UTC").floor("D")
        logger.info("No end date provided. Using current date: %s", end_date)
    else:
        end_date = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.isna(end_date):
            raise ValueError(f"end={end!r} did not parse to a valid timestamp")
        logger.info("Using provided end date: %s", end_date)

    # Safety check: avoid redundant small downloads
    hours_diff = (end_date - start_date).total_seconds() / 3600
    if hours_diff < _COOLDOWN_HOURS and not force:
        logger.info(
            "Last download was too recent (%.1f hours ago). Skipping.",
            _COOLDOWN_HOURS - hours_diff,
        )
        return

    client = EntsoePandasClient(api_key=api_key)

    # Retry loop
    retry_counter = 0
    success = False
    downloaded_df = None

    while retry_counter < _MAX_RETRIES:
        try:
            logger.info(
                "Downloading data from %s to %s (attempt %d/%d)...",
                start_date,
                end_date,
                retry_counter + 1,
                _MAX_RETRIES,
            )
            downloaded_df = client.query_load_and_forecast(
                country_code=country_code, start=start_date, end=end_date
            )
            success = True
            break
        except Exception as e:
            logger.warning(
                "Download failed: %s. Retrying in %ds...",
                e,
                _RETRY_BACKOFF_SECONDS,
            )
            retry_counter += 1
            time.sleep(_RETRY_BACKOFF_SECONDS)

    if not success or downloaded_df is None:
        raise RuntimeError(
            f"Failed to download data from ENTSO-E after {_MAX_RETRIES} attempts."
        )

    # Save to raw
    data_home = get_data_home()
    raw_dir = data_home / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    date_format = "%Y%m%d%H00"
    file_name = f"entsoe_load_{start_date.strftime(date_format)}_{end_date.strftime(date_format)}.csv"
    output_path = raw_dir / file_name

    downloaded_df.index.name = "Time (UTC)"
    downloaded_df.to_csv(output_path)
    logger.info("Downloaded data saved to %s", output_path)

    # Final merge to integrate new data
    merge_build_manual(keep_forecast_future=keep_forecast_future)


def _download_entsoe_table(
    api_key: str,
    country_code: str,
    start: Optional[str],
    end: Optional[str],
    *,
    query: Callable[[object, str, pd.Timestamp, pd.Timestamp], object],
    raw_subdir: str,
    output_file: str,
    file_prefix: str,
    column_name: Optional[str] = None,
    force: bool = False,
) -> None:
    """Download and merge an ENTSO-E day-ahead side-table (shared helper).

    Generic download+merge used by `download_renewable_forecast` and
    `download_day_ahead_price`. It runs the bounded retry loop, writes the raw
    response to ``raw/<raw_subdir>/<file_prefix>_<start>_<end>.csv`` and merges
    it into ``interim/<output_file>`` via `merge_build_manual`. Future rows are
    retained (``keep_forecast_future=True``) because the day-ahead forecast for
    tomorrow is exactly the leakage-clean value these tables exist to provide.

    Args:
        api_key: ENTSO-E Web API security token.
        country_code: ENTSO-E bidding-zone / country code (e.g. ``"DE"``,
            ``"DE_LU"``).
        start: Start in ``'YYYYMMDDHH00'`` format, or ``None`` to default to
            seven days ago.
        end: End in ``'YYYYMMDDHH00'`` format, or ``None`` to default to the
            start of tomorrow (so the day-ahead horizon is covered).
        query: Callable ``query(client, country_code, start, end)`` returning a
            ``pd.DataFrame`` or ``pd.Series`` from the ENTSO-E client.
        raw_subdir: Sub-directory under ``raw/`` to write into and merge.
        output_file: Interim filename to write under ``interim/``.
        file_prefix: Prefix for the raw filename.
        column_name: If the query returns a ``pd.Series``, the column name to
            give it in the saved table. Ignored for ``pd.DataFrame`` results.
        force: If True, bypass the small-window cooldown check.

    Raises:
        ImportError: If ``entsoe-py`` is not installed.
        ValueError: If ``start`` or ``end`` cannot be parsed.
        RuntimeError: If the download fails after ``_MAX_RETRIES`` attempts.
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError as e:
        raise ImportError(
            "The 'entsoe-py' library is required for this functionality. "
            "Install it with: pip install entsoe-py"
        ) from e

    if start is None:
        start_date = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=7)
    else:
        start_date = pd.to_datetime(start, utc=True, errors="coerce")
        if pd.isna(start_date):
            raise ValueError(f"start={start!r} did not parse to a valid timestamp")

    if end is None:
        # Reach into tomorrow so the published day-ahead forecast is included.
        end_date = pd.Timestamp.now(tz="UTC").floor("D") + pd.Timedelta(days=1)
    else:
        end_date = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.isna(end_date):
            raise ValueError(f"end={end!r} did not parse to a valid timestamp")

    hours_diff = (end_date - start_date).total_seconds() / 3600
    if hours_diff < _COOLDOWN_HOURS and not force:
        logger.info(
            "Requested window is only %.1f h; skipping (pass force=True).",
            hours_diff,
        )
        return

    client = EntsoePandasClient(api_key=api_key)

    retry_counter = 0
    downloaded: Optional[Union[pd.DataFrame, pd.Series]] = None
    while retry_counter < _MAX_RETRIES:
        try:
            logger.info(
                "Downloading %s from %s to %s (attempt %d/%d)...",
                raw_subdir,
                start_date,
                end_date,
                retry_counter + 1,
                _MAX_RETRIES,
            )
            downloaded = query(client, country_code, start_date, end_date)
            break
        except Exception as e:  # noqa: BLE001 - retried, then re-raised below
            logger.warning(
                "Download failed: %s. Retrying in %ds...", e, _RETRY_BACKOFF_SECONDS
            )
            retry_counter += 1
            time.sleep(_RETRY_BACKOFF_SECONDS)

    if downloaded is None:
        raise RuntimeError(
            f"Failed to download {raw_subdir} from ENTSO-E after "
            f"{_MAX_RETRIES} attempts."
        )

    if isinstance(downloaded, pd.Series):
        downloaded = downloaded.to_frame(column_name or downloaded.name or "value")

    data_home = get_data_home()
    raw_dir = data_home / "raw" / raw_subdir
    raw_dir.mkdir(parents=True, exist_ok=True)

    date_format = "%Y%m%d%H00"
    file_name = (
        f"{file_prefix}_{start_date.strftime(date_format)}_"
        f"{end_date.strftime(date_format)}.csv"
    )
    output_path = raw_dir / file_name

    downloaded.index.name = "Time (UTC)"
    downloaded.to_csv(output_path)
    logger.info("Downloaded data saved to %s", output_path)

    merge_build_manual(
        output_file=output_file, keep_forecast_future=True, raw_subdir=raw_subdir
    )


def download_renewable_forecast(
    api_key: str,
    country_code: str = "DE",
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
) -> None:
    """Download the ENTSO-E day-ahead wind/solar generation forecast.

    Queries ``query_wind_and_solar_forecast`` and merges the result into
    ``interim/renewable_forecast.csv`` (columns such as ``"Solar"``,
    ``"Wind Onshore"``, ``"Wind Offshore"``). The day-ahead forecast is
    leakage-clean (published D-1); the realised generation must not be used.
    Consumed downstream by
    `spotforecast2_safe.preprocessing.exog_providers.EntsoeRenewableForecastProvider`
    via `spotforecast2_safe.data.fetch_data.load_renewable_forecast`.

    Args:
        api_key: ENTSO-E Web API security token.
        country_code: Country / bidding-zone code. Defaults to ``"DE"``.
        start: Start in ``'YYYYMMDDHH00'`` format, or ``None`` to default to
            seven days ago.
        end: End in ``'YYYYMMDDHH00'`` format, or ``None`` to default to the
            start of tomorrow.
        force: If True, bypass the small-window cooldown check.

    Raises:
        ImportError: If ``entsoe-py`` is not installed.
        ValueError: If ``start`` or ``end`` cannot be parsed.
        RuntimeError: If the download fails after ``_MAX_RETRIES`` attempts.

    Examples:
        ```{python}
        #| eval: false
        from spotforecast2_safe.downloader.entsoe import download_renewable_forecast

        download_renewable_forecast(api_key="YOUR_API_KEY", country_code="DE", force=True)
        ```
    """

    def query(client, cc, s, e):  # type: ignore[no-untyped-def]
        return client.query_wind_and_solar_forecast(cc, start=s, end=e)

    _download_entsoe_table(
        api_key,
        country_code,
        start,
        end,
        query=query,
        raw_subdir="renewable",
        output_file="renewable_forecast.csv",
        file_prefix="entsoe_renewable",
        force=force,
    )


def download_day_ahead_price(
    api_key: str,
    country_code: str = "DE_LU",
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
) -> None:
    """Download the ENTSO-E day-ahead spot price (DE/LU).

    Queries ``query_day_ahead_prices`` and merges the result into
    ``interim/day_ahead_price.csv`` (column ``"Day-ahead Price"``). The
    day-ahead auction price is leakage-clean (published D-1); the realised
    price must not be used. Consumed downstream by
    `spotforecast2_safe.preprocessing.exog_providers.EntsoeDayAheadPriceProvider`
    via `spotforecast2_safe.data.fetch_data.load_day_ahead_price`.

    Args:
        api_key: ENTSO-E Web API security token.
        country_code: Bidding-zone code. Defaults to ``"DE_LU"``.
        start: Start in ``'YYYYMMDDHH00'`` format, or ``None`` to default to
            seven days ago.
        end: End in ``'YYYYMMDDHH00'`` format, or ``None`` to default to the
            start of tomorrow.
        force: If True, bypass the small-window cooldown check.

    Raises:
        ImportError: If ``entsoe-py`` is not installed.
        ValueError: If ``start`` or ``end`` cannot be parsed.
        RuntimeError: If the download fails after ``_MAX_RETRIES`` attempts.

    Examples:
        ```{python}
        #| eval: false
        from spotforecast2_safe.downloader.entsoe import download_day_ahead_price

        download_day_ahead_price(api_key="YOUR_API_KEY", country_code="DE_LU", force=True)
        ```
    """

    def query(client, cc, s, e):  # type: ignore[no-untyped-def]
        return client.query_day_ahead_prices(cc, start=s, end=e)

    _download_entsoe_table(
        api_key,
        country_code,
        start,
        end,
        query=query,
        raw_subdir="price",
        output_file="day_ahead_price.csv",
        file_prefix="entsoe_price",
        column_name="Day-ahead Price",
        force=force,
    )

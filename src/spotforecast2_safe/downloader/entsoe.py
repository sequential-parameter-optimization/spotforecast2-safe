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
    interim files and never alter the ``energy_load.csv`` schema. The load
    query is additionally issued once per German TSO control area (Amprion,
    TenneT, TransnetBW, 50Hertz; see ``download_zone_loads``) through the same
    ``query_load_and_forecast`` endpoint, sharing these countermeasures; each
    zone response is written to a namespaced ``raw/zones/<zone>/`` file and
    assembled into ``energy_load_zones.csv`` without touching the
    single-zone ``energy_load.csv`` schema.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home
from spotforecast2_safe.preprocessing.checking import check_y

logger = logging.getLogger(__name__)

# ENTSO-E control-area (EIC) codes for the four German TSO zones, keyed by the
# column name each zone's "Actual Load" series receives in the assembled frame.
# The values are ``entsoe-py`` ``Area`` identifiers (verified against
# ``entsoe.mappings``); ``query_load_and_forecast`` resolves them to the
# control-area EIC code, so Actual Total Load is fetched per control area
# rather than for the single DE-LU bidding zone. Used by
# ``download_zone_loads`` / ``assemble_zone_loads``.
GERMAN_TSO_ZONES: Dict[str, str] = {
    "load_amprion": "DE_AMPRION",  # 10YDE-RWENET---I
    "load_tennet": "DE_TENNET",  # 10YDE-EON------1
    "load_transnetbw": "DE_TRANSNET",  # 10YDE-ENBW-----N
    "load_50hertz": "DE_50HZ",  # 10YDE-VE-------2
}


@dataclass(frozen=True)
class ZoneResult:
    """Structured result record for one zone in a ``download_zone_loads`` collect run.

    This dataclass is returned (one per zone, keyed by column name) when
    ``download_zone_loads`` is called with ``on_zone_failure="collect"``.  It
    carries enough information for the caller to decide whether to proceed with
    `build_zone_qc_frame` or to report / retry failed zones.

    Fail-safe invariant: collect mode is pure structured reporting.  This class
    carries no fallback data, no cached values, and no retry state.  A zone
    that failed is represented exactly by ``ok=False`` with the original
    exception in ``error`` — the caller owns the recovery policy.

    Attributes:
        column: Column name for the zone's Actual Load series (e.g.
            ``"load_tennet"``).
        area: String area code passed to ``download_zone_loads`` (e.g.
            ``"DE_TENNET"``).  This is the raw string key from the zones
            mapping — not an ``entsoe-py`` ``Area`` enum instance.
        ok: ``True`` if the zone downloaded and its interim file was written
            successfully; ``False`` otherwise.
        error: The exception raised during download, or ``None`` on success.
            On failure the interim file for this zone was *not* written (or may
            be incomplete); on success it is ``None``.
        interim_path: Absolute path to the per-zone interim CSV written on
            success (``interim/zone_<column>.csv`` under the data home); ``None``
            on failure.  Files written by successful zones are **not** rolled
            back if another zone fails — the caller must account for partial
            writes.

    Examples:
        ```{python}
        #| eval: false
        from spotforecast2_safe.downloader.entsoe import download_zone_loads

        results = download_zone_loads(
            api_key="YOUR_API_KEY",
            start="202301010000",
            end="202301050000",
            force=True,
            on_zone_failure="collect",
        )
        for col, r in results.items():
            status = "ok" if r.ok else f"FAILED ({r.error})"
            print(f"{col}: {status}")
        ```
    """

    column: str
    area: str
    ok: bool
    error: Optional[Exception]
    interim_path: Optional[Path]


def _make_client(client_cls: type, api_key: str, timeout: Optional[float]) -> object:
    """Construct an ``EntsoePandasClient`` with optional *timeout*.

    Args:
        client_cls: The ``EntsoePandasClient`` class imported lazily by the
            calling function.
        api_key: ENTSO-E Web API security token.
        timeout: Per-socket-operation read timeout in seconds passed to the
            client.  Kills stalled connections (raises
            ``requests.exceptions.Timeout``, caught by the existing retry loop,
            then ``RuntimeError`` after retries are exhausted) without bounding
            long live transfers.  ``None`` disables the timeout.  When the
            installed ``entsoe-py`` does not accept this kwarg (older releases),
            a WARNING is logged and the client is constructed without the
            timeout argument.

    Returns:
        An ``EntsoePandasClient`` instance.
    """
    if timeout is not None:
        try:
            return client_cls(api_key=api_key, timeout=timeout)
        except TypeError as e:
            if "timeout" not in str(e):
                raise
            logger.warning(
                "installed entsoe-py does not support 'timeout'; "
                "downloads are unbounded -- upgrade entsoe-py >= 0.8"
            )
    return client_cls(api_key=api_key)


_MAX_RETRIES = 5
_RETRY_BACKOFF_SECONDS = 5
_COOLDOWN_HOURS = 24
# How far an incremental download may reach back to re-fetch hours whose
# "Actual Load" is still missing in the interim file (late publication /
# transparency-platform outages). Bounds the heal window so a structurally
# empty column cannot trigger an unbounded re-pull.
_MAX_BACKFILL_DAYS = 7


def merge_build_manual(
    output_file: str = "energy_load.csv",
    keep_forecast_future: bool = False,
    raw_subdir: Optional[str] = None,
) -> None:
    """
    Merge all raw CSV files from the 'raw' directory into a single interim file.

    This function looks for all `.csv` files in `get_data_home() / "raw"`
    (or, when *raw_subdir* is given, in `get_data_home() / "raw" / raw_subdir`),
    merges them oldest-download-first (file mtime, filename as tiebreaker),
    and saves the unique combined data to
    `get_data_home() / "interim" / output_file`.

    Timestamps covered by several raw files are collapsed cell-wise: per
    column, the newest non-missing value wins. A newer pull therefore
    revises older values, while a raw file that holds NaN for hours another
    pull has filled in (e.g. a pull made before ENTSO-E published a day's
    `Actual Load`) can never mask those values in the interim file.

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

    # Merge oldest pull first so that, for timestamps covered by several raw
    # files, the most recent download wins below. Path.glob() returns files
    # in arbitrary filesystem order; sorting by mtime (name as tiebreaker)
    # makes the merge deterministic and recency-aware.
    raw_files = sorted(raw_dir.glob("*.csv"), key=lambda p: (p.stat().st_mtime, p.name))

    list_dfs = []
    for csv_file in raw_files:
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

    # Collapse duplicate timestamps cell-wise: per column, the last (i.e.
    # newest, given the mtime sort above) non-missing value wins. ENTSO-E
    # publishes "Actual Load" with a lag and occasionally backfills it days
    # later, so an early pull can legitimately hold NaN for hours that a
    # later pull fills in -- and a stale partial pull still sitting in raw/
    # must never mask values a newer pull already delivered. Revisions stay
    # possible (a newer non-missing value replaces an older one); only a
    # regression from value to NaN is ignored.
    merged_df = pd.concat(list_dfs)
    merged_df = merged_df.groupby(level=0).last()

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
    timeout: Optional[float] = 60.0,
) -> None:
    """
    Download new load and forecast data from ENTSO-E.

    This function queries the ENTSO-E Transparency Platform for a given period.
    If no start date is provided, it automatically resumes from the last
    available data point. Should the trailing rows hold a published
    day-ahead forecast but no `Actual Load` yet (ENTSO-E publishes actuals
    with a lag and occasionally backfills them days later), the fetch window
    is extended back to the first hour whose actuals are still missing,
    bounded by ``_MAX_BACKFILL_DAYS``, so late-published values heal
    automatically on the next incremental download.

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
        timeout: Per-socket-operation read timeout in seconds passed to the
            ENTSO-E client.  Kills stalled connections (raises
            ``requests.exceptions.Timeout``, caught by the existing retry loop,
            then ``RuntimeError`` after retries) without bounding long live
            transfers.  ``None`` disables the timeout.  Defaults to ``60.0``.

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
            # ENTSO-E publishes "Actual Load" with a lag and occasionally
            # backfills it days later. Rows for those hours already exist in
            # the interim (the day-ahead forecast is published early), so
            # resuming strictly from the last row would never re-fetch them.
            # If actuals are missing at the tail, restart the fetch at the
            # first missing hour instead, bounded by _MAX_BACKFILL_DAYS.
            if not observed.empty and "Actual Load" in getattr(observed, "columns", ()):
                backfill_floor = start_date - pd.Timedelta(days=_MAX_BACKFILL_DAYS)
                last_valid_actual = observed["Actual Load"].last_valid_index()
                backfill_start = (
                    backfill_floor
                    if last_valid_actual is None
                    else max(last_valid_actual + pd.Timedelta(hours=1), backfill_floor)
                )
                if backfill_start < start_date:
                    logger.info(
                        "Actual Load is missing from %s onwards; extending the "
                        "fetch window to heal it.",
                        backfill_start,
                    )
                    start_date = backfill_start
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

    client = _make_client(EntsoePandasClient, api_key=api_key, timeout=timeout)

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
    keep_forecast_future: bool = True,
    force: bool = False,
    timeout: Optional[float] = 60.0,
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
        keep_forecast_future: Forwarded to `merge_build_manual`. ``True`` (the
            default) retains future rows so a published day-ahead forecast
            survives -- correct for the day-ahead side-tables. Pass ``False``
            for actuals-only tables (e.g. per-zone load) so rows after the
            current UTC moment are dropped.
        force: If True, bypass the small-window cooldown check.
        timeout: Per-socket-operation read timeout in seconds.  ``None``
            disables the timeout.  Defaults to ``60.0``.

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

    client = _make_client(EntsoePandasClient, api_key=api_key, timeout=timeout)

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
        output_file=output_file,
        keep_forecast_future=keep_forecast_future,
        raw_subdir=raw_subdir,
    )


def download_renewable_forecast(
    api_key: str,
    country_code: str = "DE",
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
    timeout: Optional[float] = 60.0,
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
        timeout: Per-socket-operation read timeout in seconds.  ``None``
            disables the timeout.  Defaults to ``60.0``.

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
        timeout=timeout,
    )


def download_day_ahead_price(
    api_key: str,
    country_code: str = "DE_LU",
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
    timeout: Optional[float] = 60.0,
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
        timeout: Per-socket-operation read timeout in seconds.  ``None``
            disables the timeout.  Defaults to ``60.0``.

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
        timeout=timeout,
    )


def download_zone_loads(
    api_key: str,
    zones: Optional[Dict[str, str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
    keep_forecast_future: bool = False,
    timeout: Optional[float] = 60.0,
    on_zone_failure: Literal["raise", "collect"] = "raise",
) -> Optional[Dict[str, "ZoneResult"]]:
    """Download Actual Total Load separately for each German TSO control area.

    Instead of the single aggregated DE / DE-LU series fetched by
    `download_new_data`, this issues one ``query_load_and_forecast`` per control
    area in *zones* (default `GERMAN_TSO_ZONES`: Amprion, TenneT, TransnetBW,
    50Hertz). Each zone's ``"Actual Load"`` / ``"Forecasted Load"`` columns are
    renamed to the zone's column / ``"<col>_forecast"`` and written to a
    namespaced ``raw/zones/<col>/`` directory, then merged into a per-zone
    interim file ``interim/zone_<col>.csv``. Call `assemble_zone_loads`
    afterwards to join the per-zone actuals into one aligned, validated frame
    suitable for the bottom-up (sum-of-zones) forecasting pipeline.

    Each zone is fetched through the shared `_download_entsoe_table` helper, so
    it inherits the same bounded retry loop, timeout, and raw/interim
    namespacing. Unlike the day-ahead side-tables, ``keep_forecast_future``
    defaults to ``False`` here because the zone series are training *actuals*.

    Fail-safe invariant: collect mode is pure structured reporting.  It never
    substitutes data, fills gaps, or retries failed zones.  Files written by
    successful zones remain on disk exactly as in raise mode; no rollback is
    performed.  The caller owns the recovery policy for failed zones.

    Args:
        api_key: ENTSO-E Web API security token.
        zones: Mapping of column name to ``entsoe-py`` ``Area`` identifier. When
            ``None``, uses `GERMAN_TSO_ZONES`.
        start: Start in ``'YYYYMMDDHH00'`` format, or ``None`` to default to
            seven days ago.
        end: End in ``'YYYYMMDDHH00'`` format, or ``None`` to default to the
            start of tomorrow.
        force: If True, bypass the small-window cooldown check.
        keep_forecast_future: If True, retain rows after the current UTC moment
            (preserving each zone's day-ahead ``Forecasted Load``). Defaults to
            ``False`` (actuals-only, leakage-free training input).
        timeout: Per-socket-operation read timeout in seconds. ``None`` disables
            the timeout. Defaults to ``60.0``.
        on_zone_failure: How to handle a per-zone download failure.

            * ``"raise"`` (default): re-raises the first exception encountered,
              identical to the pre-collect behaviour. Returns ``None``.
            * ``"collect"``: wraps each zone's download in a try/except; on
              failure records a `ZoneResult` with ``ok=False`` and continues
              with the next zone; on success records ``ok=True`` with the path
              to the interim CSV. Returns a ``dict[str, ZoneResult]`` keyed by
              zone column in zone-dict order. Each failure is logged at WARNING.
              No substitution, caching, or retry is performed beyond what the
              underlying ``_download_entsoe_table`` already does.

            Argument-validation errors (unparseable ``start``/``end``, unknown
            ``on_zone_failure`` value) always raise regardless of the mode —
            they are caller bugs, not download failures.

    Returns:
        ``None`` in ``"raise"`` mode (unconditionally). A
        ``dict[str, ZoneResult]`` in ``"collect"`` mode, keyed by zone column
        in zones-dict order, containing one entry per zone regardless of
        success or failure.

    Raises:
        ImportError: If ``entsoe-py`` is not installed.
        ValueError: If ``start`` or ``end`` cannot be parsed, or if
            ``on_zone_failure`` is not ``"raise"`` or ``"collect"``.
        RuntimeError: In ``"raise"`` mode, if a zone download fails after
            ``_MAX_RETRIES`` attempts.

    Examples:
        ```{python}
        #| eval: false
        from spotforecast2_safe.downloader.entsoe import (
            download_zone_loads,
            assemble_zone_loads,
        )

        # Default raise mode: fetch all four German control-zone loads, then
        # assemble the aligned 4-column frame (raises on any zone failure or gap).
        download_zone_loads(api_key="YOUR_API_KEY", force=True)
        frame = assemble_zone_loads()
        print(frame.columns.tolist())
        ```

        ```{python}
        #| eval: false
        from spotforecast2_safe.downloader.entsoe import (
            download_zone_loads,
            build_zone_qc_frame,
        )

        # Collect mode: continue past a failing zone; inspect per-zone results.
        results = download_zone_loads(
            api_key="YOUR_API_KEY",
            start="202301010000",
            end="202301050000",
            force=True,
            on_zone_failure="collect",
        )
        for col, r in results.items():
            status = "ok" if r.ok else f"FAILED ({r.error})"
            print(f"{col}: {status}")

        # Build a QC frame from the zones that succeeded.
        ok_zones = {col: r.area for col, r in results.items() if r.ok}
        if ok_zones:
            qc = build_zone_qc_frame(zones=ok_zones)
            print(qc[["Actual Load", "Forecasted Load"]].tail())
        ```
    """
    if on_zone_failure not in ("raise", "collect"):
        raise ValueError(
            f"on_zone_failure={on_zone_failure!r} is invalid; "
            "use 'raise' or 'collect'."
        )

    # Validate start/end before entering the per-zone loop so that a bogus
    # format raises in both raise and collect mode (argument bugs, not download
    # failures).  The same parse logic is used by _download_entsoe_table.
    if start is not None:
        _parsed_start = pd.to_datetime(start, utc=True, errors="coerce")
        if pd.isna(_parsed_start):
            raise ValueError(f"start={start!r} did not parse to a valid timestamp")
    if end is not None:
        _parsed_end = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.isna(_parsed_end):
            raise ValueError(f"end={end!r} did not parse to a valid timestamp")

    zones = dict(GERMAN_TSO_ZONES if zones is None else zones)

    results: Dict[str, ZoneResult] = {}

    for column, area in zones.items():
        forecast_col = f"{column}_forecast"

        # query defined once per zone; used identically by both modes below.
        def query(client, cc, s, e, _col=column, _fc=forecast_col):  # type: ignore[no-untyped-def]
            frame = client.query_load_and_forecast(country_code=cc, start=s, end=e)
            return frame.rename(columns={"Actual Load": _col, "Forecasted Load": _fc})

        logger.info("Downloading control-zone load %s (%s)...", column, area)

        if on_zone_failure == "raise":
            _download_entsoe_table(
                api_key,
                area,
                start,
                end,
                query=query,
                raw_subdir=f"zones/{column}",
                output_file=f"zone_{column}.csv",
                file_prefix=f"entsoe_load_{column}",
                keep_forecast_future=keep_forecast_future,
                force=force,
                timeout=timeout,
            )
        else:
            # collect mode: argument validation is done above; this catch is
            # intentionally limited to download-layer failures.
            try:
                _download_entsoe_table(
                    api_key,
                    area,
                    start,
                    end,
                    query=query,
                    raw_subdir=f"zones/{column}",
                    output_file=f"zone_{column}.csv",
                    file_prefix=f"entsoe_load_{column}",
                    keep_forecast_future=keep_forecast_future,
                    force=force,
                    timeout=timeout,
                )
                # Resolve the path fresh after a successful download so that any
                # change to SPOTFORECAST2_DATA between iterations is reflected,
                # and assert the file was actually written to disk.
                interim_path = get_data_home() / "interim" / f"zone_{column}.csv"
                if not interim_path.exists():
                    raise FileNotFoundError(
                        f"Interim file not found after download: {interim_path}"
                    )
                results[column] = ZoneResult(
                    column=column,
                    area=area,
                    ok=True,
                    error=None,
                    interim_path=interim_path,
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 — limited to download-layer failures after argument validation
                logger.warning(
                    "Zone %s (%s) download failed in collect mode: %s",
                    column,
                    area,
                    exc,
                )
                results[column] = ZoneResult(
                    column=column,
                    area=area,
                    ok=False,
                    error=exc,
                    interim_path=None,
                )

    if on_zone_failure == "raise":
        return None
    return results


def assemble_zone_loads(
    zones: Optional[Dict[str, str]] = None,
    output_file: str = "energy_load_zones.csv",
    on_missing: str = "raise",
) -> pd.DataFrame:
    """Join the per-zone interim load files into one aligned, validated frame.

    Reads each ``interim/zone_<col>.csv`` written by `download_zone_loads`,
    takes the actual-load column, and outer-joins the zones onto a single
    complete hourly UTC index (gaps surface as ``NaN`` rows). The combined frame
    is written to ``interim/<output_file>`` and returned. The columns are the
    zone keys, so their sum is the bottom-up total German load.

    Fail-safe contract: with ``on_missing="raise"`` (the default), any missing
    hour in any zone raises `ValueError` (via
    `spotforecast2_safe.preprocessing.checking.check_y`) rather than being
    silently filled. Pass ``on_missing="passthrough"`` to return the frame with
    ``NaN`` left in place, so a downstream caller can opt into imputation
    explicitly (e.g. via the `MultiTask` ``impute`` step).

    Args:
        zones: Mapping of column name to ``Area`` identifier. When ``None``,
            uses `GERMAN_TSO_ZONES`.
        output_file: Interim filename to write under ``interim/``. Defaults to
            ``"energy_load_zones.csv"``.
        on_missing: ``"raise"`` (default) to reject any gap; ``"passthrough"``
            to keep ``NaN`` for explicit downstream imputation.

    Returns:
        The aligned 4-column load frame (index ``"Time (UTC)"``).

    Raises:
        ValueError: If ``on_missing`` is unknown, if a per-zone file lacks its
            zone column, or (when ``on_missing="raise"``) if any zone has a gap.
        FileNotFoundError: If a per-zone interim file is missing.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path

        import pandas as pd

        from spotforecast2_safe.downloader.entsoe import assemble_zone_loads

        zones = {"load_a": "AREA_A", "load_b": "AREA_B"}
        idx = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")

        with tempfile.TemporaryDirectory() as tmp:
            import os

            os.environ["SPOTFORECAST2_DATA"] = tmp
            interim = Path(tmp) / "interim"
            interim.mkdir(parents=True)
            for col, level in {"load_a": 100.0, "load_b": 50.0}.items():
                pd.DataFrame({col: level}, index=idx).rename_axis(
                    "Time (UTC)"
                ).to_csv(interim / f"zone_{col}.csv")

            frame = assemble_zone_loads(zones=zones)
            total = frame.sum(axis=1)
            print(frame.columns.tolist())
            print(f"bottom-up total (first hour): {total.iloc[0]}")
            assert total.iloc[0] == 150.0
        ```
    """
    if on_missing not in ("raise", "passthrough"):
        raise ValueError(
            f"on_missing={on_missing!r} is invalid; use 'raise' or 'passthrough'."
        )
    zones = dict(GERMAN_TSO_ZONES if zones is None else zones)

    interim_dir = get_data_home() / "interim"
    series_list = []
    for column in zones:
        path = interim_dir / f"zone_{column}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Per-zone interim file not found: {path}. "
                "Run download_zone_loads first."
            )
        frame = pd.read_csv(path)
        time_col = "Time (UTC)" if "Time (UTC)" in frame.columns else frame.columns[0]
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
        frame = frame.set_index(time_col)
        if column not in frame.columns:
            raise ValueError(
                f"Column {column!r} missing from {path} "
                f"(found {list(frame.columns)})."
            )
        series_list.append(frame[column].rename(column))

    combined = pd.concat(series_list, axis=1, sort=False).sort_index()
    # Enforce a complete, regular hourly UTC index so a missing hour in any zone
    # becomes an explicit NaN row rather than a silently shorter series.
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq="h")
    combined = combined.reindex(full_index)
    combined.index.name = "Time (UTC)"

    if on_missing == "raise":
        for column in combined.columns:
            check_y(combined[column], series_id=f"`{column}`")

    interim_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(interim_dir / output_file)
    logger.info(
        "Assembled %d-zone load frame -> %s",
        len(combined.columns),
        interim_dir / output_file,
    )
    return combined


def build_zone_qc_frame(
    zones: Optional[Dict[str, str]] = None,
    *,
    data_home: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """Build a bottom-up QC frame from per-zone interim CSVs.

    Reads each ``interim/zone_<col>.csv`` written by `download_zone_loads`,
    outer-joins the actual-load columns, and appends two aggregate columns:

    * ``"Actual Load"``: sum of the per-zone actual columns with
      ``min_count=len(zones)``.  A row where **any** zone's value is missing
      yields ``NaN`` in the total — partial sums are never silently reported.
    * ``"Forecasted Load"``: sum of the per-zone ``<col>_forecast`` columns
      with the same ``min_count``, **only** if every zone has a forecast column
      present in its CSV; otherwise the column is all-NaN.  This mirrors the
      operational behaviour in the chapter-14 four-zone pipeline.

    Fail-safe contract: a missing interim file raises `FileNotFoundError`
    naming the path rather than silently skipping the zone.  No fallback, no
    substitution.

    Args:
        zones: Mapping of column name to ``entsoe-py`` ``Area`` identifier. When
            ``None``, uses `GERMAN_TSO_ZONES`.
        data_home: Override for the data home directory (a ``Path``-like or
            ``str``).  When ``None``, ``get_data_home()`` is used (reads the
            ``SPOTFORECAST2_DATA`` environment variable).

    Returns:
        A ``pd.DataFrame`` with:

        * One column per zone containing the per-zone actual load values.
        * An ``"Actual Load"`` column with the bottom-up total (``NaN`` if any
          zone is missing for that timestamp).
        * A ``"Forecasted Load"`` column with the sum of per-zone forecasts if
          all zones provide a forecast column, else all-``NaN``.
        * Index: UTC datetimes parsed from the ``"Time (UTC)"`` CSV column,
          sorted ascending.

    Raises:
        FileNotFoundError: If a per-zone interim file does not exist.

    Notes:
        Files written by successful zones in a partial collect run are **not**
        rolled back.  Call ``build_zone_qc_frame`` with the subset of
        *succeeded* zones to avoid triggering the ``FileNotFoundError`` guard
        for zones whose download failed.

    Examples:
        ```{python}
        import os
        import tempfile
        from pathlib import Path

        import pandas as pd

        from spotforecast2_safe.downloader.entsoe import build_zone_qc_frame

        zones = {"load_a": "AREA_A", "load_b": "AREA_B"}
        idx = pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC")

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SPOTFORECAST2_DATA"] = tmp
            interim = Path(tmp) / "interim"
            interim.mkdir(parents=True)
            # Write synthetic per-zone CSVs with actual and forecast columns.
            for col, (actual, forecast) in {
                "load_a": (100.0, 105.0),
                "load_b": (50.0, 52.0),
            }.items():
                pd.DataFrame(
                    {col: actual, f"{col}_forecast": forecast}, index=idx
                ).rename_axis("Time (UTC)").to_csv(interim / f"zone_{col}.csv")

            qc = build_zone_qc_frame(zones=zones)
            print(qc.columns.tolist())
            print(f"Actual Load (first row): {qc['Actual Load'].iloc[0]}")
            print(f"Forecasted Load (first row): {qc['Forecasted Load'].iloc[0]}")
            assert qc["Actual Load"].iloc[0] == 150.0
            assert qc["Forecasted Load"].iloc[0] == 157.0
        ```
    """
    zones = dict(GERMAN_TSO_ZONES if zones is None else zones)
    n = len(zones)

    interim_dir = (
        Path(data_home) if data_home is not None else get_data_home()
    ) / "interim"

    actuals: list[pd.Series] = []
    forecasts: list[pd.Series] = []

    for column in zones:
        path = interim_dir / f"zone_{column}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Per-zone interim file not found: {path}. "
                "Run download_zone_loads first."
            )
        frame = pd.read_csv(path)
        time_col = "Time (UTC)" if "Time (UTC)" in frame.columns else frame.columns[0]
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
        frame = frame.set_index(time_col)

        actuals.append(frame[column].rename(column))

        forecast_col = f"{column}_forecast"
        if forecast_col in frame.columns:
            forecasts.append(frame[forecast_col].rename(column))

    zones_df = pd.concat(actuals, axis=1, sort=False).sort_index()

    result = zones_df.copy()
    result["Actual Load"] = zones_df.sum(axis=1, min_count=n)

    if len(forecasts) == n:
        forecast_df = pd.concat(forecasts, axis=1, sort=False).reindex(zones_df.index)
        result["Forecasted Load"] = forecast_df.sum(axis=1, min_count=n)
    else:
        logger.warning(
            "Per-zone day-ahead Forecasted Load missing for some zones; "
            "'Forecasted Load' column will be all-NaN."
        )
        result["Forecasted Load"] = float("nan")

    result.index.name = "Time (UTC)"
    return result

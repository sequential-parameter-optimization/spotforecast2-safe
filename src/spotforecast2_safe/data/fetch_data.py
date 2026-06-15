# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from os import environ
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from pandas import Timestamp

from spotforecast2_safe.calendar import create_holiday_df
from spotforecast2_safe.utils.convert_to_utc import convert_to_utc
from spotforecast2_safe.weather import WeatherService

OnMissing = Literal["raise", "ffill_bfill", "passthrough"]


def _apply_on_missing(
    y: pd.Series, on_missing: OnMissing, column: str, csv_path: Path
) -> pd.Series:
    """Enforce the fail-safe contract on a loaded series.

    If ``on_missing='raise'`` (default since the 1.0 major bump) and any
    NaN is present, raises ``ValueError`` listing the first few gap
    timestamps so the caller can act on them explicitly instead of
    inheriting imputed values disguised as measurements.

    Args:
        y: The series just read from the CSV.
        on_missing: Contract selector. ``'raise'`` (default) refuses to
            return a series that silently embeds imputed values;
            ``'ffill_bfill'`` opts into the legacy forward/back-fill
            behavior; ``'passthrough'`` returns the series as read, so
            an explicit downstream imputer (e.g.
            `spotforecast2_safe.preprocessing.LinearlyInterpolateTS`)
            can decide.
        column: Name of the column for the error message.
        csv_path: Source path for the error message.

    Returns:
        The same series if ``on_missing='raise'`` and no NaNs are
        present, the input unchanged for ``'passthrough'``, or an
        imputed copy for ``'ffill_bfill'``.

    Raises:
        ValueError: If NaNs are present and ``on_missing='raise'``, or
            if ``on_missing`` is not a recognized value.
    """
    if on_missing not in ("raise", "ffill_bfill", "passthrough"):
        raise ValueError(
            f"on_missing must be 'raise', 'ffill_bfill', or "
            f"'passthrough'; got {on_missing!r}."
        )
    if on_missing == "passthrough":
        return y
    if not y.isna().any():
        return y
    if on_missing == "raise":
        gaps = y.index[y.isna()]
        preview = ", ".join(str(ts) for ts in gaps[:5])
        more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
        raise ValueError(
            f"{len(gaps)} missing value(s) detected in column '{column}' "
            f"of {csv_path}. First gaps: [{preview}]{more}. "
            "Pass on_missing='ffill_bfill' to opt into legacy imputation "
            "or on_missing='passthrough' to return raw NaN."
        )
    return y.ffill().bfill()


def get_data_home(data_home: Optional[Union[str, Path]] = None) -> Path:
    """Return the location where datasets are to be stored.
    By default the data directory is set to a folder named 'spotforecast2_data' in the
    user home folder. Alternatively, it can be set by the 'SPOTFORECAST2_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Args:
        data_home (str or pathlib.Path, optional):
            The path to spotforecast data directory. If `None`, the default path
            is `~/spotforecast2_data`.

    Returns:
        data_home (pathlib.Path):
            The path to the spotforecast data directory.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        from spotforecast2_safe.data.fetch_data import get_data_home

        with tempfile.TemporaryDirectory() as tmp:
            p = get_data_home(Path(tmp) / "spotforecast2_data")
            print(p.exists())
            assert p.is_dir()
        ```
    """
    if data_home is None:
        data_home = environ.get(
            "SPOTFORECAST2_DATA", Path.home() / "spotforecast2_data"
        )
    # Ensure data_home is a Path() object pointing to an absolute path
    data_home = Path(data_home).expanduser().absolute()
    # Create data directory if it does not exists.
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def get_package_data_home() -> Path:
    """Return the location of the internal package datasets.

    Returns:
        pathlib.Path:
            The path to the spotforecast package data directory.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import get_package_data_home
        package_data_dir = get_package_data_home()
        print(package_data_dir.name)
        print(package_data_dir.parent.name)
        ```
    """
    return Path(__file__).parent.parent / "datasets" / "csv"


def get_cache_home(
    cache_home: Optional[Union[str, Path]] = None,
    create_dir: bool = True,
) -> Path:
    """Return the location where persistent models are to be cached.

    By default the cache directory is set to a folder named
    ``.spotforecast2_cache`` in the user home folder.  Alternatively, it
    can be set by the ``SPOTFORECAST2_CACHE`` environment variable or
    programmatically by giving an explicit folder path.  The ``~`` symbol
    is expanded to the user home folder.  When ``create_dir`` is ``True``
    (the default) the directory is created automatically if it does not
    already exist.

    This directory is used to store pickled trained models for quick
    reuse across forecasting runs, following scikit-learn model
    persistence conventions.

    Args:
        cache_home: Path to the spotforecast cache directory.  If
            ``None``, the value of the ``SPOTFORECAST2_CACHE`` environment
            variable is used when set, otherwise the default path
            ``~/.spotforecast2_cache`` is used.
        create_dir: Whether to create the cache directory if it does not
            exist.  When ``True`` (the default), the directory and any
            missing parent directories are created automatically.  When
            ``False``, the resolved path is returned without touching the
            filesystem.

    Returns:
        Absolute path to the spotforecast cache directory.

    Raises:
        OSError: If ``create_dir`` is ``True`` and the directory cannot
            be created due to a permissions error or other OS-level
            failure.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import get_cache_home
        cache_dir = get_cache_home()
        print(cache_dir.name)
        print(cache_dir.parent.name)
        ```

        ```{python}
        # Using custom path
        from spotforecast2_safe.data.fetch_data import get_cache_home
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            custom_cache = get_cache_home(Path(tmp) / 'my_cache')
            print(custom_cache.exists())
        ```

        ```{python}
        # Resolve path without creating the directory
        from spotforecast2_safe.data.fetch_data import get_cache_home
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            resolved = get_cache_home(Path(tmp) / 'not_yet', create_dir=False)
            print(resolved.exists())
        ```

        ```{python}
        # Using environment variable
        from spotforecast2_safe.data.fetch_data import get_cache_home
        import os
        os.environ['SPOTFORECAST2_CACHE'] = '/tmp/spotforecast2_cache_env'
        cache_dir = get_cache_home()
        cache_dir.as_posix()
        del os.environ['SPOTFORECAST2_CACHE']
        ```
    """
    if cache_home is None:
        cache_home = environ.get(
            "SPOTFORECAST2_CACHE", Path.home() / ".spotforecast2_cache"
        )
    # Ensure cache_home is a Path() object pointing to an absolute path
    cache_home = Path(cache_home).expanduser().absolute()
    if create_dir:
        # Create cache directory if it does not exist
        cache_home.mkdir(parents=True, exist_ok=True)
    return cache_home


def fetch_data(
    filename: Optional[Union[str, Path]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    columns: Optional[list] = None,
    index_col: int = 0,
    parse_dates: bool = True,
    dayfirst: bool = False,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetches a dataset from a CSV file or processes a DataFrame.

    Args:
        filename (str or Path, optional):
            Full absolute path of the CSV file containing the dataset
            (e.g., ``'/home/data/my_data.csv'``).  Required when
            dataframe is None.  Use ``get_data_home()`` or
            ``get_package_data_home()`` to build the path, for example
            ``fetch_data(filename=get_data_home() / "my_data.csv")``.

        dataframe (pd.DataFrame, optional):
            A pandas DataFrame to process. If provided, it will be processed with
            proper timezone handling. Mutually exclusive with filename.
        columns (list, optional):
            List of columns to be included in the dataset. If None, all columns are included.
            If an empty list is provided, a ValueError is raised. Default: None.
        index_col (int):
            Column index to be used as the index. Default: 0.
        parse_dates (bool):
            Whether to parse dates in the index column. Default: True.
        dayfirst (bool):
            Whether the day comes first in date parsing. Default: False.
        timezone (str):
            Timezone to set for the datetime index. If a DataFrame with naive index is provided,
            it will be localized to this timezone then converted to UTC. Default: "UTC".

    Returns:
        pd.DataFrame: The dataset with UTC timezone.

    Raises:
        ValueError: If columns is an empty list, if both filename and dataframe are provided,
            if neither filename nor dataframe is provided, or if filename is not an absolute path.
        FileNotFoundError: If CSV file does not exist.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
        # demo02.csv is included in the package datasets
        path_demo = get_package_data_home() / "demo02.csv"
        df = fetch_data(filename=path_demo)
        df.head()
        ```
    """
    if columns is not None and len(columns) == 0:
        raise ValueError("columns must be specified and cannot be empty.")

    if filename is not None and dataframe is not None:
        raise ValueError(
            "Cannot specify both filename and dataframe. Please provide only one."
        )

    if dataframe is not None:
        df = dataframe.copy()
        df = convert_to_utc(df, timezone)
        if columns is not None:
            df = df[columns].copy()
    else:
        if filename is None:
            raise ValueError(
                "filename must be specified when dataframe is None. "
                "Provide a full absolute path (e.g., get_data_home() / 'my_data.csv') "
                "or a DataFrame."
            )

        csv_path = Path(filename)
        if not csv_path.is_absolute():
            raise ValueError(
                f"filename must be an absolute path, got: '{filename}'. "
                "Use get_data_home() or get_package_data_home() to build the path:\n"
                "    fetch_data(filename=get_data_home() / 'my_data.csv')"
            )

        if not csv_path.is_file():
            raise FileNotFoundError(f"The file {csv_path} does not exist.")

        # Determine which columns to load for efficient reading
        usecols = None
        if columns is not None:
            if isinstance(index_col, int):
                header_df = pd.read_csv(csv_path, nrows=0)
                index_col_name = header_df.columns[index_col]
            else:
                index_col_name = index_col
            usecols = [index_col_name] + columns

        df = pd.read_csv(
            csv_path,
            index_col=index_col,
            parse_dates=parse_dates,
            dayfirst=dayfirst,
            usecols=usecols,
        )
        df = convert_to_utc(df, timezone)

    if df.index.freq is None:
        try:
            df.index.freq = pd.infer_freq(df.index)
        except (ValueError, TypeError):
            pass  # If the frequency cannot be inferred, leave df.index.freq as None.

    return df


def fetch_holiday_data(
    start: str | Timestamp,
    end: str | Timestamp,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Fetches holiday data for the dataset period.

    Args:
        start (str or pd.Timestamp):
            Start date of the dataset period.
        end (str or pd.Timestamp):
            End date of the dataset period.
        tz (str):
            Timezone for the holiday data.
        freq (str):
            Frequency of the holiday data.
        country_code (str):
            Country code for the holidays.
        state (str):
            State code for the holidays.

    Returns:
        pd.DataFrame: DataFrame containing holiday information.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import fetch_holiday_data
        holiday_df = fetch_holiday_data(
            start='2023-01-01T00:00',
            end='2023-01-10T00:00',
            tz='UTC',
            freq='h',
            country_code='DE',
            state='NW'
        )
        holiday_df.head()
        ```
    """
    holiday_df = create_holiday_df(
        start=start, end=end, tz=tz, freq=freq, country_code=country_code, state=state
    )
    return holiday_df


def fetch_weather_data(
    cov_start: str,
    cov_end: str,
    latitude: float = 51.5136,
    longitude: float = 7.4653,
    timezone: str = "UTC",
    freq: str = "h",
    fallback_on_failure: bool = True,
    cache_home: Optional[Union[str, Path]] = None,
    fill_missing: bool = False,
) -> pd.DataFrame:
    """Fetch weather data for the dataset period plus forecast horizon.

    Creates a weather DataFrame using the Open-Meteo API with optional
    caching.  Caching is controlled solely by the cache_home argument:
    when a path is provided the service reads from / writes to a parquet
    cache file inside that directory; when None (the default) no caching
    is performed.

    Args:
        cov_start: Start date for covariate data.
        cov_end: End date for covariate data.
        latitude: Latitude of the location for weather data.
            Default is 51.5136 (Dortmund).
        longitude: Longitude of the location for weather data.
            Default is 7.4653 (Dortmund).
        timezone: Timezone for the weather data.
        freq: Frequency of the weather data.
        fallback_on_failure: Whether to use fallback data in case of
            failure.
        cache_home: Optional path to cache directory.  When provided,
            fetched weather data is cached in
            ``<cache_home>/weather_cache.parquet``.  When None (default),
            no caching is performed.
        fill_missing: Whether to forward- and back-fill remaining NaN
            gaps (default False).  Forwarded to
            ``WeatherService.get_dataframe``; see its docstring.

    Returns:
        pd.DataFrame: DataFrame containing weather information.

    Examples:
        ```{python}
        #| eval: false
        # Requires a live HTTP call to the Open-Meteo API; cannot execute offline.
        import tempfile
        from pathlib import Path
        from spotforecast2_safe.data.fetch_data import fetch_weather_data

        with tempfile.TemporaryDirectory() as tmp:
            weather_df = fetch_weather_data(
                cov_start='2023-01-01T00:00',
                cov_end='2023-01-03T00:00',
                latitude=51.5136,
                longitude=7.4653,
                timezone='UTC',
                freq='h',
                fallback_on_failure=True,
                cache_home=Path(tmp) / 'weather_cache',
            )
            print(weather_df.shape)
        ```
    """
    if cache_home is not None:
        cache_path = get_cache_home(cache_home=cache_home) / "weather_cache.parquet"
    else:
        cache_path = None

    service = WeatherService(
        latitude=latitude, longitude=longitude, cache_path=cache_path
    )

    weather_df = service.get_dataframe(
        start=cov_start,
        end=cov_end,
        timezone=timezone,
        freq=freq,
        fallback_on_failure=fallback_on_failure,
        fill_missing=fill_missing,
    )
    return weather_df


def _load_energy_load_column(
    column: str,
    data_home: Optional[Union[str, Path]],
    on_missing: OnMissing,
) -> pd.Series:
    """Shared loader for ``interim/energy_load.csv`` columns.

    Returns the requested column as an hourly UTC-indexed series after
    applying the ``on_missing`` contract from `_apply_on_missing()`.
    Used by `load_timeseries()` and `load_timeseries_forecast()`,
    which differ only in the column they read.
    """
    data_dir = get_data_home(data_home)
    csv_path = data_dir / "interim" / "energy_load.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}. "
            "Run the downloader first or place energy_load.csv "
            "in the 'interim' sub-directory."
        )

    df = pd.read_csv(csv_path, parse_dates=["Time (UTC)"])
    df = df.set_index("Time (UTC)")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"
    df = df.asfreq("h")

    return _apply_on_missing(df[column], on_missing, column, csv_path)


def load_timeseries(
    data_home: Optional[Union[str, Path]] = None,
    on_missing: OnMissing = "raise",
) -> pd.Series:
    """Load the actual-load time series from ``interim/energy_load.csv``.

    Reads the ``Actual Load`` column and converts the index to a UTC
    ``DatetimeIndex`` with hourly frequency.  Missing values are
    **rejected** by default so callers cannot accidentally feed
    imputed values into downstream safety-critical pipelines.  Pass
    ``on_missing='ffill_bfill'`` to opt into the legacy
    forward/backward fill behavior that was the default before the
    1.0 major release.

    Args:
        data_home: Root data directory.  If None, resolved via
            ``get_data_home()``.
        on_missing: How to handle NaN rows in ``Actual Load``.
            ``'raise'`` (default) fails fast with the gap timestamps;
            ``'ffill_bfill'`` forward- then back-fills.

    Returns:
        pd.Series: Hourly actual-load series indexed by UTC timestamps.

    Raises:
        FileNotFoundError: If ``interim/energy_load.csv`` does not exist.
        ValueError: If ``on_missing='raise'`` and the series has NaNs.

    Examples:
        ```{python}
        import os
        import shutil
        import tempfile

        import pandas as pd

        from spotforecast2_safe.data.fetch_data import (
            get_package_data_home,
            load_timeseries,
        )

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        interim = os.path.join(tmp, "interim")
        os.makedirs(interim, exist_ok=True)

        demo = get_package_data_home() / "demo01.csv"
        df = pd.read_csv(demo).rename(
            columns={
                "Time": "Time (UTC)",
                "Actual": "Actual Load",
                "Forecast": "Forecasted Load",
            }
        )
        df.to_csv(os.path.join(interim, "energy_load.csv"), index=False)

        # demo01.csv has gaps on Jan 1 of each year, so we opt into the
        # legacy ffill/bfill behavior here.  Production callers should
        # leave on_missing='raise' (the default) and surface the gaps.
        y = load_timeseries(on_missing="ffill_bfill")
        print(isinstance(y, pd.Series), y.index.tz is not None)

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """
    return _load_energy_load_column("Actual Load", data_home, on_missing)


def load_timeseries_forecast(
    data_home: Optional[Union[str, Path]] = None,
    on_missing: OnMissing = "raise",
) -> pd.Series:
    """Load the day-ahead forecast time series from ``interim/energy_load.csv``.

    Reads the ``Forecasted Load`` column and converts the index to a
    UTC ``DatetimeIndex`` with hourly frequency.  Missing values are
    **rejected** by default so callers cannot accidentally feed
    imputed values into downstream safety-critical pipelines.  Pass
    ``on_missing='ffill_bfill'`` to opt into the legacy
    forward/backward fill behavior that was the default before the
    1.0 major release.

    Args:
        data_home: Root data directory.  If None, resolved via
            ``get_data_home()``.
        on_missing: How to handle NaN rows in ``Forecasted Load``.
            ``'raise'`` (default) fails fast with the gap timestamps;
            ``'ffill_bfill'`` forward- then back-fills.

    Returns:
        pd.Series: Hourly forecasted-load series indexed by UTC timestamps.

    Raises:
        FileNotFoundError: If ``interim/energy_load.csv`` does not exist.
        KeyError: If ``Forecasted Load`` column is not present.
        ValueError: If ``on_missing='raise'`` and the series has NaNs.

    Examples:
        ```{python}
        import os
        import shutil
        import tempfile

        import pandas as pd

        from spotforecast2_safe.data.fetch_data import (
            get_package_data_home,
            load_timeseries_forecast,
        )

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        interim = os.path.join(tmp, "interim")
        os.makedirs(interim, exist_ok=True)

        demo = get_package_data_home() / "demo01.csv"
        df = pd.read_csv(demo).rename(
            columns={
                "Time": "Time (UTC)",
                "Actual": "Actual Load",
                "Forecast": "Forecasted Load",
            }
        )
        df.to_csv(os.path.join(interim, "energy_load.csv"), index=False)

        # demo01.csv has gaps on Jan 1 of each year, so we opt into the
        # legacy ffill/bfill behavior here.  Production callers should
        # leave on_missing='raise' (the default) and surface the gaps.
        y_f = load_timeseries_forecast(on_missing="ffill_bfill")
        print(isinstance(y_f, pd.Series))

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """
    return _load_energy_load_column("Forecasted Load", data_home, on_missing)


def _read_interim_frame(
    filename: str,
    data_home: Optional[Union[str, Path]],
    index_col: str = "Time (UTC)",
) -> pd.DataFrame:
    """Read an hourly UTC-indexed interim CSV produced by the ENTSO-E downloader.

    Shared low-level reader for the day-ahead forecast side-tables
    (``renewable_forecast.csv``, ``day_ahead_price.csv``) that the extended
    ENTSO-E downloader writes next to ``energy_load.csv``. It resolves the
    file under ``<data_home>/interim/``, parses *index_col* as a UTC
    ``DatetimeIndex`` and resamples onto a regular hourly grid. NaN handling
    is deliberately left to the caller via `_apply_on_missing()` so the
    fail-safe contract stays in one place.

    Args:
        filename: Name of the CSV inside the ``interim`` sub-directory.
        data_home: Root data directory. If ``None``, resolved via
            `get_data_home()`.
        index_col: Name of the timestamp column. Defaults to ``"Time (UTC)"``.

    Returns:
        pd.DataFrame: The table indexed by an hourly UTC ``DatetimeIndex``
        named ``"datetime"``.

    Raises:
        FileNotFoundError: If the interim file does not exist.
    """
    data_dir = get_data_home(data_home)
    csv_path = data_dir / "interim" / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}. Run the matching ENTSO-E "
            "downloader function first (see "
            f"spotforecast2_safe.downloader.entsoe) to create {filename}."
        )

    df = pd.read_csv(csv_path, parse_dates=[index_col])
    df = df.set_index(index_col)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"
    return df.asfreq("h")


def load_renewable_forecast(
    data_home: Optional[Union[str, Path]] = None,
    on_missing: OnMissing = "raise",
) -> pd.DataFrame:
    """Load the ENTSO-E day-ahead wind/solar generation forecast.

    Reads ``interim/renewable_forecast.csv`` (written by
    `spotforecast2_safe.downloader.entsoe.download_renewable_forecast`) and
    returns every renewable generation-forecast column it contains (for
    Germany typically ``"Solar"``, ``"Wind Onshore"`` and ``"Wind
    Offshore"``) on a regular hourly UTC grid. Each column independently
    passes through the fail-safe `_apply_on_missing()` contract, so missing
    values are **rejected** by default rather than silently imputed.

    The day-ahead renewable forecast is a near-oracle, leakage-clean prior:
    it is published on D-1 and is therefore genuinely available at forecast
    time (CR-3). Use the day-ahead forecast, never the realised generation.

    Args:
        data_home: Root data directory. If ``None``, resolved via
            `get_data_home()`.
        on_missing: How to handle NaN rows. ``'raise'`` (default) fails fast
            with the gap timestamps; ``'ffill_bfill'`` forward/back-fills;
            ``'passthrough'`` returns the raw NaN so an explicit downstream
            provider can decide.

    Returns:
        pd.DataFrame: Hourly UTC-indexed day-ahead renewable forecast columns.

    Raises:
        FileNotFoundError: If ``interim/renewable_forecast.csv`` is absent.
        ValueError: If ``on_missing='raise'`` and any column has NaNs.

    Examples:
        ```{python}
        import os
        import shutil
        import tempfile

        import pandas as pd

        from spotforecast2_safe.data.fetch_data import load_renewable_forecast

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        interim = os.path.join(tmp, "interim")
        os.makedirs(interim, exist_ok=True)

        idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
        pd.DataFrame(
            {"Solar": 1.0, "Wind Onshore": 2.0}, index=idx
        ).rename_axis("Time (UTC)").to_csv(
            os.path.join(interim, "renewable_forecast.csv")
        )

        df = load_renewable_forecast()
        print(sorted(df.columns), len(df))

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """
    df = _read_interim_frame("renewable_forecast.csv", data_home)
    csv_path = get_data_home(data_home) / "interim" / "renewable_forecast.csv"
    for col in df.columns:
        df[col] = _apply_on_missing(df[col], on_missing, col, csv_path)
    return df


def load_day_ahead_price(
    data_home: Optional[Union[str, Path]] = None,
    on_missing: OnMissing = "raise",
    column: str = "Day-ahead Price",
) -> pd.Series:
    """Load the ENTSO-E day-ahead spot price (DE/LU) as an hourly series.

    Reads *column* from ``interim/day_ahead_price.csv`` (written by
    `spotforecast2_safe.downloader.entsoe.download_day_ahead_price`) and
    converts the index to a UTC ``DatetimeIndex`` with hourly frequency.
    Missing values are **rejected** by default (fail-safe). The day-ahead
    auction price is published on D-1 and is leakage-clean at forecast time
    as long as the day-ahead value (not the realised price) is used.

    Args:
        data_home: Root data directory. If ``None``, resolved via
            `get_data_home()`.
        on_missing: How to handle NaN rows. ``'raise'`` (default) fails fast;
            ``'ffill_bfill'`` forward/back-fills; ``'passthrough'`` returns
            raw NaN.
        column: Name of the price column to read. Defaults to
            ``"Day-ahead Price"``.

    Returns:
        pd.Series: Hourly UTC-indexed day-ahead price series.

    Raises:
        FileNotFoundError: If ``interim/day_ahead_price.csv`` is absent.
        KeyError: If *column* is not present in the file.
        ValueError: If ``on_missing='raise'`` and the series has NaNs.

    Examples:
        ```{python}
        import os
        import shutil
        import tempfile

        import pandas as pd

        from spotforecast2_safe.data.fetch_data import load_day_ahead_price

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        interim = os.path.join(tmp, "interim")
        os.makedirs(interim, exist_ok=True)

        idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
        pd.DataFrame(
            {"Day-ahead Price": 95.0}, index=idx
        ).rename_axis("Time (UTC)").to_csv(
            os.path.join(interim, "day_ahead_price.csv")
        )

        s = load_day_ahead_price()
        print(isinstance(s, pd.Series), len(s))

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """
    df = _read_interim_frame("day_ahead_price.csv", data_home)
    csv_path = get_data_home(data_home) / "interim" / "day_ahead_price.csv"
    if column not in df.columns:
        raise KeyError(
            f"Column {column!r} not present in {csv_path}. "
            f"Available columns: {list(df.columns)}."
        )
    return _apply_on_missing(df[column], on_missing, column, csv_path)


def load_interim(
    mode: str = "combined",
    *,
    data_home: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Read the ENTSO-E interim CSV(s) written by the downloader.

    This is the library-level counterpart of the inline ``load_interim()``
    helpers that every submission script defines identically.  The two modes
    reproduce the exact CSV-reading logic from those scripts so callers can
    replace their local copies with a single import.

    Args:
        mode: Which interim layout to read.

            ``"combined"`` (default) — reads ``<data_home>/interim/energy_load.csv``
            (written by ``download_new_data``).  This is the single DE-total
            series path used by chronos, optuna, and spotoptim submission
            scripts.

            ``"four_zone"`` — assembles the per-zone QC frame via
            ``spotforecast2_safe.downloader.entsoe.build_zone_qc_frame`` and
            also writes the zones-only modelling CSV
            (``interim/energy_load_zones.csv``).  This is the normal four-zone
            path used by team4_4zones_submit.

        data_home: Root data directory.  If ``None``, resolved via
            ``get_data_home()``.

    Returns:
        pd.DataFrame: A UTC-indexed interim frame ready for ``assert_submission_coverage``,
        ``select_key_lags``, and ``entsoe_predictions``.  In ``"combined"`` mode the
        frame contains at least ``Actual Load`` and ``Forecasted Load``.  In
        ``"four_zone"`` mode the frame additionally contains the per-zone columns
        (``load_amprion``, ``load_tennet``, ``load_transnetbw``, ``load_50hertz``).

    Raises:
        FileNotFoundError: When the required interim file(s) are missing.
        ValueError: When ``mode`` is not ``"combined"`` or ``"four_zone"``.

    Examples:
        ```{python}
        import os, shutil, tempfile
        import pandas as pd
        from spotforecast2_safe.data.fetch_data import get_data_home, load_interim

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        interim_dir = os.path.join(tmp, "interim")
        os.makedirs(interim_dir, exist_ok=True)

        idx = pd.date_range("2026-06-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"Actual Load": 40000.0, "Forecasted Load": 41000.0}, index=idx
        )
        df.index.name = "Time (UTC)"
        df.to_csv(os.path.join(interim_dir, "energy_load.csv"))

        frame = load_interim(mode="combined")
        print(frame.index.tz, len(frame))

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """
    import logging

    _logger = logging.getLogger(__name__)

    if mode not in ("combined", "four_zone"):
        raise ValueError(
            f"mode must be 'combined' or 'four_zone', got {mode!r}."
        )

    interim_dir = get_data_home(data_home) / "interim"

    if mode == "combined":
        combined_csv = interim_dir / "energy_load.csv"
        if not combined_csv.exists():
            raise FileNotFoundError(
                f"No interim cache at {combined_csv}; run the downloader first "
                "(download_new_data) or drop --skip-download."
            )
        interim = pd.read_csv(combined_csv, index_col=0, parse_dates=True)
        interim.index = pd.to_datetime(interim.index, utc=True)
        _logger.info(
            "interim (combined): %s  (%d rows, %s -> %s)",
            combined_csv,
            len(interim),
            interim.index.min(),
            interim.index.max(),
        )
        return interim

    # --- four_zone mode ---
    from pathlib import Path as _Path

    from spotforecast2_safe.downloader.entsoe import (
        ZONE_MODEL_CSV,
        build_zone_qc_frame,
    )
    from spotforecast2_safe.downloader.resilience import ZONE_COLUMNS

    try:
        interim = build_zone_qc_frame()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No per-zone interim cache found; cannot proceed without a "
            f"successful four-zone download. Details: {exc}"
        ) from exc

    # Write the zones-only modelling CSV (build_zone_qc_frame does not do this).
    # Contains only per-zone Actual Load columns so the bottom-up total cannot
    # leak in as a feature.
    zone_cols = [c for c in interim.columns if c in ZONE_COLUMNS]
    model_csv = interim_dir / _Path(ZONE_MODEL_CSV).name
    interim[zone_cols].rename_axis("Time (UTC)").to_csv(model_csv)

    if (
        "Forecasted Load" not in interim.columns
        or interim["Forecasted Load"].isna().all()
    ):
        _logger.warning(
            "per-zone day-ahead Forecasted Load missing for some zones; "
            "the aggregate deviation QC and the ENTSO-E baseline degrade."
        )

    _logger.info(
        "interim (four_zone): %s  (%d rows, %s -> %s; zones=%s)",
        model_csv,
        len(interim),
        interim.index.min(),
        interim.index.max(),
        ZONE_COLUMNS,
    )
    return interim


def load_school_holidays_de() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Load the bundled German school-holiday interval table.

    Reads ``datasets/csv/school_holidays_de.csv`` (ODbL-1.0) from the package
    data directory and the companion validity-range metadata from
    ``school_holidays_de_meta.csv``.  No download is needed; both files ship
    with the package.

    The CSV has four columns: ``state`` (ISO 3166-2 subdivision short code,
    e.g. ``"NW"``), ``name`` (German name of the holiday period), ``start_date``
    and ``end_date`` (both inclusive, parsed as datetime64 (resolution depends
    on the pandas version)).  Coverage is 2022-01-01 to 2027-12-31 (all 16
    German Bundesländer).

    Data provenance: OpenHolidays API (https://openholidaysapi.org), database
    https://github.com/openpotato/openholidaysapi.data, ODC Open Database
    License (ODbL-1.0).

    Regeneration command (requires network access):

    ```text
    for code in BW BY BE BB HB HH HE MV NI NW RP SL SN ST SH TH:
        GET https://openholidaysapi.org/SchoolHolidays?countryIsoCode=DE
            &subdivisionCode=DE-<code>&validFrom=2022-01-01&validTo=2024-12-31
            &languageIsoCode=DE  (split into two 3-year windows to respect the
            1095-day API limit; second window: validFrom=2025-01-01,
            validTo=2027-12-31).  Keep every record whose startDate falls within
            [valid_from, valid_to]; endDate may extend beyond valid_to and is
            kept verbatim (queries past valid_to raise).
    ```

    Returns:
        tuple: A three-tuple ``(df, valid_from, valid_to)`` where:

        - **df** — DataFrame with columns ``state``, ``name``,
          ``start_date`` (datetime64, resolution depends on the pandas version),
          ``end_date`` (datetime64, resolution depends on the pandas version),
          sorted by ``(state, start_date)``.
        - **valid_from** — `pd.Timestamp` for the first covered day
          (``2022-01-01``).
        - **valid_to** — `pd.Timestamp` for the last covered day
          (``2027-12-31``).

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import load_school_holidays_de

        df, valid_from, valid_to = load_school_holidays_de()
        print("states:", sorted(df["state"].unique()))
        print("valid_from:", valid_from.date())
        print("valid_to:", valid_to.date())
        print("shape:", df.shape)
        assert len(df["state"].unique()) == 16
        assert valid_from == pd.Timestamp("2022-01-01")
        assert valid_to == pd.Timestamp("2027-12-31")
        ```
    """
    pkg_dir = get_package_data_home()
    csv_path = pkg_dir / "school_holidays_de.csv"
    meta_path = pkg_dir / "school_holidays_de_meta.csv"

    df = pd.read_csv(
        csv_path,
        parse_dates=["start_date", "end_date"],
    )
    df = df.sort_values(["state", "start_date"]).reset_index(drop=True)

    meta = pd.read_csv(meta_path, parse_dates=["valid_from", "valid_to"])
    valid_from = pd.Timestamp(meta.at[0, "valid_from"])
    valid_to = pd.Timestamp(meta.at[0, "valid_to"])

    return df, valid_from, valid_to

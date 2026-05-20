# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utility functions for timezone conversion."""

from typing import Optional, Union

import pandas as pd


def to_utc_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """Coerce a string or Timestamp to a UTC-aware `pandas.Timestamp`.

    Strings are parsed with ``utc=True``; existing Timestamps are returned
    unchanged.  This deduplicates the same three-line pattern repeated
    across public feature builders in this package.

    Args:
        value: A date/time string or an existing `pandas.Timestamp`.

    Returns:
        A UTC-aware `pandas.Timestamp`.

    Examples:
        ```{python}
        import pandas as pd

        from spotforecast2_safe.utils.convert_to_utc import to_utc_timestamp

        # Date-only string → UTC-aware Timestamp
        result = to_utc_timestamp("2024-01-01")
        print(result)
        assert result.tz is not None
        assert str(result.tz) == "UTC"
        assert result == pd.Timestamp("2024-01-01", tz="UTC")

        # Datetime string with offset → UTC-aware Timestamp
        result2 = to_utc_timestamp("2024-06-15 12:00:00+02:00")
        print(result2)
        assert str(result2.tz) == "UTC"
        assert result2 == pd.Timestamp("2024-06-15 10:00:00", tz="UTC")

        # Pre-existing UTC Timestamp → returned unchanged (identity)
        ts = pd.Timestamp("2024-06-15", tz="UTC")
        assert to_utc_timestamp(ts) is ts
        print("pass-through identity: ok")
        ```
    """
    if isinstance(value, str):
        return pd.to_datetime(value, utc=True)
    return value


def convert_to_utc(df: pd.DataFrame, timezone: Optional[str]) -> pd.DataFrame:
    """Convert DataFrame index timezone to UTC.

    Args:
        df: DataFrame with DatetimeIndex.
        timezone: Optional timezone string. Required if index has no timezone.

    Returns:
        DataFrame with UTC timezone index.

    Raises:
        ValueError: If index is not DatetimeIndex or has no timezone and
            timezone is None.

    Examples:
        ```{python}
        import pandas as pd

        from spotforecast2_safe.utils.convert_to_utc import convert_to_utc

        # Scenario 1: tz-naive index → localise to Europe/Berlin then convert to UTC
        df_naive = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
        )
        result_naive = convert_to_utc(df_naive, "Europe/Berlin")
        print("tz-naive result:")
        print(result_naive)
        assert str(result_naive.index.tz) == "UTC"
        # 2022-01-01 CET = UTC+1, so midnight Berlin = 23:00 UTC previous day
        assert result_naive.index[0] == pd.Timestamp("2021-12-31 23:00:00", tz="UTC")
        ```

        ```{python}
        import pandas as pd

        from spotforecast2_safe.utils.convert_to_utc import convert_to_utc

        # Scenario 2: non-UTC tz-aware index (US/Eastern) → converted to UTC
        idx = pd.date_range("2022-06-01", periods=3, freq="h", tz="US/Eastern")
        df_eastern = pd.DataFrame({"value": [10, 20, 30]}, index=idx)
        result_eastern = convert_to_utc(df_eastern, timezone=None)
        print("US/Eastern → UTC result:")
        print(result_eastern)
        assert str(result_eastern.index.tz) == "UTC"
        # US/Eastern is UTC-4 in June (EDT), so 00:00 Eastern = 04:00 UTC
        assert result_eastern.index[0] == pd.Timestamp("2022-06-01 04:00:00", tz="UTC")
        ```
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "No DatetimeIndex found. Please specify the time column via 'index_col'"
        )
    if df.index.tz is None:
        if timezone is not None:
            df.index = df.index.tz_localize(timezone)
        else:
            raise ValueError(
                "Index has no timezone information. Please provide a timezone."
            )

    df.index = df.index.tz_convert("UTC")

    return df

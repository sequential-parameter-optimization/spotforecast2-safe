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
        >>> from spotforecast2_safe.utils.convert_to_utc import to_utc_timestamp
        >>> to_utc_timestamp("2024-01-01")
        Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
        >>> import pandas as pd
        >>> ts = pd.Timestamp("2024-06-15", tz="UTC")
        >>> to_utc_timestamp(ts) is ts
        True
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
        >>> from spotforecast2_safe.utils.convert_to_utc import convert_to_utc
        >>> df = pd.DataFrame({"value": [1, 2, 3]}, index=pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]))
        >>> convert_to_utc(df, "Europe/Berlin")
                   value
        2022-01-01 00:00:00+01:00
        2022-01-02 00:00:00+01:00
        2022-01-03 00:00:00+01:00
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

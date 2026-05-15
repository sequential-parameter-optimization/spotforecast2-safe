# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared helpers for the :mod:`spotforecast2_safe.calendar` package."""

from typing import Union

import pandas as pd


def to_utc_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """Coerce a string or Timestamp to a UTC-aware :class:`pandas.Timestamp`.

    Strings are parsed with ``utc=True``; existing Timestamps are returned
    unchanged.  This dedupes the same three-line pattern previously
    repeated across every public feature builder in this package.
    """
    if isinstance(value, str):
        return pd.to_datetime(value, utc=True)
    return value

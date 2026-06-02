# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``utils.convert_to_utc``.

Covers the two fail-safe ValueError guards (non-DatetimeIndex, and a tz-naive
index with no timezone supplied) plus the localise/convert happy paths.
"""

import pandas as pd
import pytest

from spotforecast2_safe.utils.convert_to_utc import convert_to_utc, to_utc_timestamp


class TestConvertToUtc:
    def test_non_datetimeindex_raises(self):
        df = pd.DataFrame({"v": [1, 2, 3]})  # default RangeIndex
        with pytest.raises(ValueError, match="No DatetimeIndex"):
            convert_to_utc(df, timezone="UTC")

    def test_tznaive_without_timezone_raises(self):
        df = pd.DataFrame({"v": [1]}, index=pd.to_datetime(["2022-01-01"]))
        with pytest.raises(ValueError, match="no timezone"):
            convert_to_utc(df, timezone=None)

    def test_tznaive_localized_then_converted(self):
        df = pd.DataFrame({"v": [1]}, index=pd.to_datetime(["2022-01-01"]))
        out = convert_to_utc(df, timezone="Europe/Berlin")
        assert str(out.index.tz) == "UTC"
        assert out.index[0] == pd.Timestamp("2021-12-31 23:00:00", tz="UTC")

    def test_tzaware_converted_to_utc(self):
        idx = pd.date_range("2022-06-01", periods=3, freq="h", tz="US/Eastern")
        out = convert_to_utc(pd.DataFrame({"v": [1, 2, 3]}, index=idx), timezone=None)
        assert str(out.index.tz) == "UTC"


class TestToUtcTimestamp:
    def test_string_parsed_to_utc(self):
        assert str(to_utc_timestamp("2024-01-01").tz) == "UTC"

    def test_existing_timestamp_passthrough(self):
        ts = pd.Timestamp("2024-06-15", tz="UTC")
        assert to_utc_timestamp(ts) is ts

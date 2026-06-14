# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the pure (no-network) utility helpers in the ENTSO-E downloader."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.downloader.entsoe import (
    derive_download_window,
    extract_entsoe_hourly_forecast,
)

# ---------------------------------------------------------------------------
# derive_download_window
# ---------------------------------------------------------------------------


def test_derive_download_window_default():
    """Default window with train_years=3 from 2026-06-14.

    Formula: today - Timedelta(days=365*3+45) = 2026-06-14 - 1140 d = 2023-05-01.
    End: today + 2 days = 2026-06-16.
    """
    now = pd.Timestamp("2026-06-14", tz="UTC")
    start, end = derive_download_window(now, train_years=3)
    assert start == "202305010000"
    assert end == "202606160000"


def test_derive_download_window_explicit_data_end():
    """Explicit data_end passes through unchanged; start is still computed."""
    now = pd.Timestamp("2026-06-14", tz="UTC")
    start, end = derive_download_window(now, train_years=3, data_end="202612312300")
    assert start == "202305010000"
    assert end == "202612312300"


def test_derive_download_window_malformed_data_end_raises():
    """A data_end that lacks minutes raises ValueError mentioning YYYYMMDDHHMM."""
    now = pd.Timestamp("2026-06-14", tz="UTC")
    with pytest.raises(ValueError, match="YYYYMMDDHHMM"):
        derive_download_window(now, train_years=3, data_end="2026-12-31")


def test_derive_download_window_train_years_zero_raises():
    now = pd.Timestamp("2026-06-14", tz="UTC")
    with pytest.raises(ValueError):
        derive_download_window(now, train_years=0)


def test_derive_download_window_train_years_negative_raises():
    now = pd.Timestamp("2026-06-14", tz="UTC")
    with pytest.raises(ValueError):
        derive_download_window(now, train_years=-1)


def test_derive_download_window_train_years_bool_raises():
    """bool is rejected even though bool is a subclass of int."""
    now = pd.Timestamp("2026-06-14", tz="UTC")
    with pytest.raises(ValueError):
        derive_download_window(now, train_years=True)


def test_derive_download_window_negative_start_margin_raises():
    now = pd.Timestamp("2026-06-14", tz="UTC")
    with pytest.raises(ValueError):
        derive_download_window(now, train_years=1, start_margin_days=-1)


def test_derive_download_window_now_str_raises_type_error():
    """Passing a string for now raises TypeError (no silent coercion)."""
    with pytest.raises(TypeError):
        derive_download_window("2026-06-14", train_years=3)  # type: ignore[arg-type]


def test_derive_download_window_now_tz_naive_raises_type_error():
    """A tz-naive Timestamp raises TypeError (window must be anchored to UTC)."""
    naive = pd.Timestamp("2026-06-14")
    assert naive.tzinfo is None
    with pytest.raises(TypeError, match="timezone-aware"):
        derive_download_window(naive, train_years=3)


def test_derive_download_window_start_differs_by_365_days_for_different_train_years():
    """start_dl for train_years=3 vs train_years=2 must differ by exactly 365 days."""
    now = pd.Timestamp("2026-06-14", tz="UTC")
    start3, _ = derive_download_window(now, train_years=3, start_margin_days=0)
    start2, _ = derive_download_window(now, train_years=2, start_margin_days=0)
    t3 = pd.to_datetime(start3, format="%Y%m%d%H%M", utc=True)
    t2 = pd.to_datetime(start2, format="%Y%m%d%H%M", utc=True)
    assert (t2 - t3).days == 365


def test_derive_download_window_start_margin_bool_raises():
    """bool is rejected for start_margin_days."""
    now = pd.Timestamp("2026-06-14", tz="UTC")
    with pytest.raises(ValueError):
        derive_download_window(now, train_years=1, start_margin_days=False)


# ---------------------------------------------------------------------------
# extract_entsoe_hourly_forecast
# ---------------------------------------------------------------------------


def _make_15min_frame(column: str = "Forecasted Load") -> pd.DataFrame:
    """Build a tiny 15-minute DataFrame for testing the resampler."""
    idx = pd.date_range("2026-06-14 00:00", periods=8, freq="15min", tz="UTC")
    values = [100.0, 110.0, 90.0, 120.0, 80.0, 95.0, 105.0, 115.0]
    return pd.DataFrame({column: values}, index=idx)


def test_extract_hourly_forecast_basic():
    """15-min frame with 'Forecasted Load' -> hourly mean series."""
    df = _make_15min_frame()
    out = extract_entsoe_hourly_forecast(df)
    assert isinstance(out, pd.Series)
    assert out.name == "Forecasted Load"
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.freq is not None or len(out) == 2  # two hours
    assert out.index.is_monotonic_increasing
    assert len(out) == 2
    # hour 00: mean(100, 110, 90, 120) = 105.0
    assert out.iloc[0] == pytest.approx(105.0)
    # hour 01: mean(80, 95, 105, 115) = 98.75
    assert out.iloc[1] == pytest.approx(98.75)


def test_extract_hourly_forecast_name_preserved():
    df = _make_15min_frame()
    out = extract_entsoe_hourly_forecast(df)
    assert out.name == "Forecasted Load"


def test_extract_hourly_forecast_index_is_hourly_and_monotonic():
    df = _make_15min_frame()
    out = extract_entsoe_hourly_forecast(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    diffs = out.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta("1h")).all()


def test_extract_hourly_forecast_missing_column_raises_key_error():
    df = _make_15min_frame()
    with pytest.raises(KeyError):
        extract_entsoe_hourly_forecast(df, column="Nonexistent")


def test_extract_hourly_forecast_non_dataframe_raises_type_error():
    idx = pd.date_range("2026-06-14", periods=4, freq="15min", tz="UTC")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    with pytest.raises(TypeError):
        extract_entsoe_hourly_forecast(series)  # type: ignore[arg-type]


def test_extract_hourly_forecast_range_index_raises_type_error():
    """A DataFrame with a plain RangeIndex (not DatetimeIndex) must raise TypeError."""
    df = pd.DataFrame({"Forecasted Load": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        extract_entsoe_hourly_forecast(df)


def test_extract_hourly_forecast_custom_column():
    """Custom column name 'Actual Load' is respected."""
    df = _make_15min_frame(column="Actual Load")
    out = extract_entsoe_hourly_forecast(df, column="Actual Load")
    assert out.name == "Actual Load"
    assert len(out) == 2


def test_extract_hourly_forecast_all_nan_raises_value_error():
    """A column that is entirely NaN raises ValueError."""
    idx = pd.date_range("2026-06-14 00:00", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({"Forecasted Load": [np.nan, np.nan, np.nan, np.nan]}, index=idx)
    with pytest.raises(ValueError, match="no non-NaN hourly observations"):
        extract_entsoe_hourly_forecast(df)

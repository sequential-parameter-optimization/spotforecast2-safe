# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the day-type calendar refinement (is_workday + day_type)."""

import pandas as pd
import pytest

from spotforecast2_safe.calendar import create_day_type_df, get_day_type_features
from spotforecast2_safe.calendar.holiday import (
    DAY_TYPE_HOLIDAY,
    DAY_TYPE_SATURDAY,
    DAY_TYPE_SUNDAY,
    DAY_TYPE_WORKDAY,
)


class TestCreateDayTypeDf:
    def test_classes_over_a_week(self):
        # 2024-01-01 Mon = New Year (holiday); 02–05 workdays; 06 Sat; 07 Sun.
        df = create_day_type_df("2024-01-01", "2024-01-07", freq="D")
        assert df.loc["2024-01-01", "day_type"] == DAY_TYPE_HOLIDAY
        assert df.loc["2024-01-02", "day_type"] == DAY_TYPE_WORKDAY
        assert df.loc["2024-01-06", "day_type"] == DAY_TYPE_SATURDAY
        assert df.loc["2024-01-07", "day_type"] == DAY_TYPE_SUNDAY

    def test_is_workday_matches_day_type(self):
        df = create_day_type_df("2024-01-01", "2024-01-31", freq="D")
        assert ((df["day_type"] == DAY_TYPE_WORKDAY) == (df["is_workday"] == 1)).all()

    def test_holiday_on_weekend_is_holiday_class(self):
        # 2025-01-01 is a Wednesday; pick a holiday landing on a weekend:
        # 2022-01-01 (New Year) fell on a Saturday → still classed as holiday.
        df = create_day_type_df("2022-01-01", "2022-01-01", freq="D")
        assert df.loc["2022-01-01", "day_type"] == DAY_TYPE_HOLIDAY
        assert df.loc["2022-01-01", "is_workday"] == 0

    def test_hourly_broadcast(self):
        df = create_day_type_df("2024-01-02", "2024-01-02 23:00", freq="h")
        assert df.shape == (24, 2)
        # Whole workday: every hour identical.
        assert (df["is_workday"] == 1).all()
        assert (df["day_type"] == DAY_TYPE_WORKDAY).all()

    def test_no_nans(self):
        df = create_day_type_df("2024-01-01", "2024-12-31", freq="D")
        assert not df.isna().any().any()

    def test_deterministic(self):
        a = create_day_type_df("2024-05-01", "2024-05-31", freq="h")
        b = create_day_type_df("2024-05-01", "2024-05-31", freq="h")
        pd.testing.assert_frame_equal(a, b)


class TestGetDayTypeFeatures:
    def test_shape_and_alignment(self):
        forecast_horizon = 24
        n_data = 48
        data = pd.DataFrame(
            {"load": range(n_data)},
            index=pd.date_range("2024-01-01", periods=n_data, freq="h", tz="UTC"),
        )
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))
        feats = get_day_type_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
        )
        assert feats.shape == (n_data + forecast_horizon, 2)
        assert feats.columns.tolist() == ["is_workday", "day_type"]
        # 2024-01-01 is New Year (holiday).
        assert feats.loc["2024-01-01 00:00:00+00:00", "day_type"] == DAY_TYPE_HOLIDAY
        assert feats.loc["2024-01-01 00:00:00+00:00", "is_workday"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

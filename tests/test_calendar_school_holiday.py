# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for per-Bundesland school-holiday calendar features."""

import pandas as pd
import pytest

from spotforecast2_safe.calendar import (
    create_school_holiday_df,
    get_school_holiday_features,
)
from spotforecast2_safe.data.fetch_data import load_school_holidays_de
from spotforecast2_safe.manager.features import select_exogenous_features


class TestCreateSchoolHolidayDf:
    """Unit tests for create_school_holiday_df."""

    def test_determinism(self):
        a = create_school_holiday_df("2024-07-01", "2024-07-31", freq="h", state="NW")
        b = create_school_holiday_df("2024-07-01", "2024-07-31", freq="h", state="NW")
        pd.testing.assert_frame_equal(a, b)

    def test_dtype_no_nan_values_binary(self):
        df = create_school_holiday_df("2024-01-01", "2024-12-31", freq="D", state="NW")
        assert df["is_school_holiday"].dtype == int
        assert not df["is_school_holiday"].isna().any()
        assert set(df["is_school_holiday"].unique()).issubset({0, 1})

    def test_nw_osterferien_2024(self):
        # NW Osterferien 2024: 2024-03-25 → 2024-04-06 (inclusive).
        df = create_school_holiday_df("2024-03-24", "2024-04-07", freq="D", state="NW")
        assert df.loc["2024-03-24", "is_school_holiday"] == 0
        assert df.loc["2024-03-25", "is_school_holiday"] == 1
        assert df.loc["2024-04-06", "is_school_holiday"] == 1
        assert df.loc["2024-04-07", "is_school_holiday"] == 0

    def test_nw_sommerferien_2024(self):
        # NW Sommerferien 2024: 2024-07-08 → 2024-08-20 (inclusive).
        df = create_school_holiday_df("2024-07-06", "2024-08-22", freq="D", state="NW")
        assert df.loc["2024-07-07", "is_school_holiday"] == 0
        assert df.loc["2024-07-08", "is_school_holiday"] == 1
        assert df.loc["2024-08-20", "is_school_holiday"] == 1
        assert df.loc["2024-08-21", "is_school_holiday"] == 0

    def test_nw_herbstferien_2024(self):
        # NW Herbstferien 2024: 2024-10-14 → 2024-10-26 (inclusive).
        df = create_school_holiday_df("2024-10-13", "2024-10-27", freq="D", state="NW")
        assert df.loc["2024-10-13", "is_school_holiday"] == 0
        assert df.loc["2024-10-14", "is_school_holiday"] == 1
        assert df.loc["2024-10-26", "is_school_holiday"] == 1
        assert df.loc["2024-10-27", "is_school_holiday"] == 0

    def test_state_isolation_by_vs_nw(self):
        # BY Sommerferien 2024: 2024-07-29 → 2024-09-09.
        # NW Sommerferien 2024: 2024-07-08 → 2024-08-20.
        # 2024-08-21 is 0 for NW (after NW Sommerferien) but 1 for BY (still in BY Sommerferien).
        df_nw = create_school_holiday_df(
            "2024-08-21", "2024-08-21", freq="D", state="NW"
        )
        df_by = create_school_holiday_df(
            "2024-08-21", "2024-08-21", freq="D", state="BY"
        )
        assert df_nw.loc["2024-08-21", "is_school_holiday"] == 0
        assert df_by.loc["2024-08-21", "is_school_holiday"] == 1

    def test_inclusive_edges(self):
        # First and last day of NW Sommerferien must be 1.
        df = create_school_holiday_df("2024-07-08", "2024-08-20", freq="D", state="NW")
        assert df.loc["2024-07-08", "is_school_holiday"] == 1
        assert df.loc["2024-08-20", "is_school_holiday"] == 1

    def test_hourly_broadcast_vacation_day(self):
        # 2024-07-08 is in NW Sommerferien → all 24 hours must be 1.
        df = create_school_holiday_df(
            "2024-07-08", "2024-07-08 23:00", freq="h", state="NW"
        )
        assert df.shape == (24, 1)
        assert (df["is_school_holiday"] == 1).all()

    def test_fail_safe_start_before_valid_from(self):
        with pytest.raises(ValueError, match="2022-01-01"):
            create_school_holiday_df("2021-12-01", "2022-06-01", freq="D", state="NW")

    def test_fail_safe_end_after_valid_to(self):
        with pytest.raises(ValueError, match="2027-12-31"):
            create_school_holiday_df("2027-06-01", "2028-06-01", freq="D", state="NW")

    def test_country_code_not_de_raises(self):
        with pytest.raises(ValueError, match="country_code"):
            create_school_holiday_df(
                "2024-01-01", "2024-01-31", freq="D", country_code="FR"
            )

    def test_single_column(self):
        df = create_school_holiday_df("2024-07-01", "2024-07-31", freq="D", state="NW")
        assert df.columns.tolist() == ["is_school_holiday"]
        assert df.shape[1] == 1


class TestGetSchoolHolidayFeatures:
    """Unit tests for get_school_holiday_features."""

    def _make_data(self, start_str: str, n_data: int = 48) -> pd.DataFrame:
        return pd.DataFrame(
            {"load": range(n_data)},
            index=pd.date_range(start_str, periods=n_data, freq="h", tz="UTC"),
        )

    def test_shape_and_columns(self):
        forecast_horizon = 24
        n_data = 48
        data = self._make_data("2024-07-06", n_data)
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))
        feats = get_school_holiday_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            state="NW",
        )
        assert feats.shape == (n_data + forecast_horizon, 1)
        assert feats.columns.tolist() == ["is_school_holiday"]

    def test_dtype_no_nan_binary(self):
        forecast_horizon = 24
        n_data = 48
        data = self._make_data("2024-07-06", n_data)
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))
        feats = get_school_holiday_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            state="NW",
        )
        assert feats["is_school_holiday"].dtype == int
        assert not feats["is_school_holiday"].isna().any()
        assert set(feats["is_school_holiday"].unique()).issubset({0, 1})

    def test_determinism(self):
        forecast_horizon = 24
        n_data = 48
        data = self._make_data("2024-07-06", n_data)
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))
        a = get_school_holiday_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            state="NW",
        )
        b = get_school_holiday_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            state="NW",
        )
        pd.testing.assert_frame_equal(a, b)

    def test_known_dates_nw_sommerferien_2024(self):
        # Grid starts 2024-07-06; NW Sommerferien starts 2024-07-08.
        forecast_horizon = 24
        n_data = 48
        data = self._make_data("2024-07-06", n_data)
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))
        feats = get_school_holiday_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            state="NW",
        )
        assert feats.loc["2024-07-07 00:00:00+00:00", "is_school_holiday"] == 0
        assert feats.loc["2024-07-08 00:00:00+00:00", "is_school_holiday"] == 1

    def test_fail_safe_start_before_valid_from(self):
        forecast_horizon = 24
        n_data = 48
        data = self._make_data("2021-11-01", n_data)
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))
        with pytest.raises(ValueError, match="2022-01-01"):
            get_school_holiday_features(
                data=data,
                start=start,
                cov_end=cov_end,
                forecast_horizon=forecast_horizon,
                state="NW",
            )

    def test_fail_safe_end_after_valid_to(self):
        # Start late enough that cov_end (start + 71h) extends past 2027-12-31.
        # 2027-12-31 00:00 + 71h = 2028-01-02 23:00 → beyond valid_to.
        forecast_horizon = 24
        n_data = 48
        data = self._make_data("2027-12-31", n_data)
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))
        with pytest.raises(ValueError, match="2027-12-31"):
            get_school_holiday_features(
                data=data,
                start=start,
                cov_end=cov_end,
                forecast_horizon=forecast_horizon,
                state="NW",
            )

    def test_country_code_not_de_raises(self):
        forecast_horizon = 24
        n_data = 48
        data = self._make_data("2024-07-06", n_data)
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))
        with pytest.raises(ValueError, match="country_code"):
            get_school_holiday_features(
                data=data,
                start=start,
                cov_end=cov_end,
                forecast_horizon=forecast_horizon,
                country_code="FR",
            )


class TestSelectExogenousSchoolHoliday:
    """Unit tests for select_exogenous_features with school-holiday toggle."""

    def _make_exog_with_school_holiday(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        import numpy as np

        idx = pd.date_range("2024-07-01", periods=24, freq="h", tz="UTC")
        weather = pd.DataFrame({"wind_speed": np.ones(24)}, index=idx)
        exog = pd.DataFrame(
            {
                "hour_sin": np.sin(2 * 3.14159 * idx.hour / 24),
                "hour_cos": np.cos(2 * 3.14159 * idx.hour / 24),
                "wind_speed": weather["wind_speed"],
                "is_school_holiday": 0,
            },
            index=idx,
        )
        return exog, weather

    def test_include_school_holiday_true_includes_column(self):
        exog, weather = self._make_exog_with_school_holiday()
        selected = select_exogenous_features(
            exogenous_features=exog,
            weather_aligned=weather,
            include_school_holiday_features=True,
        )
        assert "is_school_holiday" in selected

    def test_include_school_holiday_false_excludes_column(self):
        exog, weather = self._make_exog_with_school_holiday()
        selected = select_exogenous_features(
            exogenous_features=exog,
            weather_aligned=weather,
            include_school_holiday_features=False,
        )
        assert "is_school_holiday" not in selected

    def test_include_school_holiday_default_excludes_column(self):
        exog, weather = self._make_exog_with_school_holiday()
        selected = select_exogenous_features(
            exogenous_features=exog,
            weather_aligned=weather,
        )
        assert "is_school_holiday" not in selected

    def test_missing_column_silently_absent(self):
        """When is_school_holiday is not in exog, enabling the flag is a no-op."""
        import numpy as np

        idx = pd.date_range("2024-07-01", periods=24, freq="h", tz="UTC")
        weather = pd.DataFrame({"wind_speed": np.ones(24)}, index=idx)
        exog = pd.DataFrame(
            {"hour_sin": np.zeros(24), "hour_cos": np.zeros(24)},
            index=idx,
        )
        selected = select_exogenous_features(
            exogenous_features=exog,
            weather_aligned=weather,
            include_school_holiday_features=True,
        )
        assert "is_school_holiday" not in selected


class TestBundledDataIntegrity:
    """Tests that verify the bundled CSV files are well-formed."""

    def test_sixteen_states(self):
        df, _, _ = load_school_holidays_de()
        assert len(df["state"].unique()) == 16

    def test_all_expected_state_codes_present(self):
        expected = {
            "BW",
            "BY",
            "BE",
            "BB",
            "HB",
            "HH",
            "HE",
            "MV",
            "NI",
            "NW",
            "RP",
            "SL",
            "SN",
            "ST",
            "SH",
            "TH",
        }
        df, _, _ = load_school_holidays_de()
        assert set(df["state"].unique()) == expected

    def test_schema(self):
        df, _, _ = load_school_holidays_de()
        assert list(df.columns) == ["state", "name", "start_date", "end_date"]

    def test_row_count_in_range(self):
        df, _, _ = load_school_holidays_de()
        assert 500 <= len(df) <= 650, f"Row count {len(df)} not in [500, 650]"

    def test_start_le_end(self):
        df, _, _ = load_school_holidays_de()
        assert (df["start_date"] <= df["end_date"]).all()

    def test_meta_parses(self):
        _, valid_from, valid_to = load_school_holidays_de()
        assert valid_from == pd.Timestamp("2022-01-01")
        assert valid_to == pd.Timestamp("2027-12-31")

    def test_date_columns_are_timestamps(self):
        df, _, _ = load_school_holidays_de()
        assert pd.api.types.is_datetime64_any_dtype(df["start_date"])
        assert pd.api.types.is_datetime64_any_dtype(df["end_date"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

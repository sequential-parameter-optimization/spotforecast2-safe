# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for stats.errors (error_summary, error_profile)."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.stats.errors import error_profile, error_summary

IDX = pd.date_range("2026-06-10 00:00", periods=48, freq="h", tz="UTC")


class TestErrorSummary:
    def test_hand_checkable_values(self):
        table = error_summary({"m": pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])})
        col = table["m"]
        assert col["mean"] == 0.0
        assert col["median"] == 0.0
        assert col["min"] == -2.0
        assert col["max"] == 2.0
        assert col["std"] == pytest.approx(np.std([-2, -1, 0, 1, 2], ddof=1))

    def test_row_order_matches_manuscript_table(self):
        table = error_summary({"m": pd.Series([1.0, 2.0])})
        assert list(table.index) == [
            "mean",
            "median",
            "q0.05",
            "q0.95",
            "std",
            "min",
            "max",
        ]

    def test_quantiles_match_pandas(self):
        e = pd.Series(np.arange(100.0))
        table = error_summary({"m": e})
        assert table.loc["q0.05", "m"] == e.quantile(0.05)
        assert table.loc["q0.95", "m"] == e.quantile(0.95)

    def test_custom_quantiles(self):
        table = error_summary(
            {"m": pd.Series([1.0, 2.0, 3.0])}, quantiles=(0.25, 0.5, 0.75)
        )
        assert "q0.25" in table.index and "q0.75" in table.index

    def test_column_order_follows_mapping(self):
        table = error_summary({"z": pd.Series([1.0]), "a": pd.Series([2.0])})
        assert list(table.columns) == ["z", "a"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            error_summary({})

    def test_bad_quantile_raises(self):
        with pytest.raises(ValueError, match="outside"):
            error_summary({"m": pd.Series([1.0])}, quantiles=(1.5,))

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            error_summary({"m": [1.0, 2.0]})  # type: ignore[dict-item]


class TestErrorProfile:
    def test_hourly_identity_profile(self):
        errors = {"m": pd.Series(np.tile(np.arange(24.0), 2), index=IDX)}
        profile = error_profile(errors, by="hour")
        assert list(profile.index) == list(range(24))
        assert list(profile["m"]) == list(np.arange(24.0))

    def test_matches_manuscript_groupby(self):
        rng = np.random.default_rng(11)
        e = pd.Series(rng.normal(0, 100, 48), index=IDX)
        profile = error_profile({"m": e}, by="hour")
        manual = e.groupby(e.index.hour).mean()
        assert list(profile["m"]) == list(manual)

    def test_two_entries_align_on_union(self):
        e1 = pd.Series(1.0, index=IDX[:24])
        e2 = pd.Series(2.0, index=IDX[24:])
        profile = error_profile({"a": e1, "b": e2}, by="hour")
        assert (profile["a"] == 1.0).all() and (profile["b"] == 2.0).all()

    def test_dayofweek_key(self):
        profile = error_profile({"m": pd.Series(1.0, index=IDX)}, by="dayofweek")
        # 2026-06-10 is a Wednesday (2); 48 h span Wednesday and Thursday.
        assert list(profile.index) == [2, 3]

    def test_agg_median(self):
        # Three days: every hour-of-day group is [0, 0, 9] -> median 0, mean 3.
        idx3 = pd.date_range("2026-06-10 00:00", periods=72, freq="h", tz="UTC")
        e = pd.Series(np.repeat([0.0, 0.0, 9.0], 24), index=idx3)
        by_hour_median = error_profile({"m": e}, by="hour", agg="median")
        by_hour_mean = error_profile({"m": e}, by="hour", agg="mean")
        assert (by_hour_median["m"] == 0.0).all()
        assert (by_hour_mean["m"] == 3.0).all()

    def test_index_name_is_key(self):
        profile = error_profile({"m": pd.Series(1.0, index=IDX)}, by="hour")
        assert profile.index.name == "hour"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="by must be one of"):
            error_profile({"m": pd.Series(1.0, index=IDX)}, by="weekofyear")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            error_profile({})

    def test_non_datetime_index_raises(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            error_profile({"m": pd.Series([1.0, 2.0])})

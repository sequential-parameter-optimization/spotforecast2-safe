# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

from spotforecast2_safe.calendar import create_holiday_df


def test_create_holiday_df_christmas():
    """Test standard Christmas holidays in Germany."""
    start = "2023-12-24"
    end = "2023-12-26"
    df = create_holiday_df(start, end, freq="D", country_code="DE", state="NW")

    # 24th is usually not a full public holiday in DE/NW (Heiligabend),
    # but 25th and 26th are.
    # checking holidays library behavior for DE:
    # 24th is not a holiday by default in python-holidays for DE unless specifically added or custom.
    # 25th (1st Xmas Day) -> 1
    # 26th (2nd Xmas Day) -> 1

    assert df.loc["2023-12-25", "is_holiday"] == 1
    assert df.loc["2023-12-26", "is_holiday"] == 1
    # Note: Depending on holidays version, 24th might be treated differently,
    # but typically it's 0 or 0.5 if handled as half-day.
    # Assuming 0 based on original test.
    if "2023-12-24" in df.index:
        assert df.loc["2023-12-24", "is_holiday"] == 0


def test_hourly_frequency():
    """Test that hourly frequency works and fills the whole day."""
    start = "2023-12-25 00:00"
    end = "2023-12-25 23:00"
    df = create_holiday_df(start, end, freq="h", country_code="DE", state="NW")

    assert len(df) == 24
    assert df["is_holiday"].sum() == 24  # All hours should be holidays


def test_timezone_handling():
    """Test explicit timezone."""
    start = "2023-12-25"
    end = "2023-12-25"
    df = create_holiday_df(start, end, freq="D", tz="Europe/Berlin", country_code="DE")

    assert str(df.index.tz) == "Europe/Berlin"
    assert df.iloc[0]["is_holiday"] == 1


def test_inferred_timezone():
    """Test inferred timezone from Timestamp."""
    start = pd.Timestamp("2023-12-25", tz="US/Pacific")
    end = pd.Timestamp("2023-12-26", tz="US/Pacific")
    df = create_holiday_df(start, end, freq="D", country_code="US", state="CA")

    assert str(df.index.tz) == "US/Pacific"
    assert df.iloc[0]["is_holiday"] == 1  # Xmas


def test_empty_range():
    """Test behavior with single point or empty range if needed, though pandas handles this."""
    start = "2023-01-01"
    end = "2023-01-01"
    df = create_holiday_df(start, end, freq="D", country_code="DE")  # New Year
    assert len(df) == 1
    assert df.iloc[0]["is_holiday"] == 1


def test_mixed_naive_start_aware_end_raises_typeerror():
    """A naive ``start`` paired with a tz-aware ``end`` raises ``TypeError``.

    The source code's elif branch intends to infer the timezone from
    ``end`` when ``start`` lacks tz info, but ``pd.date_range`` rejects
    mismatched-tz endpoints before the inferred timezone is applied.
    Pinning the current contract so a future enhancement that supports
    this input shape must update the test deliberately.
    """
    import pytest

    start = "2023-12-24"
    end = pd.Timestamp("2023-12-26", tz="Europe/Berlin")
    with pytest.raises(TypeError):
        create_holiday_df(start, end, freq="D", country_code="DE", state="NW")


def test_non_de_country_code_us():
    """Non-default ``country_code`` / ``state`` plumbing reaches python-holidays."""
    df = create_holiday_df(
        "2023-07-03",
        "2023-07-05",
        freq="D",
        country_code="US",
        state="NY",
    )

    assert df["is_holiday"].tolist() == [0, 1, 0]  # Independence Day = July 4


# =============================================================================
# create_holiday_adjacency_df
# =============================================================================


from spotforecast2_safe.calendar import create_holiday_adjacency_df  # noqa: E402


class TestCreateHolidayAdjacencyDf:
    """Tests for create_holiday_adjacency_df (DE/NW unless stated, freq="D"
    unless stated — mirrors the existing test style above).
    """

    def test_ascension_brueckentag(self):
        """Ascension 2024-05-09 (Thu) makes 2024-05-10 (Fri) a Brückentag.

        05-08 Wed: before_holiday=1 (next=holiday)
        05-09 Thu: is_holiday → all 0
        05-10 Fri: prev=holiday, next=Sat(weekend) → brueckentag=1, after_holiday=1
        05-11 Sat: weekend → brueckentag=0
        05-12 Sun: weekend → brueckentag=0
        """
        df = create_holiday_adjacency_df(
            "2024-05-08", "2024-05-12", freq="D", country_code="DE", state="NW"
        )
        assert df.loc["2024-05-10", "is_brueckentag"] == 1
        # Only 05-10 is a Brückentag in this range
        brueckentag_days = df.index[df["is_brueckentag"] == 1]
        assert len(brueckentag_days) == 1

    def test_unity_day_brueckentag(self):
        """Unity Day 2024-10-03 (Thu) makes 2024-10-04 (Fri) a Brückentag."""
        df = create_holiday_adjacency_df(
            "2024-10-02", "2024-10-06", freq="D", country_code="DE", state="NW"
        )
        assert df.loc["2024-10-04", "is_brueckentag"] == 1
        brueckentag_days = df.index[df["is_brueckentag"] == 1]
        assert len(brueckentag_days) == 1

    def test_christmas_cluster(self):
        """Christmas cluster 2024-12-23 to 2024-12-28.

        - is_before_holiday 1 only on 12-24 (next = 12-25 Christmas)
        - is_after_holiday 1 only on 12-27 (prev = 12-26 Boxing Day)
        - is_brueckentag 1 only on 12-27 (working Fri between Boxing Day
          and the weekend)
        - 12-27 has BOTH after_holiday and brueckentag (overlap allowed).
        """
        df = create_holiday_adjacency_df(
            "2024-12-23", "2024-12-28", freq="D", country_code="DE", state="NW"
        )
        assert df.loc["2024-12-24", "is_before_holiday"] == 1
        # Overlap: 12-27 has both after_holiday and brueckentag
        assert df.loc["2024-12-27", "is_after_holiday"] == 1
        assert df.loc["2024-12-27", "is_brueckentag"] == 1
        # Only 12-24 has before_holiday in this range
        before_days = df.index[df["is_before_holiday"] == 1]
        assert len(before_days) == 1

    def test_may_day_no_brueckentag(self):
        """May Day (2024-05-01, Wed) — no Brückentag possible in this window.

        04-30 Tue: is_before_holiday=1 (next=01-May holiday)
        05-01 Wed: is_holiday → all 0
        05-02 Thu: is_after_holiday=1 (prev=01-May holiday); not brueckentag
                   because next(05-03 Fri) is a normal working day, so the
                   day is not sandwiched between two non-working days.
        """
        df = create_holiday_adjacency_df(
            "2024-04-30", "2024-05-02", freq="D", country_code="DE", state="NW"
        )
        assert df["is_brueckentag"].sum() == 0
        assert df.loc["2024-04-30", "is_before_holiday"] == 1
        assert df.loc["2024-05-02", "is_after_holiday"] == 1

    def test_weekend_brueckentag_always_zero(self):
        """Saturdays and Sundays must always have is_brueckentag == 0."""
        df = create_holiday_adjacency_df(
            "2024-01-01", "2024-01-31", freq="D", country_code="DE", state="NW"
        )
        weekend_mask = df.index.dayofweek >= 5
        assert (df.loc[weekend_mask, "is_brueckentag"] == 0).all()

    def test_disjoint_from_is_holiday(self):
        """No timestamp can have is_holiday==1 together with any adjacency flag==1."""
        from spotforecast2_safe.calendar import create_holiday_df

        df_hol = create_holiday_df(
            "2024-01-01", "2024-12-31", freq="D", country_code="DE", state="NW"
        )
        df_adj = create_holiday_adjacency_df(
            "2024-01-01", "2024-12-31", freq="D", country_code="DE", state="NW"
        )
        shared = df_hol.index.intersection(df_adj.index)
        is_holiday = df_hol.loc[shared, "is_holiday"]
        any_adjacency = (
            df_adj.loc[shared, "is_brueckentag"]
            | df_adj.loc[shared, "is_before_holiday"]
            | df_adj.loc[shared, "is_after_holiday"]
        )
        # No row may have both is_holiday==1 and any adjacency flag==1
        assert not (is_holiday & any_adjacency).any()

    def test_hourly_expansion_brueckentag(self):
        """All 24 hourly rows for a Brückentag must have is_brueckentag==1."""
        # 2024-10-04 is a Brückentag (day after Unity Day, before weekend)
        df = create_holiday_adjacency_df(
            "2024-10-04 00:00",
            "2024-10-04 23:00",
            freq="h",
            country_code="DE",
            state="NW",
        )
        assert len(df) == 24
        assert df["is_brueckentag"].sum() == 24
        assert (df["is_brueckentag"] == 1).all()

    def test_single_day_brueckentag_boundary(self):
        """Single-day range spanning a Brückentag must emit is_brueckentag==1."""
        df = create_holiday_adjacency_df(
            "2024-05-10", "2024-05-10", freq="D", country_code="DE", state="NW"
        )
        assert df.loc["2024-05-10", "is_brueckentag"] == 1

    def test_single_day_before_holiday_boundary(self):
        """Single-day range spanning a before-holiday day must emit flag==1."""
        # 2024-12-24 (Tue) → next = 2024-12-25 Christmas
        df = create_holiday_adjacency_df(
            "2024-12-24", "2024-12-24", freq="D", country_code="DE", state="NW"
        )
        assert df.loc["2024-12-24", "is_before_holiday"] == 1

    def test_non_de_us_around_independence_day(self):
        """US Independence Day 2024 (Thu 2024-07-04, NY).

        07-03 Wed: is_before_holiday=1 (next=holiday)
        07-04 Thu: is_holiday → all 0
        07-05 Fri: is_after_holiday=1 (prev=holiday); next=Sat(weekend)
                   and prev=holiday → brueckentag=1 as well.
        """
        df = create_holiday_adjacency_df(
            "2024-07-03", "2024-07-05", freq="D", country_code="US", state="NY"
        )
        assert df.loc["2024-07-03", "is_before_holiday"] == 1
        assert df.loc["2024-07-05", "is_after_holiday"] == 1
        assert df.loc["2024-07-05", "is_brueckentag"] == 1

    def test_dtype_integer_values_subset_01(self):
        """All three columns must have integer dtype and values in {0, 1}."""
        import numpy as np

        df = create_holiday_adjacency_df(
            "2024-01-01", "2024-03-31", freq="D", country_code="DE", state="NW"
        )
        for col in ["is_brueckentag", "is_before_holiday", "is_after_holiday"]:
            assert df[col].dtype in (np.int64, np.int32, int, "int64", "int32")
            assert set(df[col].unique()).issubset({0, 1})

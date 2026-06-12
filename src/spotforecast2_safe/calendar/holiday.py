# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Holiday DataFrame generators and forecasting-grid alignment.

Provides:

- `create_holiday_df()` — build a tz-aware DataFrame whose index covers
  ``[start, end]`` at the requested frequency and whose single column
  ``is_holiday`` is ``1`` on public-holiday days and ``0`` otherwise.
- `get_holiday_features()` — align those indicators to a forecast grid,
  validating temporal coverage via
  `curate_holidays()`.
- `create_holiday_adjacency_df()` — build a tz-aware DataFrame with three
  binary int columns ``is_brueckentag``, ``is_before_holiday``, and
  ``is_after_holiday``, all disjoint from ``is_holiday``.
- `get_holiday_adjacency_features()` — align those adjacency indicators to a
  forecast grid, mirroring the contract of `get_holiday_features()`.
- `create_day_type_df()` / `get_day_type_features()` — a day-type refinement
  of the holiday column: a binary working-day indicator and an integer
  day-type class (working day / Saturday / Sunday / public holiday), derived
  purely from the weekday and the public-holiday calendar.
- `create_school_holiday_df()` / `get_school_holiday_features()` — per-Bundesland
  school-holiday binary indicator (``is_school_holiday``) built from the bundled
  OpenHolidays API dataset (ODbL-1.0), coverage 2022-01-01 to 2027-12-31.
  Only ``country_code="DE"`` is supported; 16 German Bundesländer are available.
  Requests outside the covered date range raise ``ValueError`` (fail-safe).
"""

from typing import Union

import holidays
import pandas as pd

from spotforecast2_safe.utils.convert_to_utc import to_utc_timestamp


def create_holiday_df(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Create a DataFrame with datetime index and a binary holiday indicator column.

    Expands daily holidays to all timestamps in the desired frequency.

    Args:
        start: Start date/datetime.
        end: End date/datetime.
        tz: Timezone to use if not inferred from start/end.
        freq: Frequency of the resulting DataFrame.
        country_code: Country code for holidays (e.g. "DE", "US").
        state: State code for holidays (e.g. "NW", "CA").

    Returns:
        pd.DataFrame: DataFrame with index covering [start, end] at `freq`,
                      and an ``is_holiday`` column (1 if holiday, 0 otherwise).

    Examples:
        ```{python}
        from spotforecast2_safe.calendar import create_holiday_df

        # Christmas Day and Boxing Day are public holidays in Germany (NW).
        df = create_holiday_df("2023-12-24", "2023-12-26", freq="D")
        print("is_holiday:", df["is_holiday"].tolist())
        assert df["is_holiday"].tolist() == [0, 1, 1]
        assert df.shape == (3, 1)
        ```
    """
    # If start/end are Timestamps with timezones, use that timezone instead of
    # the default. This avoids conflicts when timezone-aware Timestamps are
    # passed with a different tz parameter.
    inferred_tz = None
    if isinstance(start, pd.Timestamp) and start.tz is not None:
        inferred_tz = str(start.tz)
    elif isinstance(end, pd.Timestamp) and end.tz is not None:
        inferred_tz = str(end.tz)

    if inferred_tz is not None:
        # pandas infers the tz from the Timestamps; don't pass tz= alongside.
        full_index = pd.date_range(start=start, end=end, freq=freq)
    else:
        full_index = pd.date_range(start=start, end=end, freq=freq, tz=tz)

    country_holidays = holidays.country_holidays(country_code, subdiv=state)
    unique_dates = pd.Series(full_index.normalize().unique())
    is_holiday_series = pd.Series(
        [1 if d.date() in country_holidays else 0 for d in unique_dates],
        index=unique_dates,
    )

    df_full = pd.DataFrame(index=full_index)
    df_full["is_holiday"] = (
        full_index.normalize().map(is_holiday_series).fillna(0).astype(int)
    )
    return df_full


def get_holiday_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Build public-holiday indicators and align them to a regular time grid.

    Generates holiday indicators via `create_holiday_df()`, validates
    coverage with
    `curate_holidays()`,
    and reindexes the result to a full ``[start, cov_end]`` grid with
    ``fill_value=0`` so that non-holiday timestamps are always zero.

    Args:
        data: Reference time series DataFrame used for temporal coverage
            validation inside
            `curate_holidays()`.
        start: Start timestamp.  String values are parsed with
            ``utc=True``.
        cov_end: Inclusive end timestamp (should cover the full forecast
            horizon).  String values are parsed with ``utc=True``.
        forecast_horizon: Number of forecast steps ahead; passed to
            `curate_holidays()`.
        tz: Timezone applied to the generated index and passed to
            `create_holiday_df()`.  Defaults to ``"UTC"``.
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        country_code: ISO 3166-1 alpha-2 country code.  Defaults to
            ``"DE"`` (Germany).
        state: Sub-national state/region code.  Defaults to ``"NW"``
            (North Rhine-Westphalia).

    Returns:
        pd.DataFrame: DataFrame with a single integer column
        ``is_holiday``.  The index is a tz-aware
        `DatetimeIndex` with the requested ``freq``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.calendar import get_holiday_features

        # Build a minimal synthetic reference DataFrame.
        # curate_holidays requires: holiday_df.shape[0] == data.shape[0] + forecast_horizon.
        # With n_data=48 rows and forecast_horizon=24, we need 72 hourly steps total,
        # so cov_end = start + 71 h (inclusive date_range).
        forecast_horizon = 24
        n_data = 48
        data = pd.DataFrame(
            {"load": range(n_data)},
            index=pd.date_range("2024-01-01", periods=n_data, freq="h", tz="UTC"),
        )
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))

        hf = get_holiday_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            country_code="DE",
            state="NW",
        )
        print("shape:", hf.shape)
        print("columns:", hf.columns.tolist())
        # New Year's Day (2024-01-01) is a public holiday in Germany.
        print("Jan 1 00:00 is_holiday:", hf.loc["2024-01-01 00:00:00+00:00", "is_holiday"])
        assert hf.shape == (n_data + forecast_horizon, 1)
        assert hf.loc["2024-01-01 00:00:00+00:00", "is_holiday"] == 1
        ```
    """
    # Local import to avoid a hard cycle: preprocessing.curate_data does
    # not depend on calendar/, but reversing this could change once
    # validation logic grows.
    from spotforecast2_safe.preprocessing.curate_data import curate_holidays

    start = to_utc_timestamp(start)
    cov_end = to_utc_timestamp(cov_end)

    holiday_df = create_holiday_df(
        start=start,
        end=cov_end,
        tz=tz,
        freq=freq,
        country_code=country_code,
        state=state,
    )

    curate_holidays(holiday_df, data, forecast_horizon=forecast_horizon)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=tz)
    return holiday_df.reindex(extended_index, fill_value=0).astype(int)


def create_holiday_adjacency_df(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Create a DataFrame with binary adjacency indicators for public holidays.

    Returns three int columns, all disjoint from ``is_holiday``:

    - ``is_brueckentag``: 1 when day ``d`` is a working day (Mon–Fri AND not a
      public holiday) AND both ``d-1`` and ``d+1`` are non-working (public
      holiday **or** weekend, i.e. ``dayofweek >= 5``).  Saturday, Sunday, and
      public holidays are always 0.
    - ``is_before_holiday``: 1 when ``d+1`` is a public holiday AND ``d`` is
      not itself a public holiday.
    - ``is_after_holiday``: 1 when ``d-1`` is a public holiday AND ``d`` is
      not itself a public holiday.

    A day may be flagged by more than one column simultaneously; for example,
    2024-12-27 (Friday after Christmas/Boxing Day, before a long weekend) is
    both ``is_after_holiday`` and ``is_brueckentag``.

    Weekend membership is determined by ``dayofweek >= 5``; the ``holidays``
    library knows nothing about weekends, so this rule is applied explicitly.

    Boundary rule: the first day of the requested range needs to know the
    holiday/weekend status of the day before ``start``, and the last day needs
    the status of the day after ``end``.  Neighbour-day look-ups (``d ± 1d``)
    are performed directly against the holiday calendar object, which resolves
    any date on demand, so edge rows are never incorrectly zeroed.

    Args:
        start: Start date/datetime.
        end: End date/datetime.
        tz: Timezone to use if not inferred from start/end.
        freq: Frequency of the resulting DataFrame.
        country_code: Country code for holidays (e.g. ``"DE"``, ``"US"``).
        state: State code for holidays (e.g. ``"NW"``, ``"CA"``).

    Returns:
        pd.DataFrame: DataFrame with index covering ``[start, end]`` at *freq*
        and three integer columns in order: ``is_brueckentag``,
        ``is_before_holiday``, ``is_after_holiday``.  All values are in
        ``{0, 1}``; no NaNs.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.calendar import create_holiday_adjacency_df

        # Unity Day 2024 (Thu 2024-10-03) is a public holiday in Germany.
        # 2024-10-04 (Fri) is therefore sandwiched between the holiday and the
        # weekend (Sat 2024-10-05, Sun 2024-10-06) → Brückentag.
        df = create_holiday_adjacency_df(
            "2024-10-02", "2024-10-06", freq="D", country_code="DE", state="NW"
        )
        print(df)
        assert df.loc["2024-10-04", "is_brueckentag"] == 1
        assert df.loc["2024-10-03", "is_brueckentag"] == 0  # is_holiday, not Brückentag
        assert df.loc["2024-10-02", "is_before_holiday"] == 1  # day before Unity Day
        assert df.loc["2024-10-04", "is_after_holiday"] == 1   # day after Unity Day
        ```
    """
    # --- tz inference: mirror create_holiday_df verbatim ---
    inferred_tz = None
    if isinstance(start, pd.Timestamp) and start.tz is not None:
        inferred_tz = str(start.tz)
    elif isinstance(end, pd.Timestamp) and end.tz is not None:
        inferred_tz = str(end.tz)

    if inferred_tz is not None:
        full_index = pd.date_range(start=start, end=end, freq=freq)
    else:
        full_index = pd.date_range(start=start, end=end, freq=freq, tz=tz)

    # Build the country holiday calendar.
    cal = holidays.country_holidays(country_code, subdiv=state)

    # Unique calendar days in the requested range.
    unique_days = pd.DatetimeIndex(full_index.normalize().unique())

    if len(unique_days) == 0:
        # Edge case: empty index — return empty DataFrame with the right columns.
        return pd.DataFrame(
            index=full_index,
            columns=["is_brueckentag", "is_before_holiday", "is_after_holiday"],
            dtype=int,
        )

    # Neighbour days (prev/next) are looked up directly against the holiday
    # calendar, so days just outside the requested range are evaluated
    # correctly and range edges never receive wrong zeros.
    def _is_non_working(d: pd.Timestamp) -> bool:
        """Return True when d is a weekend day OR a public holiday."""
        return d.dayofweek >= 5 or d.date() in cal

    def _is_holiday_only(d: pd.Timestamp) -> bool:
        return d.date() in cal

    # Build per-day Series for each flag.
    brueckentag_map: dict = {}
    before_hol_map: dict = {}
    after_hol_map: dict = {}

    for d in unique_days:
        is_hol = _is_holiday_only(d)
        is_weekend = d.dayofweek >= 5
        prev_d = d - pd.Timedelta(days=1)
        next_d = d + pd.Timedelta(days=1)

        # is_brueckentag: working day squeezed between two non-working days.
        if not is_hol and not is_weekend:
            brueckentag_map[d] = int(
                _is_non_working(prev_d) and _is_non_working(next_d)
            )
        else:
            brueckentag_map[d] = 0

        # is_before_holiday: day before a public holiday, not itself a holiday.
        before_hol_map[d] = int(not is_hol and _is_holiday_only(next_d))

        # is_after_holiday: day after a public holiday, not itself a holiday.
        after_hol_map[d] = int(not is_hol and _is_holiday_only(prev_d))

    brueckentag_series = pd.Series(brueckentag_map)
    before_hol_series = pd.Series(before_hol_map)
    after_hol_series = pd.Series(after_hol_map)

    df_full = pd.DataFrame(index=full_index)
    norm = full_index.normalize()
    df_full["is_brueckentag"] = norm.map(brueckentag_series).fillna(0).astype(int)
    df_full["is_before_holiday"] = norm.map(before_hol_series).fillna(0).astype(int)
    df_full["is_after_holiday"] = norm.map(after_hol_series).fillna(0).astype(int)
    return df_full


def get_holiday_adjacency_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Build holiday-adjacency indicators and align them to a regular time grid.

    Generates ``is_brueckentag``, ``is_before_holiday``, and
    ``is_after_holiday`` indicators via `create_holiday_adjacency_df()`,
    validates temporal coverage with `curate_holidays()`, and reindexes the
    result to a full ``[start, cov_end]`` grid with ``fill_value=0`` so that
    non-flagged timestamps are always zero.

    All three columns are disjoint from ``is_holiday``: a public holiday itself
    never receives a non-zero value in any of the three adjacency columns.

    Args:
        data: Reference time series DataFrame used for temporal coverage
            validation inside `curate_holidays()`.
        start: Start timestamp.  String values are parsed with ``utc=True``.
        cov_end: Inclusive end timestamp (should cover the full forecast
            horizon).  String values are parsed with ``utc=True``.
        forecast_horizon: Number of forecast steps ahead; passed to
            `curate_holidays()`.
        tz: Timezone applied to the generated index and passed to
            `create_holiday_adjacency_df()`.  Defaults to ``"UTC"``.
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        country_code: ISO 3166-1 alpha-2 country code.  Defaults to
            ``"DE"`` (Germany).
        state: Sub-national state/region code.  Defaults to ``"NW"``
            (North Rhine-Westphalia).

    Returns:
        pd.DataFrame: DataFrame with three integer columns
        ``is_brueckentag``, ``is_before_holiday``, ``is_after_holiday``.
        The index is a tz-aware `DatetimeIndex` with the requested *freq*.
        All values are in ``{0, 1}``; no NaNs.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.calendar import get_holiday_adjacency_features

        # Build a minimal synthetic reference DataFrame.
        # curate_holidays requires:
        #   adjacency_df.shape[0] == data.shape[0] + forecast_horizon.
        # With n_data=48 rows and forecast_horizon=24, we need 72 hourly
        # steps total, so cov_end = start + 71 h (inclusive date_range).
        forecast_horizon = 24
        n_data = 48
        data = pd.DataFrame(
            {"load": range(n_data)},
            index=pd.date_range("2024-01-01", periods=n_data, freq="h", tz="UTC"),
        )
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))

        adj = get_holiday_adjacency_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            country_code="DE",
            state="NW",
        )
        print("shape:", adj.shape)
        print("columns:", adj.columns.tolist())
        assert adj.shape == (n_data + forecast_horizon, 3)
        assert list(adj.columns) == [
            "is_brueckentag", "is_before_holiday", "is_after_holiday"
        ]
        ```
    """
    from spotforecast2_safe.preprocessing.curate_data import curate_holidays

    start = to_utc_timestamp(start)
    cov_end = to_utc_timestamp(cov_end)

    adjacency_df = create_holiday_adjacency_df(
        start=start,
        end=cov_end,
        tz=tz,
        freq=freq,
        country_code=country_code,
        state=state,
    )

    curate_holidays(adjacency_df, data, forecast_horizon=forecast_horizon)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=tz)
    return adjacency_df.reindex(extended_index, fill_value=0).astype(int)


# Integer codes for the ``day_type`` column. Ordering is by "working intensity"
# (a working day differs most from a holiday), but tree models split on the
# values regardless of order.
DAY_TYPE_WORKDAY = 0
DAY_TYPE_SATURDAY = 1
DAY_TYPE_SUNDAY = 2
DAY_TYPE_HOLIDAY = 3


def create_day_type_df(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Create a day-type refinement of the public-holiday column.

    Returns two integer columns derived purely from the weekday and the
    public-holiday calendar (pure calendar arithmetic — known years ahead,
    leakage-free):

    - ``is_workday``: ``1`` when the day is Monday–Friday **and** not a public
      holiday, else ``0``.
    - ``day_type``: an integer class with public-holiday precedence —
      ``0`` working day, ``1`` Saturday (non-holiday), ``2`` Sunday
      (non-holiday), ``3`` public holiday (any weekday). A public holiday that
      falls on a weekend is still classed as ``3``.

    These remove some of the worst single-day errors a plain holiday flag
    leaves behind (Ziel 2018, ``ziel18a``).

    Args:
        start: Start date/datetime.
        end: End date/datetime.
        tz: Timezone to use if not inferred from start/end.
        freq: Frequency of the resulting DataFrame.
        country_code: Country code for holidays (e.g. ``"DE"``, ``"US"``).
        state: State code for holidays (e.g. ``"NW"``, ``"CA"``).

    Returns:
        pd.DataFrame: Index covering ``[start, end]`` at *freq* with integer
        columns ``is_workday`` and ``day_type``; no NaNs.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.calendar import create_day_type_df

        # 2024-01-01 Mon = New Year (holiday), 02 Tue = workday,
        # 06 Sat, 07 Sun.
        df = create_day_type_df("2024-01-01", "2024-01-07", freq="D")
        print(df["is_workday"].tolist())
        print(df["day_type"].tolist())
        assert df.loc["2024-01-01", "day_type"] == 3  # holiday
        assert df.loc["2024-01-02", "is_workday"] == 1
        assert df.loc["2024-01-06", "day_type"] == 1  # Saturday
        assert df.loc["2024-01-07", "day_type"] == 2  # Sunday
        ```
    """
    inferred_tz = None
    if isinstance(start, pd.Timestamp) and start.tz is not None:
        inferred_tz = str(start.tz)
    elif isinstance(end, pd.Timestamp) and end.tz is not None:
        inferred_tz = str(end.tz)

    if inferred_tz is not None:
        full_index = pd.date_range(start=start, end=end, freq=freq)
    else:
        full_index = pd.date_range(start=start, end=end, freq=freq, tz=tz)

    cal = holidays.country_holidays(country_code, subdiv=state)
    unique_days = pd.DatetimeIndex(full_index.normalize().unique())

    def _day_type(d: pd.Timestamp) -> int:
        if d.date() in cal:
            return DAY_TYPE_HOLIDAY
        dow = d.dayofweek
        if dow == 5:
            return DAY_TYPE_SATURDAY
        if dow == 6:
            return DAY_TYPE_SUNDAY
        return DAY_TYPE_WORKDAY

    day_type_series = pd.Series({d: _day_type(d) for d in unique_days}, dtype="int64")
    is_workday_series = (day_type_series == DAY_TYPE_WORKDAY).astype(int)

    df_full = pd.DataFrame(index=full_index)
    norm = full_index.normalize()
    df_full["is_workday"] = norm.map(is_workday_series).fillna(0).astype(int)
    df_full["day_type"] = norm.map(day_type_series).fillna(DAY_TYPE_WORKDAY).astype(int)
    return df_full


def get_day_type_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Build day-type indicators and align them to a regular time grid.

    Generates ``is_workday`` and ``day_type`` via `create_day_type_df()`,
    validates temporal coverage with `curate_holidays()`, and reindexes onto
    the full ``[start, cov_end]`` grid. Trailing/leading grid cells outside the
    generated range are filled with the working-day defaults
    (``is_workday=0`` is wrong for a workday, so the grid is generated to fully
    cover the request and no fill is expected; ``fill_value`` only guards the
    degenerate empty-overlap case).

    Args:
        data: Reference time series DataFrame used for temporal coverage
            validation inside `curate_holidays()`.
        start: Start timestamp. String values are parsed with ``utc=True``.
        cov_end: Inclusive end timestamp (should cover the full forecast
            horizon). String values are parsed with ``utc=True``.
        forecast_horizon: Number of forecast steps ahead; passed to
            `curate_holidays()`.
        tz: Timezone applied to the generated index. Defaults to ``"UTC"``.
        freq: Pandas-compatible frequency string. Defaults to ``"h"``.
        country_code: ISO 3166-1 alpha-2 country code. Defaults to ``"DE"``.
        state: Sub-national state/region code. Defaults to ``"NW"``.

    Returns:
        pd.DataFrame: Integer columns ``is_workday`` and ``day_type``; tz-aware
        `DatetimeIndex` with the requested *freq*.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.calendar import get_day_type_features

        forecast_horizon = 24
        n_data = 48
        data = pd.DataFrame(
            {"load": range(n_data)},
            index=pd.date_range("2024-01-01", periods=n_data, freq="h", tz="UTC"),
        )
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))

        feats = get_day_type_features(
            data=data, start=start, cov_end=cov_end,
            forecast_horizon=forecast_horizon,
        )
        print("columns:", feats.columns.tolist())
        print("shape:", feats.shape)
        # 2024-01-01 is New Year (holiday) → day_type 3, not a workday.
        assert feats.loc["2024-01-01 00:00:00+00:00", "day_type"] == 3
        assert feats.loc["2024-01-01 00:00:00+00:00", "is_workday"] == 0
        assert feats.shape == (n_data + forecast_horizon, 2)
        ```
    """
    from spotforecast2_safe.preprocessing.curate_data import curate_holidays

    start = to_utc_timestamp(start)
    cov_end = to_utc_timestamp(cov_end)

    day_type_df = create_day_type_df(
        start=start,
        end=cov_end,
        tz=tz,
        freq=freq,
        country_code=country_code,
        state=state,
    )

    curate_holidays(day_type_df, data, forecast_horizon=forecast_horizon)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=tz)
    return day_type_df.reindex(extended_index, fill_value=0).astype(int)


# ---------------------------------------------------------------------------
# School-holiday features (per-Bundesland, Germany only)
# ---------------------------------------------------------------------------

_SCHOOL_HOLIDAY_COUNTRY_SUPPORTED = ("DE",)


def create_school_holiday_df(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Create a DataFrame with a binary school-holiday indicator for a German state.

    Builds a tz-aware time grid over ``[start, end]`` at *freq* and marks
    every timestamp that falls within a school-holiday period of the requested
    Bundesland as ``1``; all others are ``0``.  Both edges of each interval
    are inclusive.

    Data source: OpenHolidays API (https://openholidaysapi.org), ODbL-1.0.
    Coverage: 2022-01-01 to 2027-12-31 for all 16 German Bundesländer.

    Only ``country_code="DE"`` is supported.  Requests whose span extends
    beyond the covered range at either edge raise ``ValueError`` — there is
    no fill or extrapolation.

    Args:
        start: Start date/datetime of the requested grid.
        end: End date/datetime of the requested grid (inclusive).
        tz: Timezone for the resulting index. Ignored when *start* or *end*
            is already a tz-aware ``pd.Timestamp``.
        freq: Pandas-compatible frequency string. Defaults to ``"h"``
            (hourly).
        country_code: Must be ``"DE"`` (Germany). Any other value raises
            ``ValueError``.
        state: ISO 3166-2 subdivision short code for the Bundesland, e.g.
            ``"NW"`` (North Rhine-Westphalia), ``"BY"`` (Bavaria).  Defaults
            to ``"NW"``.

    Returns:
        pd.DataFrame: Single integer column ``is_school_holiday`` (values in
        ``{0, 1}``; no NaNs) with a tz-aware `DatetimeIndex` at *freq*.

    Raises:
        ValueError: If *country_code* is not ``"DE"``, or if the requested
            span extends beyond the dataset validity range at either edge.

    Examples:
        ```{python}
        from spotforecast2_safe.calendar import create_school_holiday_df

        # NW Sommerferien 2024: 2024-07-08 → 2024-08-20 (inclusive).
        # Day before (2024-07-07) must be 0; first day (2024-07-08) must be 1.
        df = create_school_holiday_df(
            "2024-07-06", "2024-07-10", freq="D", state="NW"
        )
        print(df)
        assert df.loc["2024-07-07", "is_school_holiday"] == 0
        assert df.loc["2024-07-08", "is_school_holiday"] == 1
        assert df.loc["2024-07-09", "is_school_holiday"] == 1
        ```
    """
    if country_code not in _SCHOOL_HOLIDAY_COUNTRY_SUPPORTED:
        raise ValueError(
            f"country_code={country_code!r} is not supported for school-holiday "
            f"features.  Only {_SCHOOL_HOLIDAY_COUNTRY_SUPPORTED} is available."
        )

    from spotforecast2_safe.data.fetch_data import load_school_holidays_de

    intervals_df, valid_from, valid_to = load_school_holidays_de()

    # Normalise start/end to timezone-naive dates for range check.
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_date_only = (
        start_ts.normalize().tz_localize(None)
        if start_ts.tz is None
        else start_ts.tz_convert(None).normalize()
    )
    end_date_only = (
        end_ts.normalize().tz_localize(None)
        if end_ts.tz is None
        else end_ts.tz_convert(None).normalize()
    )

    if start_date_only < valid_from or end_date_only > valid_to:
        raise ValueError(
            f"Requested span [{start_date_only.date()}, {end_date_only.date()}] "
            f"extends beyond the school-holiday dataset validity range "
            f"[{valid_from.date()}, {valid_to.date()}].  "
            "There is no fill or extrapolation for out-of-coverage dates."
        )

    # Build the time grid.
    # When either endpoint is already tz-aware, normalise both to a consistent
    # tz-aware form so pd.date_range gets two compatible endpoints.
    inferred_tz = None
    if isinstance(start, pd.Timestamp) and start.tz is not None:
        inferred_tz = str(start.tz)
    elif isinstance(end, pd.Timestamp) and end.tz is not None:
        inferred_tz = str(end.tz)

    if inferred_tz is not None:
        start_grid = (
            pd.Timestamp(start).tz_localize(inferred_tz)
            if pd.Timestamp(start).tz is None
            else pd.Timestamp(start)
        )
        end_grid = (
            pd.Timestamp(end).tz_localize(inferred_tz)
            if pd.Timestamp(end).tz is None
            else pd.Timestamp(end)
        )
        full_index = pd.date_range(start=start_grid, end=end_grid, freq=freq)
    else:
        full_index = pd.date_range(start=start, end=end, freq=freq, tz=tz)

    # Filter intervals to the requested state.
    state_intervals = intervals_df[intervals_df["state"] == state].copy()

    # Map each normalised calendar day to 0/1.
    unique_days = pd.DatetimeIndex(full_index.normalize().unique())

    day_flags: dict = {}
    for d in unique_days:
        d_naive = d.tz_localize(None) if d.tz is not None else d
        d_date = d_naive.normalize()
        in_holiday = (
            (state_intervals["start_date"] <= d_date)
            & (d_date <= state_intervals["end_date"])
        ).any()
        day_flags[d] = int(in_holiday)

    flag_series = pd.Series(day_flags)
    df_full = pd.DataFrame(index=full_index)
    df_full["is_school_holiday"] = (
        full_index.normalize().map(flag_series).fillna(0).astype(int)
    )
    return df_full


def get_school_holiday_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Build per-Bundesland school-holiday indicators and align them to a forecast grid.

    Generates the ``is_school_holiday`` binary indicator via
    `create_school_holiday_df()`, validates temporal coverage with
    `curate_holidays()`, and reindexes onto the full ``[start, cov_end]``
    grid with ``fill_value=0``.

    The requested span ``[start, cov_end]`` must lie entirely within the
    dataset validity range 2022-01-01 to 2027-12-31.  If either edge falls
    outside this range a ``ValueError`` is raised immediately — there is no
    fill or extrapolation.

    Only ``country_code="DE"`` is supported; passing any other value raises
    ``ValueError``.

    Args:
        data: Reference time series DataFrame used for temporal coverage
            validation inside `curate_holidays()`.
        start: Start timestamp.  String values are parsed with ``utc=True``.
        cov_end: Inclusive end timestamp covering the full forecast horizon.
            String values are parsed with ``utc=True``.
        forecast_horizon: Number of forecast steps ahead; passed to
            `curate_holidays()`.
        tz: Timezone applied to the generated index. Defaults to ``"UTC"``.
        freq: Pandas-compatible frequency string. Defaults to ``"h"``.
        country_code: Must be ``"DE"``. Any other value raises ``ValueError``.
        state: ISO 3166-2 subdivision short code for the Bundesland.
            Defaults to ``"NW"`` (North Rhine-Westphalia).

    Returns:
        pd.DataFrame: Single integer column ``is_school_holiday`` (values in
        ``{0, 1}``; no NaNs).  The index is a tz-aware `DatetimeIndex` with
        the requested *freq* and shape ``(len(data) + forecast_horizon, 1)``.

    Raises:
        ValueError: If *country_code* is not ``"DE"``, or if the requested
            span extends beyond the dataset validity range
            ``[2022-01-01, 2027-12-31]``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.calendar import get_school_holiday_features

        forecast_horizon = 24
        n_data = 48
        data = pd.DataFrame(
            {"load": range(n_data)},
            index=pd.date_range("2024-07-06", periods=n_data, freq="h", tz="UTC"),
        )
        start = data.index[0]
        cov_end = start + pd.Timedelta(hours=(n_data + forecast_horizon - 1))

        feats = get_school_holiday_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=forecast_horizon,
            state="NW",
        )
        print("shape:", feats.shape)
        print("columns:", feats.columns.tolist())
        # NW Sommerferien 2024: 2024-07-08 is a school holiday (is_school_holiday=1).
        print("2024-07-07 00:00 UTC:", feats.loc["2024-07-07 00:00:00+00:00", "is_school_holiday"])
        print("2024-07-08 00:00 UTC:", feats.loc["2024-07-08 00:00:00+00:00", "is_school_holiday"])
        assert feats.shape == (n_data + forecast_horizon, 1)
        assert feats.loc["2024-07-07 00:00:00+00:00", "is_school_holiday"] == 0
        assert feats.loc["2024-07-08 00:00:00+00:00", "is_school_holiday"] == 1
        ```
    """
    from spotforecast2_safe.preprocessing.curate_data import curate_holidays

    start = to_utc_timestamp(start)
    cov_end = to_utc_timestamp(cov_end)

    school_holiday_df = create_school_holiday_df(
        start=start,
        end=cov_end,
        tz=tz,
        freq=freq,
        country_code=country_code,
        state=state,
    )

    curate_holidays(school_holiday_df, data, forecast_horizon=forecast_horizon)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=tz)
    return school_holiday_df.reindex(extended_index, fill_value=0).astype(int)

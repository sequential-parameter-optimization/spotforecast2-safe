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

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

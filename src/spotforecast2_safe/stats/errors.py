# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Descriptive statistics of forecast errors — pure computation.

Summaries and grouped profiles of hourly forecast errors
(``forecast - actual``).  The rendering counterparts live in
`spotforecast2.plots.evaluation`.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

_PROFILE_KEYS = ("hour", "dayofweek", "month", "dayofyear")


def error_summary(
    errors: Mapping[str, pd.Series],
    *,
    quantiles: tuple[float, ...] = (0.05, 0.95),
) -> pd.DataFrame:
    """Summarise the distribution of forecast errors, one column per entry.

    For each entry the summary reports mean, median, the requested
    quantiles, standard deviation (``ddof=1``), minimum, and maximum —
    the row layout of a manuscript error-statistics table.

    Args:
        errors: Mapping of entry name to its error series
            (``forecast - actual``).
        quantiles: Quantile levels to include, each in ``(0, 1)``.
            Defaults to ``(0.05, 0.95)``.

    Returns:
        pd.DataFrame: Indexed by statistic name
        (``["mean", "median", *[f"q{q:g}"], "std", "min", "max"]``), one
        column per entry, in mapping order.

    Raises:
        TypeError: When an entry value is not a ``pd.Series``.
        ValueError: When ``errors`` is empty or a quantile is outside
            ``(0, 1)``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.stats.errors import error_summary

        errors = {"model": pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])}
        table = error_summary(errors)
        print(table.round(2).to_string())
        assert table.loc["mean", "model"] == 0.0
        assert table.loc["min", "model"] == -2.0
        ```
    """
    if not errors:
        raise ValueError("errors is empty; nothing to summarise.")
    for q in quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile {q!r} is outside (0, 1).")

    labels = ["mean", "median", *[f"q{q:g}" for q in quantiles], "std", "min", "max"]
    columns: dict[str, list[float]] = {}
    for name, series in errors.items():
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"errors[{name!r}] must be a pd.Series, got "
                f"{type(series).__name__!r}."
            )
        columns[name] = [
            float(series.mean()),
            float(series.median()),
            *[float(series.quantile(q)) for q in quantiles],
            float(series.std()),
            float(series.min()),
            float(series.max()),
        ]
    return pd.DataFrame(columns, index=pd.Index(labels, name="statistic"))


def error_profile(
    errors: Mapping[str, pd.Series],
    *,
    by: str = "hour",
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate forecast errors by a calendar key, one column per entry.

    Groups every error series by an attribute of its ``DatetimeIndex``
    (hour of day, by default) and aggregates each group — the systematic
    hour-of-day error profile of a day-ahead forecaster, for example.
    Render the result with `spotforecast2.plots.evaluation.plot_error_profile`.

    Args:
        errors: Mapping of entry name to its error series
            (``forecast - actual``), each with a ``DatetimeIndex``.
        by: Calendar key to group by. Options:
            - ``"hour"``: hour of day (0-23).
            - ``"dayofweek"``: day of week (0 = Monday).
            - ``"month"``: month of year (1-12).
            - ``"dayofyear"``: day of year (1-366).
            Defaults to ``"hour"``.
        agg: Aggregation applied per group (any pandas aggregation
            name, e.g. ``"mean"``, ``"median"``, ``"std"``). Defaults to
            ``"mean"``.

    Returns:
        pd.DataFrame: Indexed by the sorted grouping-key values, one
        column per entry, in mapping order.

    Raises:
        TypeError: When an entry value is not a ``pd.Series`` with a
            ``DatetimeIndex``.
        ValueError: When ``errors`` is empty or ``by`` is not a
            supported key.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.stats.errors import error_profile

        idx = pd.date_range("2026-06-10", periods=48, freq="h", tz="UTC")
        errors = {"model": pd.Series(np.tile(np.arange(24.0), 2), index=idx)}
        profile = error_profile(errors, by="hour")
        print(profile.head(3).to_string())
        assert list(profile.index) == list(range(24))
        assert (profile["model"] == np.arange(24.0)).all()
        ```
    """
    if not errors:
        raise ValueError("errors is empty; nothing to profile.")
    if by not in _PROFILE_KEYS:
        raise ValueError(f"by must be one of {_PROFILE_KEYS}, got {by!r}.")

    for name, series in errors.items():
        if not isinstance(series, pd.Series) or not isinstance(
            series.index, pd.DatetimeIndex
        ):
            raise TypeError(
                f"errors[{name!r}] must be a pd.Series with a DatetimeIndex."
            )

    frame = pd.concat(dict(errors), axis=1)
    grouped = frame.groupby(getattr(frame.index, by)).agg(agg)
    grouped.index.name = by
    return grouped

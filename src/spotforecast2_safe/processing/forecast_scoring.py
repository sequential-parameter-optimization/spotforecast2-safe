# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Score and compare several forecasts against a shared actual — pure computation.

Motivation (2026-06-13 team_4 post-mortem): the four-zone bottom-up sum lost to
the single aggregate ("combined") model.  Deciding whether bottom-up aggregation
helps or merely amplifies bias is an apples-to-apples comparison question: run a
backtest for each modelling approach (with
`spotforecast2_safe.backtesting.validation.backtesting_forecaster`), then score
every approach's forecast against the same actual on the same metrics.

`score_forecasts` is that second step — a pure, source-agnostic comparison
primitive.  It takes the per-approach forecast series (e.g. the 4-zone bottom-up
sum and the combined model's prediction) plus the actual, and returns a tidy
"approach x metric" table sorted by the leading metric, so the better setup is
read off directly.

`score_forecasts_by_period` extends the same metric kernel to campaign
evaluation: it scores every forecast per calendar period (per day, by default)
so the resulting long table can feed leaderboard-style aggregation
(`aggregate_period_scores`) and paired statistical comparison
(`spotforecast2_safe.stats.comparison`).  `mase_scaling_factors` supplies the
day-ahead MASE denominator for such per-period scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

#: Metrics ``score_forecasts`` can compute, in canonical order.
SUPPORTED_METRICS: tuple[str, ...] = ("mae", "rmse", "bias", "mape", "upr")


def _compute_metric(name: str, error: pd.Series, actual: pd.Series) -> float:
    """Return one metric over an aligned error/actual pair (NaN if empty)."""
    if len(error) == 0:
        return float("nan")
    if name == "mae":
        return float(error.abs().mean())
    if name == "rmse":
        return float(np.sqrt((error**2).mean()))
    if name == "bias":
        return float(error.mean())
    if name == "mape":
        denom = actual.abs().replace(0.0, np.nan)
        return float((error.abs() / denom).mean() * 100.0)
    if name == "upr":
        return float((error < 0).mean() * 100.0)
    raise ValueError(  # pragma: no cover - guarded by caller
        f"unknown metric {name!r}; supported: {SUPPORTED_METRICS}."
    )


def score_forecasts(
    forecasts: Mapping[str, pd.Series],
    actual: pd.Series,
    *,
    metrics: tuple[str, ...] = SUPPORTED_METRICS,
) -> pd.DataFrame:
    """Score several forecasts against a shared actual and rank them.

    Each forecast is aligned to ``actual`` on the index intersection and scored
    on the requested ``metrics``.  The result is a tidy table indexed by
    approach name, with one column per metric plus an ``n`` column (overlap
    length), sorted ascending by the first requested metric so the best
    approach is the top row.

    This is **pure**: no logging, no plotting, no mutation.  Use it to compare,
    for example, a four-zone bottom-up sum against a single combined model
    (compute each approach's forecast first, e.g. via ``backtesting_forecaster``).

    Args:
        forecasts: Mapping of approach name to its forecast series.
        actual: The ground-truth series every forecast is scored against.
        metrics: Subset of `SUPPORTED_METRICS` to compute, in output order.
            ``"mae"``, ``"rmse"``, and ``"bias"`` are in the units of the
            series; ``"mape"`` is a percentage; ``"upr"`` is the
            under-prediction rate, the percentage of points with
            ``forecast < actual``. The ranking uses ``metrics[0]``.

    Returns:
        A ``pd.DataFrame`` indexed by approach name with columns
        ``[*metrics, "n"]``, sorted ascending by ``metrics[0]``.

    Raises:
        TypeError: When ``actual`` is not a ``pd.Series`` or a forecast value
            is not a ``pd.Series``.
        ValueError: When ``actual`` is empty, ``forecasts`` is empty, or
            ``metrics`` contains an unsupported name / is empty.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.processing.forecast_scoring import score_forecasts

        idx = pd.date_range("2026-06-13 00:00", periods=24, freq="h", tz="UTC")
        actual = pd.Series([43_858.0] * 24, index=idx)

        forecasts = {
            "combined": actual + 300.0,        # small mixed-ish offset
            "four_zone_sum": actual + 1_780.0,  # flat over-prediction
        }
        table = score_forecasts(forecasts, actual, metrics=("mae", "bias"))
        print(table.round(2).to_string())
        # combined ranks first (lower MAE).
        assert table.index[0] == "combined"
        ```
    """
    if not isinstance(actual, pd.Series):
        raise TypeError(f"actual must be a pd.Series, got {type(actual).__name__!r}.")
    if actual.empty:
        raise ValueError("actual is empty.")
    if not forecasts:
        raise ValueError("forecasts is empty; nothing to score.")
    if not metrics:
        raise ValueError("metrics is empty; request at least one metric.")
    unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            f"unsupported metric(s) {unknown}; supported: {SUPPORTED_METRICS}."
        )

    rows: dict[str, dict[str, float]] = {}
    for name, forecast in forecasts.items():
        if not isinstance(forecast, pd.Series):
            raise TypeError(
                f"forecast {name!r} must be a pd.Series, got "
                f"{type(forecast).__name__!r}."
            )
        common = forecast.index.intersection(actual.index)
        a = actual.loc[common]
        error = forecast.loc[common] - a
        row: dict[str, float] = {m: _compute_metric(m, error, a) for m in metrics}
        row["n"] = float(len(common))
        rows[name] = row

    table = pd.DataFrame.from_dict(rows, orient="index")[list(metrics) + ["n"]]
    table["n"] = table["n"].astype(int)
    return table.sort_values(by=metrics[0], kind="stable")


def score_forecasts_by_period(
    forecasts: Mapping[str, pd.Series],
    actual: pd.Series,
    *,
    freq: str = "D",
    metrics: tuple[str, ...] = SUPPORTED_METRICS,
    min_obs: int | None = None,
    entry_col: str = "entry",
    period_col: str = "period",
) -> pd.DataFrame:
    """Score several forecasts against a shared actual, per calendar period.

    Each forecast is aligned to ``actual`` on the index intersection, the
    aligned pairs are grouped into calendar periods of length ``freq``
    (per day, by default), and every period is scored on the requested
    ``metrics`` with the same metric kernel as `score_forecasts`.  The
    result is one long tidy frame — one row per (entry, period) — ready
    for `aggregate_period_scores` or for paired statistical comparison
    with `spotforecast2_safe.stats.comparison`.

    This is **pure**: no logging, no plotting, no mutation.

    Args:
        forecasts: Mapping of entry name to its forecast series.  Every
            series needs a tz-consistent ``DatetimeIndex`` compatible with
            ``actual``.
        actual: The ground-truth series every forecast is scored against.
            Must have a ``DatetimeIndex``.
        metrics: Subset of `SUPPORTED_METRICS` to compute, in output
            order. Defaults to `SUPPORTED_METRICS`.
        freq: Pandas offset alias defining the period length.  Defaults
            to ``"D"`` (one calendar day per row).
        min_obs: When given, periods with fewer than ``min_obs`` aligned
            observations are dropped (e.g. ``24`` keeps only complete
            days of hourly data).  Defaults to None (keep all periods).
        entry_col: Name of the entry column in the result. Defaults to
            ``"entry"``.
        period_col: Name of the period column in the result. Defaults to
            ``"period"``.

    Returns:
        pd.DataFrame: Long frame with columns
        ``[entry_col, period_col, *metrics, "n"]``, sorted by
        ``(entry, period)``.  ``period`` holds the period-start
        ``pd.Timestamp`` (tz-aware when the inputs are); ``n`` is the
        number of aligned observations inside the period.

    Raises:
        TypeError: When ``actual`` or a forecast is not a ``pd.Series``,
            or ``actual`` has no ``DatetimeIndex``.
        ValueError: When ``actual`` is empty, ``forecasts`` is empty, or
            ``metrics`` contains an unsupported name / is empty.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.processing.forecast_scoring import (
            score_forecasts_by_period,
        )

        idx = pd.date_range("2026-06-10", periods=48, freq="h", tz="UTC")
        actual = pd.Series([40_000.0] * 48, index=idx)
        forecasts = {
            "over": actual + 100.0,   # constant over-prediction
            "under": actual - 200.0,  # constant under-prediction
        }
        daily = score_forecasts_by_period(
            forecasts, actual, metrics=("mae", "bias", "upr"), min_obs=24
        )
        print(daily.round(1).to_string(index=False))
        assert set(daily["entry"]) == {"over", "under"}
        assert (daily.loc[daily["entry"] == "under", "upr"] == 100.0).all()
        ```
    """
    if not isinstance(actual, pd.Series):
        raise TypeError(f"actual must be a pd.Series, got {type(actual).__name__!r}.")
    if not isinstance(actual.index, pd.DatetimeIndex):
        raise TypeError(
            "actual must have a DatetimeIndex, got " f"{type(actual.index).__name__!r}."
        )
    if actual.empty:
        raise ValueError("actual is empty.")
    if not forecasts:
        raise ValueError("forecasts is empty; nothing to score.")
    if not metrics:
        raise ValueError("metrics is empty; request at least one metric.")
    unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            f"unsupported metric(s) {unknown}; supported: {SUPPORTED_METRICS}."
        )

    rows: list[dict[str, object]] = []
    for name in forecasts:
        forecast = forecasts[name]
        if not isinstance(forecast, pd.Series):
            raise TypeError(
                f"forecast {name!r} must be a pd.Series, got "
                f"{type(forecast).__name__!r}."
            )
        common = forecast.index.intersection(actual.index)
        a = actual.loc[common]
        error = forecast.loc[common] - a
        for period, err_group in error.groupby(pd.Grouper(freq=freq)):
            if len(err_group) == 0:
                continue  # Grouper emits empty bins for gaps; never score them
            if min_obs is not None and len(err_group) < min_obs:
                continue
            a_group = a.loc[err_group.index]
            row: dict[str, object] = {entry_col: name, period_col: period}
            row.update({m: _compute_metric(m, err_group, a_group) for m in metrics})
            row["n"] = len(err_group)
            rows.append(row)

    columns = [entry_col, period_col, *metrics, "n"]
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values([entry_col, period_col], kind="stable", ignore_index=True)


def mase_scaling_factors(
    actual: pd.Series,
    periods: Sequence[object] | pd.DatetimeIndex,
    *,
    seasonality: int = 1,
) -> pd.Series:
    """Compute the day-ahead MASE scaling factor for each target period.

    For every entry in ``periods`` the factor is the mean absolute
    ``seasonality``-step difference of ``actual`` over all observations
    **strictly before** that period's start:
    $\\text{mean}\\, |y_t - y_{t-s}|$ for $t < \\text{period}$.
    Dividing a period's MAE by its factor yields the MASE of the
    manuscript-style expanding-history definition — for a single
    ``(y_true, y_pred, y_train)`` triple this equals the denominator
    `spotforecast2_safe.forecaster.metrics.mean_absolute_scaled_error`
    computes internally.

    Args:
        actual: Ground-truth series with a ``DatetimeIndex``, ordered in
            time.
        periods: Period starts to compute factors for.  Entries may be
            ``pd.Timestamp`` values or strings; tz-naive entries are
            localized to the timezone of ``actual``'s index.
        seasonality: Step of the naive difference. Defaults to 1
            (one-step naive).

    Returns:
        pd.Series: Factors indexed by ``periods`` exactly as given
        (strings stay strings), name ``"mase_scaling"``.  A period with
        no preceding observations gets ``NaN``.

    Raises:
        TypeError: When ``actual`` is not a ``pd.Series`` with a
            ``DatetimeIndex``.
        ValueError: When ``actual`` is empty or ``seasonality`` is not a
            positive integer.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.processing.forecast_scoring import (
            mase_scaling_factors,
        )

        idx = pd.date_range("2026-06-01", periods=96, freq="h", tz="UTC")
        ramp = pd.Series(np.arange(96, dtype=float), index=idx)
        factors = mase_scaling_factors(ramp, ["2026-06-03", "2026-06-04"])
        print(factors.to_string())
        # every one-step difference of a ramp is exactly 1.0
        assert (factors == 1.0).all()
        ```
    """
    if not isinstance(actual, pd.Series):
        raise TypeError(f"actual must be a pd.Series, got {type(actual).__name__!r}.")
    if not isinstance(actual.index, pd.DatetimeIndex):
        raise TypeError(
            "actual must have a DatetimeIndex, got " f"{type(actual.index).__name__!r}."
        )
    if actual.empty:
        raise ValueError("actual is empty.")
    if not isinstance(seasonality, int) or seasonality < 1:
        raise ValueError(
            f"seasonality must be a positive integer, got {seasonality!r}."
        )

    diffs = actual.diff(seasonality).abs()
    tz = actual.index.tz
    values: list[float] = []
    for period in periods:
        ts = pd.Timestamp(period)
        if ts.tz is None and tz is not None:
            ts = ts.tz_localize(tz)
        values.append(float(diffs[diffs.index < ts].mean()))
    return pd.Series(values, index=pd.Index(list(periods)), name="mase_scaling")


def aggregate_period_scores(
    period_scores: pd.DataFrame,
    *,
    entry_col: str = "entry",
    period_col: str = "period",
    metrics: Sequence[str] | None = None,
    rank_by: str = "mae",
    ascending: bool = True,
) -> pd.DataFrame:
    """Aggregate per-period scores into a ranked leaderboard table.

    Takes the long frame produced by `score_forecasts_by_period` (or any
    frame with an entry column, a period column, and numeric metric
    columns), averages every metric over each entry's own periods, and
    ranks entries by ``rank_by`` with the number of scored periods
    breaking ties (more periods rank higher).

    Args:
        period_scores: Long frame with one row per (entry, period).
        entry_col: Name of the entry column. Defaults to ``"entry"``.
        period_col: Name of the period column. Defaults to ``"period"``.
        metrics: Metric columns to average.  Defaults to None, which
            selects every numeric column except ``entry_col``,
            ``period_col``, and ``"n"``.
        rank_by: Metric that determines the ranking. Defaults to
            ``"mae"``.
        ascending: Whether lower ``rank_by`` is better. Defaults to True.

    Returns:
        pd.DataFrame: Indexed by entry, columns
        ``[*metrics, "n_periods", "rank"]``, sorted by rank.  ``rank``
        is 1-based.

    Raises:
        ValueError: When ``period_scores`` is empty, a named column is
            missing, or ``rank_by`` is not among ``metrics``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.processing.forecast_scoring import (
            aggregate_period_scores,
            score_forecasts_by_period,
        )

        idx = pd.date_range("2026-06-10", periods=72, freq="h", tz="UTC")
        actual = pd.Series([40_000.0] * 72, index=idx)
        daily = score_forecasts_by_period(
            {"good": actual + 50.0, "bad": actual + 900.0},
            actual,
            metrics=("mae", "bias"),
        )
        board = aggregate_period_scores(daily, rank_by="mae")
        print(board.round(1).to_string())
        assert list(board.index) == ["good", "bad"]
        assert list(board["rank"]) == [1, 2]
        ```
    """
    if period_scores.empty:
        raise ValueError("period_scores is empty; nothing to aggregate.")
    for col in (entry_col, period_col):
        if col not in period_scores.columns:
            raise ValueError(f"column {col!r} not in period_scores.")
    if metrics is None:
        metrics = [
            c
            for c in period_scores.columns
            if c not in (entry_col, period_col, "n")
            and pd.api.types.is_numeric_dtype(period_scores[c])
        ]
    else:
        missing = [m for m in metrics if m not in period_scores.columns]
        if missing:
            raise ValueError(f"metric column(s) {missing} not in period_scores.")
    if rank_by not in metrics:
        raise ValueError(
            f"rank_by {rank_by!r} is not among the metrics {list(metrics)}."
        )

    grouped = period_scores.groupby(entry_col)
    table = grouped[list(metrics)].mean()
    table["n_periods"] = grouped.size()
    table = table.sort_values(
        [rank_by, "n_periods"], ascending=[ascending, False], kind="stable"
    )
    table["rank"] = range(1, len(table) + 1)
    return table

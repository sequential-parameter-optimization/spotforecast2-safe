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
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

#: Metrics ``score_forecasts`` can compute, in canonical order.
SUPPORTED_METRICS: tuple[str, ...] = ("mae", "rmse", "bias", "mape")


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
            series; ``"mape"`` is a percentage. The ranking uses ``metrics[0]``.

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

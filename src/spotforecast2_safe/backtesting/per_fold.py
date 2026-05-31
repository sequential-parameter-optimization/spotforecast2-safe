# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-fold metrics for backtesting predictions.

``backtesting_forecaster`` aggregates its metrics into a single row pooled over
all rolling-origin folds. This module recovers the *per-fold* distribution of
those metrics from the predictions DataFrame, which carries a ``fold`` column.
Metric names are resolved through the same
:func:`spotforecast2_safe.forecaster.metrics._get_metric` path that
``backtesting_forecaster`` uses, so the per-fold columns match the aggregated
ones exactly.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from spotforecast2_safe.forecaster.metrics import _get_metric, add_y_train_argument


def metrics_per_fold(
    predictions: pd.DataFrame,
    y: pd.Series,
    metric: str | Callable | list[str | Callable] = "mean_absolute_error",
) -> pd.DataFrame:
    """Compute per-fold metrics from a backtesting predictions DataFrame.

    ``backtesting_forecaster`` returns ``(metrics, predictions)`` where
    ``metrics`` is a single aggregated row pooled over all folds, and
    ``predictions`` carries a ``fold`` column identifying the rolling-origin
    fold of each prediction. This helper regroups ``predictions`` by ``fold``
    and evaluates the requested metric(s) on each fold, yielding the per-fold
    distribution (one row per fold).

    Args:
        predictions: Second return value of ``backtesting_forecaster``. Must
            contain a ``fold`` column and a ``pred`` column, indexed by the
            prediction timestamps.
        y: The target series passed to ``backtesting_forecaster`` (the ground
            truth). Must cover every timestamp in ``predictions.index``.
        metric: Metric name(s) understood by ``_get_metric`` (e.g.
            ``"mean_absolute_error"``), or callable(s) with the signature
            ``(y_true, y_pred[, y_train])``. A single value or a list.

    Returns:
        A DataFrame indexed by fold id, with one column per metric (named after
        the metric function). Aggregating a column reproduces the order of
        magnitude of the single value returned by ``backtesting_forecaster``.

    Raises:
        ValueError: If ``predictions`` is not a DataFrame, lacks a ``fold`` or
            ``pred`` column, is empty, or its index is not fully covered by
            ``y``.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMRegressor

        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.splitter import TimeSeriesFold
        from spotforecast2_safe.backtesting import (
            backtesting_forecaster,
            metrics_per_fold,
        )

        rng = np.random.default_rng(0)
        idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
        y = pd.Series(
            50 + 10 * np.sin(np.arange(400) / 12) + rng.normal(0, 1, 400),
            index=idx,
            name="load",
        )

        forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(
                n_estimators=20, n_jobs=1, verbose=-1, random_state=0,
                deterministic=True, force_col_wise=True,
            ),
            lags=24,
        )
        forecaster.fit(y=y.iloc[:200])
        cv = TimeSeriesFold(
            steps=24, initial_train_size=200, refit=False, verbose=False
        )
        _, predictions = backtesting_forecaster(
            forecaster=forecaster, y=y, cv=cv, metric="mean_absolute_error",
            show_progress=False, verbose=False,
        )

        per_fold = metrics_per_fold(predictions, y, metric="mean_absolute_error")
        print(f"folds: {len(per_fold)}")
        print(per_fold.columns.tolist())
        ```
    """
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("`predictions` must be a pandas DataFrame.")
    for col in ("fold", "pred"):
        if col not in predictions.columns:
            raise ValueError(
                f"`predictions` must contain a '{col}' column; pass the second "
                "return value of `backtesting_forecaster`."
            )
    if predictions.empty:
        raise ValueError("`predictions` is empty; nothing to evaluate.")
    if not predictions.index.isin(y.index).all():
        raise ValueError("`y` must cover every timestamp in `predictions.index`.")

    metric_list = metric if isinstance(metric, list) else [metric]
    resolved = [
        _get_metric(m) if isinstance(m, str) else add_y_train_argument(m)
        for m in metric_list
    ]

    # History before the backtest window, used by scale-dependent metrics
    # (e.g. MASE); ignored by pointwise metrics such as MAE/MSE.
    y_train = y.loc[y.index < predictions.index.min()]

    rows: dict[int, dict[str, float]] = {}
    for fold_id, group in predictions.groupby("fold"):
        y_true = y.loc[group.index]
        y_pred = group["pred"]
        rows[int(fold_id)] = {
            m.__name__: float(m(y_true=y_true, y_pred=y_pred, y_train=y_train))
            for m in resolved
        }

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "fold"
    return result

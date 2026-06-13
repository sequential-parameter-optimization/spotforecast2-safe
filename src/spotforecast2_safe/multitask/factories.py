# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Forecaster factories for the multitask pipeline.

The default factory builds the LightGBM-backed ``ForecasterRecursive`` that
``BaseTask`` has historically created inline.  Lifting the construction into
a standalone function lets the upcoming ENTSO-E integration (and any future
single-target task) supply its own factory via ``config.forecaster_factory``
without subclassing ``BaseTask``.

A second factory, :func:`quantile_lgbm_forecaster_factory`, builds a *probabilistic
head*: one LightGBM ``ForecasterRecursive`` per quantile (``objective="quantile"``),
giving native quantile regression for calibrated bands. It complements — and is
distinct from — the residual-based ``predict_interval`` / ``predict_quantiles``
already on ``ForecasterRecursive`` (which derive intervals by bootstrapping or
split-conformal from a single point model). :func:`predict_quantile_band` assembles
the per-quantile forecasts into one non-crossing band.
"""

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures as RollingFeaturesUnified

# Default lower/median/upper quantiles for the probabilistic head.
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def default_lgbm_forecaster_factory(
    config: Any,
    *,
    weight_func: Optional[Any] = None,
    target: Optional[str] = None,
) -> ForecasterRecursive:
    """Return a fresh, unfitted LightGBM ``ForecasterRecursive``.

    Mirrors the construction previously inlined in
    ``BaseTask.create_forecaster``.  ``target`` is accepted (and ignored by
    this default) so that custom factories can specialise per target without
    a signature change.

    Args:
        config: Any object satisfying the ``PipelineConfig`` protocol from
            ``spotforecast2_safe.multitask.base``.  Reads ``random_state``,
            ``lags_consider``, ``window_size`` and ``lgbm_n_jobs`` (the
            LightGBM thread count, ``getattr`` default ``1`` for config-like
            objects that predate the field; see ``ConfigMulti.lgbm_n_jobs``).
        weight_func: Optional per-sample weight function produced by the
            imputation step (``apply_imputation``).
        target: Target column name.  Ignored by this default factory; provided
            for the benefit of custom factories that need it.

    Returns:
        A new ``ForecasterRecursive`` ready to be fit.

    Examples:
        ```{python}
        import types
        from spotforecast2_safe.multitask.factories import default_lgbm_forecaster_factory
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

        # Build a minimal config-like object that satisfies the PipelineConfig
        # protocol (random_state, lags_consider, window_size).
        config = types.SimpleNamespace(
            random_state=42,
            lags_consider=[1, 2, 3],
            window_size=3,
        )

        forecaster = default_lgbm_forecaster_factory(config, target="power")
        assert isinstance(forecaster, ForecasterRecursive)
        assert list(forecaster.lags) == [1, 2, 3]
        print(f"type: {type(forecaster).__name__}")
        print(f"lags: {list(forecaster.lags)}")
        ```
    """
    del target  # default factory does not specialise per target
    return ForecasterRecursive(
        estimator=LGBMRegressor(
            random_state=config.random_state,
            verbose=-1,
            n_jobs=getattr(config, "lgbm_n_jobs", 1),
        ),
        lags=config.lags_consider[-1],
        window_features=RollingFeaturesUnified(
            stats=["mean"], window_sizes=config.window_size
        ),
        weight_func=weight_func,
    )


def _validate_quantiles(quantiles: Sequence[float]) -> list:
    """Return *quantiles* as a sorted list of floats in (0, 1), fail-safe."""
    qs = [float(q) for q in quantiles]
    if not qs:
        raise ValueError("quantiles must be non-empty.")
    if any(not (0.0 < q < 1.0) for q in qs):
        raise ValueError(f"quantiles must lie in the open interval (0, 1); got {qs}.")
    if len(set(qs)) != len(qs):
        raise ValueError(f"quantiles must be unique; got {qs}.")
    return sorted(qs)


def quantile_lgbm_forecaster_factory(
    config: Any,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    weight_func: Optional[Any] = None,
    target: Optional[str] = None,
) -> Dict[float, ForecasterRecursive]:
    """Return one quantile-regression LightGBM ``ForecasterRecursive`` per quantile.

    Each forecaster uses ``LGBMRegressor(objective="quantile", alpha=q)`` and the
    same lag/rolling configuration as :func:`default_lgbm_forecaster_factory`, so a
    caller can fit the lower/median/upper heads independently and assemble a band
    with :func:`predict_quantile_band`. Deterministic given ``config.random_state``;
    sf2-safe (LightGBM only, no torch/optuna). Refs ``hong16b``, ``roma19a``.

    Args:
        config: Object satisfying the ``PipelineConfig`` protocol; reads
            ``random_state``, ``lags_consider``, ``window_size`` and
            ``lgbm_n_jobs`` (LightGBM thread count, ``getattr`` default ``1``).
        quantiles: Quantile levels in the open interval ``(0, 1)``. Defaults to
            ``(0.1, 0.5, 0.9)``.
        weight_func: Optional per-sample weight function.
        target: Accepted and ignored (parity with the default factory).

    Returns:
        Dict[float, ForecasterRecursive]: Map from quantile level to a fresh,
        unfitted forecaster, in ascending quantile order.

    Raises:
        ValueError: If *quantiles* is empty, out of ``(0, 1)``, or has duplicates.

    Examples:
        ```{python}
        import types
        from spotforecast2_safe.multitask.factories import (
            quantile_lgbm_forecaster_factory,
        )

        config = types.SimpleNamespace(
            random_state=42, lags_consider=[1, 2, 3], window_size=3
        )
        heads = quantile_lgbm_forecaster_factory(config, quantiles=[0.1, 0.5, 0.9])
        print(sorted(heads))
        print(heads[0.1].regressor.get_params()["objective"])
        print(heads[0.1].regressor.get_params()["alpha"])
        ```
    """
    del target  # parity with default factory; not specialised per target
    qs = _validate_quantiles(quantiles)
    return {
        q: ForecasterRecursive(
            estimator=LGBMRegressor(
                objective="quantile",
                alpha=q,
                random_state=config.random_state,
                verbose=-1,
                n_jobs=getattr(config, "lgbm_n_jobs", 1),
            ),
            lags=config.lags_consider[-1],
            window_features=RollingFeaturesUnified(
                stats=["mean"], window_sizes=config.window_size
            ),
            weight_func=weight_func,
        )
        for q in qs
    }


def predict_quantile_band(
    forecasters: Mapping[float, ForecasterRecursive],
    steps: int,
    *,
    last_window: Optional[Any] = None,
    exog: Optional[Any] = None,
    enforce_monotonic: bool = True,
) -> pd.DataFrame:
    """Assemble per-quantile forecasts into one non-crossing band.

    Calls ``predict`` on each fitted forecaster from
    :func:`quantile_lgbm_forecaster_factory` and stacks the results into columns
    named ``q_<level>`` (e.g. ``q_0.1``), in ascending quantile order. Because the
    quantile heads are fitted independently they can *cross* (a higher quantile
    predicting below a lower one); with ``enforce_monotonic`` the rows are sorted
    ascending (the Chernozhukov rearrangement), a deterministic post-hoc fix that
    restores monotonicity without changing the marginal quantile levels.

    Args:
        forecasters: Map from quantile level to a **fitted** ``ForecasterRecursive``
            (as returned by :func:`quantile_lgbm_forecaster_factory`, after fit).
        steps: Forecast horizon passed to each ``predict``.
        last_window: Optional last-window override forwarded to ``predict``.
        exog: Optional exogenous frame forwarded to ``predict``.
        enforce_monotonic: When ``True`` (default), sort each row ascending so the
            band never crosses.

    Returns:
        pd.DataFrame: One column per quantile (``q_<level>``), indexed by the
        forecast horizon.

    Raises:
        ValueError: If *forecasters* is empty or its levels are invalid.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        import types
        from spotforecast2_safe.multitask.factories import (
            quantile_lgbm_forecaster_factory,
            predict_quantile_band,
        )

        idx = pd.date_range("2023-01-01", periods=300, freq="h")
        y = pd.Series(
            50 + 10 * np.sin(np.arange(300) * 2 * np.pi / 24), index=idx, name="y"
        )
        config = types.SimpleNamespace(
            random_state=0, lags_consider=[1, 24], window_size=24
        )
        heads = quantile_lgbm_forecaster_factory(config, quantiles=[0.1, 0.5, 0.9])
        for fc in heads.values():
            fc.fit(y=y)
        band = predict_quantile_band(heads, steps=6)
        print(band.columns.tolist())
        # Non-crossing after rearrangement.
        assert (band["q_0.1"] <= band["q_0.5"] + 1e-9).all()
        assert (band["q_0.5"] <= band["q_0.9"] + 1e-9).all()
        ```
    """
    qs = _validate_quantiles(list(forecasters.keys()))
    columns = [f"q_{q}" for q in qs]
    preds = []
    index = None
    for q in qs:
        pred = forecasters[q].predict(steps=steps, last_window=last_window, exog=exog)
        if index is None:
            index = pred.index
        preds.append(np.asarray(pred, dtype="float64"))

    values = np.column_stack(preds)
    if enforce_monotonic:
        values = np.sort(values, axis=1)
    return pd.DataFrame(values, index=index, columns=columns)

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Forecaster factories for the multitask pipeline.

The default factory builds the LightGBM-backed ``ForecasterRecursive`` that
``BaseTask`` has historically created inline.  Lifting the construction into
a standalone function lets the upcoming ENTSO-E integration (and any future
single-target task) supply its own factory via ``config.forecaster_factory``
without subclassing ``BaseTask``.
"""

from typing import Any, Optional

from lightgbm import LGBMRegressor

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures as RollingFeaturesUnified


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
            ``lags_consider``, and ``window_size``.
        weight_func: Optional per-sample weight function produced by the
            imputation step (``apply_imputation``).
        target: Target column name.  Ignored by this default factory; provided
            for the benefit of custom factories that need it.

    Returns:
        A new ``ForecasterRecursive`` ready to be fit.
    """
    del target  # default factory does not specialise per target
    return ForecasterRecursive(
        estimator=LGBMRegressor(random_state=config.random_state, verbose=-1),
        lags=config.lags_consider[-1],
        window_features=RollingFeaturesUnified(
            stats=["mean"], window_sizes=config.window_size
        ),
        weight_func=weight_func,
    )

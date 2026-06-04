# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Training strategies for the multitask pipeline.

Each strategy encapsulates the per-target "prepare a forecaster for the
final fit" step that the existing ``execute_lazy`` / ``execute_defaults``
functions perform between ``create_forecaster()`` and
``_train_and_predict_target()``.

The protocol is introduced here as scaffolding for the ENTSO-E integration.
Two concrete strategies are available in this safe subset:

- `LazyStrategy` — Approach 1.  Optionally applies cached tuning results;
  otherwise leaves the forecaster at default parameters.
- `DefaultsStrategy` — Approach 2.  Explicit "train with defaults, no tuning,
  no cached params."  Returns the forecaster unchanged.

Auto-tuning strategies (OptunaStrategy, SpotOptimStrategy) are available
only in the ``spotforecast2`` sibling package, which has the required
``optuna`` and ``spotoptim`` dependencies.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import pandas as pd


class TrainingStrategy(Protocol):
    """Strategy interface for preparing a forecaster before the final fit.

    Implementations return a forecaster with any tuning/parameter changes
    applied.  The final ``forecaster.fit(...)`` and prediction packaging are
    performed by ``BaseTask._train_and_predict_target`` after this call.
    """

    name: str

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Return a forecaster ready for the final fit step."""


class LazyStrategy:
    """Approach 1 — Lazy fitting with optional cached tuning.

    Mirrors the body of ``execute_lazy`` between ``create_forecaster()`` and
    ``_train_and_predict_target()``.
    """

    name = "lazy"

    def __init__(
        self,
        use_tuned_params: bool = True,
        max_age_days: Optional[float] = None,
    ) -> None:
        self.use_tuned_params = use_tuned_params
        self.max_age_days = max_age_days

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        del y_train, exog_train  # unused by this strategy
        if not self.use_tuned_params:
            return forecaster
        tuned = task.load_tuning_results(target=target, max_age_days=self.max_age_days)
        if tuned is None:
            return forecaster
        task.logger.info(
            "  Applying cached %s tuning results (from %s).",
            tuned["task_name"],
            tuned["timestamp"],
        )
        forecaster.set_params(**tuned["best_params"])
        if hasattr(forecaster, "set_lags"):
            forecaster.set_lags(tuned["best_lags"])
        return forecaster


class DefaultsStrategy:
    """Approach 2 — Train with defaults, no tuning, no cached params.

    The simplest possible training strategy: leave the forecaster at the
    parameters produced by the factory and hand it back to
    ``_train_and_predict_target`` for the explicit fit.  Use this when the
    caller wants a deterministic baseline that does not benefit from any
    cached tuning results — useful for ENTSO-E "Approach 2: Training without
    Tuning" and for regression benchmarking.

    Functionally equivalent to ``LazyStrategy(use_tuned_params=False)``;
    kept as a distinct class so the ``task="defaults"`` routing reads
    intent at the call site (no implicit cache lookup).
    """

    name = "defaults"

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        del task, target, y_train, exog_train  # no preparation needed
        return forecaster

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

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.multitask.strategies import (
            TrainingStrategy,
            LazyStrategy,
            DefaultsStrategy,
        )

        # Both concrete strategies satisfy the TrainingStrategy protocol:
        # they expose a `name` attribute and a `prepare_forecaster` method.
        for cls in (LazyStrategy, DefaultsStrategy):
            s = cls()
            assert hasattr(s, "name"), f"{cls.__name__} missing .name"
            assert callable(s.prepare_forecaster), f"{cls.__name__} missing .prepare_forecaster"
            print(f"{cls.__name__}.name = {s.name!r}")
        ```
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
        """Return a forecaster ready for the final fit step.

        Examples:
            ```{python}
            import pandas as pd
            from sklearn.linear_model import LinearRegression
            from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            from spotforecast2_safe.multitask.strategies import DefaultsStrategy

            # Demonstrate prepare_forecaster via a concrete implementation.
            # DefaultsStrategy is the simplest: it returns the forecaster unchanged.
            forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            y_train = pd.Series(range(30), dtype=float, name="target_0")

            class _NullTask:
                pass

            strategy = DefaultsStrategy()
            result = strategy.prepare_forecaster(_NullTask(), "target_0", forecaster, y_train)
            assert result is forecaster
            print(f"prepare_forecaster returned the same object: {result is forecaster}")
            ```
        """


class LazyStrategy:
    """Approach 1 — Lazy fitting with optional cached tuning.

    Mirrors the body of ``execute_lazy`` between ``create_forecaster()`` and
    ``_train_and_predict_target()``.

    Examples:
        ```{python}
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.multitask.strategies import LazyStrategy

        # Construct with default settings (cache lookup enabled).
        strategy = LazyStrategy(use_tuned_params=True, max_age_days=30.0)
        assert strategy.name == "lazy"
        assert strategy.use_tuned_params is True
        assert strategy.max_age_days == 30.0
        print(f"strategy.name={strategy.name!r}, use_tuned_params={strategy.use_tuned_params}")

        # When no cached results exist, the forecaster is returned unchanged.
        class _NullTask:
            class logger:
                @staticmethod
                def info(*a, **kw):
                    pass
            def load_tuning_results(self, target, max_age_days=None):
                return None

        forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
        y_train = pd.Series(range(30), dtype=float, name="target_0")
        result = strategy.prepare_forecaster(_NullTask(), "target_0", forecaster, y_train)
        assert result is forecaster
        print(f"Forecaster returned unchanged when cache miss: {result is forecaster}")
        ```
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
        """Optionally apply cached tuning results and return the forecaster.

        When ``use_tuned_params`` is ``False``, or when no cached results are
        found for ``target``, the forecaster is returned without modification.
        Otherwise, ``best_params`` are applied via ``forecaster.set_params``
        and ``best_lags`` via ``forecaster.set_lags`` (when available).

        Args:
            task: A ``BaseTask`` instance that exposes ``load_tuning_results``
                and a ``logger``.
            target: The target name used as key when loading cached results.
            forecaster: The unfitted forecaster returned by the factory.
            y_train: Training series (unused by this strategy; kept for
                protocol compatibility).
            exog_train: Exogenous training frame (unused by this strategy).

        Returns:
            The same forecaster object, with parameters updated in-place when
            cached tuning results were found and applied.

        Examples:
            ```{python}
            import pandas as pd
            from sklearn.linear_model import LinearRegression
            from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            from spotforecast2_safe.multitask.strategies import LazyStrategy

            forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=4)
            y_train = pd.Series(range(30), dtype=float, name="target_0")

            # --- Path 1: use_tuned_params=False — forecaster returned as-is ---
            strategy_no_cache = LazyStrategy(use_tuned_params=False)

            class _NullTask:
                class logger:
                    @staticmethod
                    def info(*a, **kw):
                        pass
                def load_tuning_results(self, target, max_age_days=None):
                    return None

            result = strategy_no_cache.prepare_forecaster(
                _NullTask(), "target_0", forecaster, y_train
            )
            assert result is forecaster
            print(f"Path 1 (no cache): lags unchanged = {list(result.lags)}")

            # --- Path 2: cached results present — lags and params applied ---
            class _CachedTask:
                class logger:
                    @staticmethod
                    def info(*a, **kw):
                        pass
                def load_tuning_results(self, target, max_age_days=None):
                    return {
                        "task_name": "lazy",
                        "timestamp": "2026-01-01T00:00:00",
                        "best_params": {},
                        "best_lags": [1, 2, 3],
                    }

            strategy_with_cache = LazyStrategy(use_tuned_params=True)
            result2 = strategy_with_cache.prepare_forecaster(
                _CachedTask(), "target_0", forecaster, y_train
            )
            assert list(result2.lags) == [1, 2, 3]
            print(f"Path 2 (cache hit): lags updated to {list(result2.lags)}")
            ```
        """
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

    Examples:
        ```{python}
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.multitask.strategies import DefaultsStrategy

        strategy = DefaultsStrategy()
        assert strategy.name == "defaults"
        print(f"strategy.name={strategy.name!r}")

        forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
        y_train = pd.Series(range(30), dtype=float, name="target_0")

        class _NullTask:
            pass

        result = strategy.prepare_forecaster(_NullTask(), "target_0", forecaster, y_train)
        assert result is forecaster
        assert list(result.lags) == [1, 2, 3, 4, 5], f"Unexpected lags: {list(result.lags)}"
        print(f"Forecaster returned unchanged: lags={list(result.lags)}")
        ```
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
        """Return the forecaster unchanged — no tuning, no cached params.

        Args:
            task: Ignored; accepted for protocol compatibility.
            target: Ignored; accepted for protocol compatibility.
            forecaster: The unfitted forecaster returned by the factory.
            y_train: Ignored; accepted for protocol compatibility.
            exog_train: Ignored; accepted for protocol compatibility.

        Returns:
            The same forecaster object, unmodified.

        Examples:
            ```{python}
            import pandas as pd
            from sklearn.linear_model import LinearRegression
            from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            from spotforecast2_safe.multitask.strategies import DefaultsStrategy

            forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            y_train = pd.Series(range(30), dtype=float, name="target_0")

            class _NullTask:
                pass

            strategy = DefaultsStrategy()
            result = strategy.prepare_forecaster(_NullTask(), "target_0", forecaster, y_train)
            assert result is forecaster
            print(f"Returned same forecaster object: {result is forecaster}")
            print(f"Lags unchanged: {list(result.lags)}")
            ```
        """
        del task, target, y_train, exog_train  # no preparation needed
        return forecaster

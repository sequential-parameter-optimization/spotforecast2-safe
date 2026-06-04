# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lazy-fitting task — Task 1.

Fits each target with default LightGBM parameters (no tuning).
When cached tuning results are available (from a prior tuning run in the
``spotforecast2`` sibling package), they are loaded and applied automatically
so that the lazy task benefits from prior tuning without re-running the search.
"""

from typing import Any, Dict, Optional

from spotforecast2_safe.multitask.base import BaseTask
from spotforecast2_safe.multitask.strategies import LazyStrategy


def execute_lazy(
    task: BaseTask,
    show: bool = False,
    use_tuned_params: bool = True,
    max_age_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute lazy fitting for all targets on ``task``.

    Thin wrapper around ``BaseTask._run_strategy`` using ``LazyStrategy``.
    When ``use_tuned_params`` is ``True`` (the default), previously saved
    tuning results are loaded from cache and applied to the forecaster.
    If no cached results are found the forecaster uses default parameters.

    Args:
        task: A ``BaseTask`` (or subclass) instance with prepared data.
        show: If ``True``, invoke the visualisation hooks.
        use_tuned_params: If ``True``, attempt to load cached tuning
            results (best parameters and lags) for each target.
        max_age_days: Maximum age in days for cached tuning results.
            ``None`` accepts any age.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["lazy"]``.
        When ``task.config.auto_save_models`` is ``True`` (the default), fitted
        models are saved to disk so ``PredictTask`` can load them directly.
    """
    strategy = LazyStrategy(
        use_tuned_params=use_tuned_params, max_age_days=max_age_days
    )
    return task._run_strategy(
        strategy,
        task_name="task 1: Lazy Fitting",
        results_key="lazy",
        show=show,
        log_prefix="[task 1] ",
    )


class LazyTask(BaseTask):
    """Task 1 — Lazy Fitting with default LightGBM parameters.

    Creates an unfitted forecaster per target and fits with default
    hyperparameters.  No cross-validation or tuning is performed.

    When cached tuning results are available (saved by a prior run in the
    ``spotforecast2`` sibling package), they are loaded and applied automatically
    so that the lazy task benefits from prior tuning without re-running
    the search.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        from spotforecast2_safe.multitask import LazyTask
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(data_frame_name="demo10", predict_size=24, cache_home=Path(tmp))
            task = LazyTask(cfg)
            print(f"Task: {task.TASK}")
            print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "lazy"

    def run(
        self,
        show: bool = False,
        use_tuned_params: bool = True,
        max_age_days: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run lazy fitting for all targets.

        Args:
            show: If ``True``, invoke the visualisation hooks.
            use_tuned_params: If ``True``, load and apply cached tuning
                results for each target.
            max_age_days: Maximum age in days for cached tuning results.
                ``None`` accepts any age.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["lazy"]``.
        """
        return execute_lazy(
            self,
            show=show,
            use_tuned_params=use_tuned_params,
            max_age_days=max_age_days,
        )

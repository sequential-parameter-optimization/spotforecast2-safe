# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backward-compatible MultiTask dispatcher.

``MultiTask`` preserves the original API where a single ``task``
parameter selects which pipeline mode to run.  It inherits from
``BaseTask`` and delegates ``run()`` to the appropriate task-specific
function.

Available tasks in ``spotforecast2-safe``: ``"lazy"``, ``"defaults"``,
``"predict"``, ``"clean"``.  Tasks requiring auto-tuning
(``"optuna"``, ``"spotoptim"``) are not available in this package.
Use the ``spotforecast2`` sibling package for those.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.multitask.base import BaseTask, PipelineConfig
from spotforecast2_safe.multitask.clean import execute_clean
from spotforecast2_safe.multitask.defaults import execute_defaults
from spotforecast2_safe.multitask.lazy import execute_lazy
from spotforecast2_safe.multitask.predict import execute_predict


class MultiTask(BaseTask):
    """Orchestrates a multi-target time-series forecasting pipeline.

    Data must be provided as a pandas DataFrame via ``dataframe``.
    A test dataset can optionally be provided via ``data_test``.

    The typical usage flow is:

    1. Instantiate with ``config`` (or omit to auto-construct ``ConfigMulti()``).
    2. Call ``prepare_data`` to load, resample, and validate data.
    3. Call ``detect_outliers`` to apply hard bounds and IsolationForest.
    4. Call ``impute`` to fill gaps.
    5. Call ``build_exogenous_features`` to construct weather / calendar /
       day-night / holiday covariates.
    6. Call ``run`` (or individual ``run_task_*`` methods) to train,
       predict, and aggregate.

    Available tasks: ``"lazy"``, ``"defaults"``, ``"predict"``, ``"clean"``.
    Tasks requiring auto-tuning (``"optuna"``, ``"spotoptim"``) raise
    ``ValueError`` — use the ``spotforecast2`` sibling package for those.

    Args:
        config: A ``PipelineConfig``-conforming object (e.g. ``ConfigMulti``).
            When ``None``, a fresh ``ConfigMulti()`` is constructed.
        task: Pipeline task mode — ``"lazy"``, ``"defaults"``,
            ``"predict"``, or ``"clean"``.  Defaults to ``"lazy"``.
        dataframe: Pre-loaded input DataFrame with training data.  The
            DataFrame must contain a datetime column matching
            ``config.index_name`` plus at least one numeric target column.
            Optional for the ``"clean"`` task, required for all others.
        data_test: Pre-loaded input DataFrame with test data.  Optional.
        cache_home: Cache directory override.  When not ``None``, replaces
            ``config.cache_home`` for this task instance.
        dry_run: If ``True``, do not clean cache or save models.
        show_progress: Whether to print progress messages during pipeline
            execution.
        log_level: Logging level for the pipeline logger.
        **overrides: Forwarded to ``config.set_params(**overrides)`` — a
            convenience for one-line tweaks without building a fresh config.
            Mutates the caller's config object.

    Examples:
        ```{python}
        import tempfile
        import pandas as pd
        import numpy as np
        from spotforecast2_safe.multitask import MultiTask
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
        df = pd.DataFrame({"a": rng.normal(100, 10, len(idx))}, index=idx)
        df.index.name = "DateTime"

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(predict_size=6, use_exogenous_features=False, cache_home=tmp)
            mt = MultiTask(cfg, dataframe=df)
            print(f"DataFrame stored: {mt._dataframe is not None}")
            print(f"Task: {mt.TASK}")
        ```
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        task: str = "lazy",
        dataframe: Optional[pd.DataFrame] = None,
        data_test: Optional[pd.DataFrame] = None,
        cache_home: Optional[Path] = None,
        dry_run: bool = False,
        show_progress: bool = False,
        log_level: int = logging.INFO,
        **overrides: Any,
    ) -> None:
        # Set _task_name before super().__init__ so self.TASK is correct
        self._task_name = task
        self._dry_run = dry_run
        self._show_progress = show_progress
        if config is None:
            config = ConfigMulti()
        super().__init__(
            config,
            dataframe=dataframe,
            data_test=data_test,
            cache_home=cache_home,
            log_level=log_level,
            **overrides,
        )

    # ------------------------------------------------------------------
    # Task-specific convenience methods
    # ------------------------------------------------------------------

    def run_task_lazy(self, show: bool = False) -> Dict[str, Any]:
        """Lazy Fitting with default LightGBM parameters.

        Args:
            show: If ``True``, invoke the visualisation hooks.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["lazy"]``.
        """
        return execute_lazy(self, show=show)

    def run_task_defaults(self, show: bool = False) -> Dict[str, Any]:
        """Defaults fitting — no tuning, no cached params.

        Distinct from ``run_task_lazy`` only in that it never consults the
        tuning-result cache.  Use this for deterministic baselines and for
        ENTSO-E "Approach 2: Training without Tuning".

        Args:
            show: If ``True``, invoke the visualisation hooks.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["defaults"]``.
        """
        return execute_defaults(self, show=show)

    def run_task_predict(
        self,
        show: bool = False,
        task_name: Optional[str] = None,
        max_age_days: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict-only using previously saved models.

        Loads fitted models from the cache directory and produces
        predictions without any training.  Raises ``RuntimeError``
        if no saved models are found.

        Args:
            show: If ``True``, invoke the visualisation hooks.
            task_name: Restrict model loading to a specific source task
                (``"lazy"``, ``"defaults"``, ``"optuna"``, or
                ``"spotoptim"``).  ``None`` loads the most recent model
                regardless of source.
            max_age_days: Maximum age in days for saved models.
                ``None`` accepts any age.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["predict"]``.

        Raises:
            RuntimeError: If no saved models are found.
        """
        return execute_predict(
            self, show=show, task_name=task_name, max_age_days=max_age_days
        )

    def run_task_clean(
        self,
        show: bool = False,
        dry_run: bool = False,
        cache_home: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Remove all cached data from the pipeline cache directory.

        Does not require ``prepare_data()`` to be called first.

        Args:
            show: Accepted for API consistency.  Not used by the clean task.
            dry_run: If ``True``, report what would be deleted without
                actually removing anything.
            cache_home: Override the directory to clean.  ``None`` uses
                the cache directory configured on this instance.

        Returns:
            Dict with keys status, cache_dir, and deleted_items.

        Raises:
            RuntimeError: If the cache directory cannot be removed.
        """
        return execute_clean(self, cache_home=cache_home, dry_run=dry_run)

    # ------------------------------------------------------------------
    # Run dispatcher
    # ------------------------------------------------------------------

    def run(
        self,
        task: Optional[str] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the task specified by ``task`` (or ``self.TASK``).

        This dispatcher selects a task only; per-task options (for example
        ``use_tuned_params`` or ``max_age_days``) must be passed to the
        corresponding ``run_task_*`` method directly.

        Args:
            task: Override the task mode.  ``None`` uses ``self.TASK``.
            show: If ``True``, invoke the visualisation hooks.

        Returns:
            Aggregated prediction package. Per-target results are stored
            on ``self.results[<task_key>]``.

        Raises:
            TypeError: If unexpected keyword arguments are supplied
                (fail-safe: they would otherwise be silently ignored).
            ValueError: If ``task`` is ``"optuna"`` or ``"spotoptim"``
                (auto-tuning not available in this package), or if
                ``task`` is not one of the supported task names.
            RuntimeError: If ``prepare_data`` has not been called
                (for training and prediction tasks).
        """
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments {sorted(kwargs)}. "
                "MultiTask.run() selects a task only; pass per-task options "
                "to the run_task_*() methods directly."
            )
        task = task or self.TASK
        if task in ("optuna", "spotoptim"):
            raise ValueError(
                f"Task {task!r} requires auto-tuning, which is not available in "
                "spotforecast2-safe. Use the spotforecast2 package, or "
                "task='lazy'/'defaults'."
            )
        dispatch = {
            "lazy": self.run_task_lazy,
            "defaults": self.run_task_defaults,
            "predict": self.run_task_predict,
        }
        if task not in {*dispatch, "clean"}:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: "
                f"{sorted({*dispatch, 'clean'})}"
            )
        if task == "clean":
            return self.run_task_clean(show=show, dry_run=self._dry_run)
        return dispatch[task](show=show)

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Defaults task — Task 2.

Fits each target with the factory's default hyperparameters.  Distinct from
``LazyTask`` in one respect only: ``DefaultsTask`` never consults the
tuning-result cache.  Use it when you want a deterministic baseline that is
guaranteed not to inherit any prior tuning run.

ENTSO-E "Approach 2: Training without Tuning" routes here.
"""

from typing import Any, Dict

from spotforecast2_safe.multitask.base import BaseTask
from spotforecast2_safe.multitask.strategies import DefaultsStrategy


def execute_defaults(
    task: BaseTask,
    show: bool = False,
) -> Dict[str, Any]:
    """Execute defaults fitting for all targets on ``task``.

    Thin wrapper around ``BaseTask._run_strategy`` using ``DefaultsStrategy``.

    Args:
        task: A ``BaseTask`` (or subclass) instance with prepared data.
        show: If ``True``, invoke the visualisation hooks.

    Returns:
        Aggregated prediction package (weighted combination of all targets,
        or the single-target package when ``len(config.targets) == 1``).
        Per-target packages are stored on ``task.results["defaults"]``.
        When ``task.config.auto_save_models`` is ``True`` (the default), fitted
        models are saved to disk so ``PredictTask(task_name="defaults")`` can
        load them directly.

    Examples:
        ```{python}
        import tempfile
        import warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from spotforecast2_safe.multitask.defaults import DefaultsTask, execute_defaults
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
        df = pd.DataFrame({"load": rng.normal(100, 10, len(idx))}, index=idx)
        df.index.name = "DateTime"

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(
                predict_size=6,
                use_exogenous_features=False,
                use_outlier_detection=False,
                auto_save_models=False,
                number_folds=2,
                cache_home=Path(tmp),
                verbose=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                task = DefaultsTask(cfg, dataframe=df)
                task.prepare_data().detect_outliers().impute().build_exogenous_features()
                result = execute_defaults(task)

        print(f"Future predictions: {len(result['future_pred'])} steps")
        assert isinstance(result["future_pred"], pd.Series)
        assert len(result["future_pred"]) == 6
        ```
    """
    return task._run_strategy(
        DefaultsStrategy(),
        task_name="task 2: Defaults",
        results_key="defaults",
        show=show,
        log_prefix="[task 2] ",
    )


class DefaultsTask(BaseTask):
    """Task 2 — Defaults fitting (no tuning, no cached params).

    Creates an unfitted forecaster per target via ``config.forecaster_factory``
    (or the package default) and fits with whatever parameters that factory
    chooses.  Unlike ``LazyTask``, never reads the tuning-result cache.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        from spotforecast2_safe.multitask import DefaultsTask
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(data_frame_name="demo10", predict_size=24, cache_home=Path(tmp))
            task = DefaultsTask(cfg)
            print(f"Task: {task.TASK}")
            print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "defaults"

    def run(
        self,
        show: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run defaults fitting for all targets.

        Args:
            show: If ``True``, invoke the visualisation hooks.
            **kwargs: Forwarded for compatibility with ``BaseTask.run``;
                ``DefaultsTask`` does not consume any extra parameters.

        Returns:
            Aggregated prediction package.  Per-target packages are stored
            on ``self.results["defaults"]``.

        Examples:
            ```{python}
            import tempfile
            import warnings
            import numpy as np
            import pandas as pd
            from pathlib import Path
            from spotforecast2_safe.multitask.defaults import DefaultsTask
            from spotforecast2_safe.configurator.config_multi import ConfigMulti

            rng = np.random.default_rng(0)
            idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
            df = pd.DataFrame({"load": rng.normal(100, 10, len(idx))}, index=idx)
            df.index.name = "DateTime"

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(
                    predict_size=6,
                    use_exogenous_features=False,
                    use_outlier_detection=False,
                    auto_save_models=False,
                    number_folds=2,
                    cache_home=Path(tmp),
                    verbose=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    task = DefaultsTask(cfg, dataframe=df)
                    task.prepare_data().detect_outliers().impute().build_exogenous_features()
                    result = task.run()

            print(f"Future predictions: {len(result['future_pred'])} steps")
            assert "defaults" in task.results
            assert isinstance(result["future_pred"], pd.Series)
            ```
        """
        del kwargs  # DefaultsTask has no tuning- or cache-related parameters
        return execute_defaults(self, show=show)

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convenience runner for the MultiTask forecasting pipeline.

Provides a single ``run`` function that wraps the full pipeline
sequence (prepare_data, detect_outliers, impute, build_exogenous_features,
run) behind a one-call interface.

Available tasks: ``"lazy"``, ``"defaults"``, ``"predict"``, ``"clean"``.
Auto-tuning tasks (``"optuna"``, ``"spotoptim"``) are not available in
``spotforecast2-safe``; use the ``spotforecast2`` sibling package for those.
"""

from typing import Any, List, Optional, Tuple

import pandas as pd

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.data.fetch_data import get_cache_home
from spotforecast2_safe.multitask.base import PipelineConfig
from spotforecast2_safe.multitask.multi import MultiTask

# Demo-dataset presets for ``run()``.
#
# These lists are calibrated for the 11-target ENTSO-E demo dataset
# (``demo10.csv``) used in the package examples and the Quarto docs.  They are
# *not* general-purpose defaults: a different dataset needs its own bounds and
# aggregation weights, supplied explicitly to ``run(bounds=..., agg_weights=...)``
# or carried by a config preset.  Kept here (not on ``ConfigMulti``) so that
# direct ``ConfigMulti()`` users don't silently inherit demo10-specific values.
_DEMO10_BOUNDS: List[Tuple[float, float]] = [
    (-2500, 4500),
    (-10, 3000),
    (0, 230),
    (0, 550),
    (0, 1400),
    (0, 1400),
    (0, 10),
    (0, 4500),
    (0, 300),
    (0, 400),
    (0, 300),
]

_DEMO10_AGG_WEIGHTS: List[float] = [
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
]

_PIPELINE_TASKS = frozenset({"lazy", "defaults", "predict"})
_ALL_TASKS = _PIPELINE_TASKS | {"clean"}


def make_demo10_config(**overrides: Any) -> ConfigMulti:
    """Build a ``ConfigMulti`` pre-loaded with the 11-target demo10 presets.

    The 11-target ENTSO-E demo dataset (``demo10.csv``) ships with the
    package and is the canonical example dataset for the Quarto docs and
    tutorial notebooks.  Its outlier ``bounds`` and aggregation
    ``agg_weights`` are dataset-specific and should not be inherited by
    arbitrary callers — this helper makes the opt-in explicit.

    Args:
        **overrides: Forwarded to ``ConfigMulti.set_params`` to tweak any
            other field on the returned config.

    Returns:
        A new ``ConfigMulti`` with ``bounds`` and ``agg_weights`` set to the
        demo10 presets, plus any caller-supplied overrides applied.

    Examples:
        ```{python}
        from spotforecast2_safe.multitask.runner import make_demo10_config

        cfg = make_demo10_config(predict_size=48)
        print(cfg.predict_size)
        print(len(cfg.bounds))
        print(len(cfg.agg_weights))
        ```
    """
    cfg = ConfigMulti(bounds=_DEMO10_BOUNDS, agg_weights=_DEMO10_AGG_WEIGHTS)
    if overrides:
        cfg.set_params(**overrides)
    return cfg


def run(
    config: Optional[PipelineConfig] = None,
    *,
    task: str = "lazy",
    dataframe: Optional[pd.DataFrame] = None,
    data_test: Optional[pd.DataFrame] = None,
    project_name: str = "test_project",
    cache_home: Optional[str] = None,
    plot_with_outliers: bool = False,
    show: bool = False,
    show_progress: bool = False,
    dry_run: bool = False,
    log_level: int = 40,
    **overrides: Any,
) -> pd.DataFrame:
    """Run the MultiTask forecasting pipeline and return predictions.

    Wraps the standard pipeline sequence into a single call.  For the
    ``"clean"`` task only the cache directory is wiped and an empty
    DataFrame is returned.  For all other tasks the full sequence

        prepare_data -> detect_outliers -> impute ->
        build_exogenous_features -> run

    is executed and the aggregated future predictions are returned as a
    DataFrame.

    Available tasks: ``"lazy"``, ``"defaults"``, ``"predict"``,
    ``"clean"``.  Passing ``task="optuna"`` or ``task="spotoptim"``
    raises ``ValueError``; use the ``spotforecast2`` sibling package for
    auto-tuning.

    When ``plot_with_outliers=True`` the call reaches ``plot_with_outliers()``
    on the ``BaseTask`` instance, which raises ``NotImplementedError`` because
    plotting is not available in ``spotforecast2-safe``.

    Args:
        config: A ``PipelineConfig``-conforming object (typically
            ``ConfigMulti``).  When ``None``, a fresh ``ConfigMulti()``
            is constructed with default fields.  Use ``make_demo10_config()``
            to opt in to the 11-target demo10 ``bounds`` / ``agg_weights``
            presets.
        task: Pipeline mode — one of ``"lazy"``, ``"defaults"``,
            ``"predict"``, or ``"clean"``.  Defaults to ``"lazy"``.
        dataframe: Input time-series data.  Must contain a datetime
            column matching ``config.index_name`` and at least one numeric
            target column.  Optional for ``"clean"``, required otherwise.
        data_test: Ground-truth DataFrame covering the prediction horizon.
            When supplied, populates ``test_actual`` and ``metrics_future``
            in the prediction package.  Optional.
        project_name: Active-dataset identifier.  Sets
            ``config.data_frame_name``, which drives cache-subdirectory
            and model-file naming.
        cache_home: Cache directory override.  When ``None``, the package
            default from ``get_cache_home()`` is used.
        plot_with_outliers: Whether to render the optional
            outlier-visualisation step.  Calling this will raise
            ``NotImplementedError`` in ``spotforecast2-safe``; set to
            ``False`` (the default) to skip it.
        show: Whether to invoke the prediction display hooks after the
            task runs.
        show_progress: Whether to print progress messages during pipeline
            execution.
        dry_run: Forwarded to ``MultiTask``; only meaningful for the
            ``"clean"`` task.
        log_level: Logging level.  Defaults to 40 (ERROR).
        **overrides: Forwarded to ``config.set_params(**overrides)``.
            Mutates the caller's config object.

    Returns:
        DataFrame whose index is the forecast horizon timestamps and
        whose single column ``"forecast"`` contains the aggregated
        predicted values.  For the ``"clean"`` task an empty DataFrame is
        returned.

    Raises:
        ValueError: If ``task`` is not one of the supported task names.

    Examples:
        Run the pipeline using cached or default model parameters
        (``"lazy"`` task):

        ```{python}
        import pandas as pd
        import warnings
        warnings.filterwarnings("ignore")
        from spotforecast2_safe.multitask.runner import run
        from spotforecast2_safe.configurator.config_multi import ConfigMulti
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home

        data_home = get_package_data_home()
        df = fetch_data(filename=str(data_home / "demo02.csv"))

        cfg = ConfigMulti(
            train_size=pd.Timedelta(days=365),
            predict_size=24,
            imputation_method="weighted",
            use_exogenous_features=False,
        )
        forecast = run(cfg, task="lazy", dataframe=df, project_name="demo02")
        print(forecast)
        ```

        Remove all cached models and artefacts for a project
        (``"clean"`` task).  Returns an empty DataFrame:

        ```{python}
        from spotforecast2_safe.multitask.runner import run

        result = run(task="clean", project_name="demo02")
        print(result.empty)
        ```
    """
    if task not in _ALL_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Choose from: {sorted(_ALL_TASKS)}. "
            "Auto-tuning tasks ('optuna', 'spotoptim') require the "
            "spotforecast2 sibling package."
        )

    if cache_home is None:
        cache_home = get_cache_home()

    if config is None:
        config = ConfigMulti()
    # ``project_name`` is the public name for what lives on the config as
    # ``data_frame_name``; keep the runner argument for ergonomic reasons.
    config.data_frame_name = project_name

    if task == "clean":
        mt = MultiTask(
            config,
            task="clean",
            cache_home=cache_home,
            dry_run=dry_run,
            log_level=log_level,
            **overrides,
        )
        mt.run()
        return pd.DataFrame()

    mt = MultiTask(
        config,
        task=task,
        dataframe=dataframe,
        data_test=data_test,
        cache_home=cache_home,
        dry_run=dry_run,
        show_progress=show_progress,
        log_level=log_level,
        **overrides,
    )
    mt.prepare_data()
    mt.detect_outliers()
    if plot_with_outliers:
        mt.plot_with_outliers()  # raises NotImplementedError in spotforecast2-safe
    mt.impute()
    mt.build_exogenous_features()
    result = mt.run(show=show)
    return result["future_pred"].to_frame("forecast")

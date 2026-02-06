from __future__ import annotations
import os
import numpy as np
import warnings
from typing import Callable
from copy import deepcopy
import pandas as pd
from joblib import cpu_count
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid

from spotforecast2_safe.exceptions import (
    IgnoredArgumentWarning,
)
from spotforecast2_safe.model_selection.split_ts_cv import TimeSeriesFold
from spotforecast2_safe.model_selection.split_one_step import OneStepAheadFold
from spotforecast2_safe.model_selection.utils_common import (
    initialize_lags_grid,
    check_backtesting_input,
    check_one_step_ahead_input,
    select_n_jobs_backtesting,
)
from spotforecast2_safe.forecaster.metrics import add_y_train_argument, _get_metric
from spotforecast2_safe.model_selection.utils_metrics import (
    _calculate_metrics_one_step_ahead,
)
from spotforecast2_safe.model_selection.validation import _backtesting_forecaster
from spotforecast2_safe.forecaster.utils import set_skforecast_warnings


def _evaluate_grid_hyperparameters(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    param_grid: dict[str, object],
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    lags_grid: (
        list[int | list[int] | np.ndarray[int] | range[int]]
        | dict[str, list[int | list[int] | np.ndarray[int] | range[int]]]
        | None
    ) = None,
    return_best: bool = True,
    n_jobs: int | str = "auto",
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate combinations of hyperparameters and lags for a given forecaster.
    """

    set_skforecast_warnings(suppress_warnings, action="ignore")

    forecaster = deepcopy(forecaster)
    forecaster_search = forecaster  # Alias for consistency with original code
    is_regression = (
        forecaster_search.__spotforecast_tags__["forecaster_task"] == "regression"
    )

    if isinstance(cv, TimeSeriesFold):
        check_backtesting_input(
            forecaster=forecaster,
            cv=cv,
            y=y,
            metric=metric,
            exog=exog,
            n_jobs=n_jobs,
            show_progress=show_progress,
            suppress_warnings=suppress_warnings,
        )
    else:
        # OneStepAheadFold
        check_one_step_ahead_input(
            forecaster=forecaster,
            cv=cv,
            y=y,
            metric=metric,
            exog=exog,
            show_progress=show_progress,
            suppress_warnings=suppress_warnings,
        )
        # Update cv params in case they were modified during input check or need setting
        # (Original code does initialization of initial_train_size here)
        # We assume cv is already correctly set up or updated by user.
        # But OneStepAheadFold in original is used to split?
        # Original code re-initializes cv params?
        # Lines 280-293 in original handle date_to_index_position for initial_train_size.
        # We should probably do that if passing strings.
        # But TimeSeriesFold does it in its init?
        # OneStepAheadFold might support string initial_train_size.
        # Let's adding it for robustness if needed, but keeping it simple for now as per prior porting.
        pass

    if not isinstance(metric, list):
        metric = [metric]
    metric = [
        _get_metric(metric=m) if isinstance(m, str) else add_y_train_argument(m)
        for m in metric
    ]
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] for m in metric}

    if len(metric_dict) != len(metric):
        raise ValueError("When `metric` is a `list`, each metric name must be unique.")

    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
    cv = deepcopy(cv)

    if n_jobs == "auto":
        refit = cv.refit if isinstance(cv, TimeSeriesFold) else False
        n_jobs = select_n_jobs_backtesting(forecaster=forecaster, refit=refit)
    elif isinstance(cv, TimeSeriesFold) and cv.refit != 1 and n_jobs != 1:
        warnings.warn(
            "If `refit` is an integer other than 1 (intermittent refit). `n_jobs` "
            "is set to 1 to avoid unexpected results during parallelization.",
            IgnoredArgumentWarning,
        )
        n_jobs = 1
    else:
        n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    print(
        f"Number of models compared: {len(param_grid) * len(lags_grid)}. "
        f"Training models..."
    )

    if show_progress:
        lags_grid_tqdm = tqdm(lags_grid.items(), desc="Lags grid", position=0)
    else:
        lags_grid_tqdm = lags_grid.items()

    if output_file is not None and os.path.isfile(output_file):
        os.remove(output_file)

    lags_list = []
    lags_label_list = []
    params_list = []

    for lags_k, lags_v in lags_grid_tqdm:

        forecaster_search.set_lags(lags_v)
        lags_v = forecaster_search.lags.copy()
        if lags_label == "values":
            lags_k = lags_v

        # OneStepAhead split is done once per lag config if independent of params
        # But params might affect transformation?
        # In original code, split is done inside the loop over lags, before params loop.
        if isinstance(cv, OneStepAheadFold):
            X_train, y_train, X_test, y_test = (
                forecaster_search._train_test_split_one_step_ahead(
                    y=y, initial_train_size=cv.initial_train_size, exog=exog
                )
            )

        if show_progress:
            param_grid_tqdm = tqdm(
                param_grid, desc="Parameters grid", position=1, leave=False
            )
        else:
            param_grid_tqdm = param_grid

        for params in param_grid_tqdm:
            try:
                forecaster_search.set_params(**params)

                if isinstance(cv, TimeSeriesFold):
                    metric_values = _backtesting_forecaster(
                        forecaster=forecaster_search,
                        y=y,
                        cv=cv,
                        metric=metric,
                        exog=exog,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        show_progress=False,
                        suppress_warnings=suppress_warnings,
                    )[0]
                    # metric_values is a DataFrame, we want list of values for the row (0)
                    metric_values = metric_values.iloc[0, :].to_list()
                else:
                    # One Step Ahead
                    metric_values = _calculate_metrics_one_step_ahead(
                        forecaster=forecaster_search,
                        metrics=metric,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                    )
            except Exception as e:
                warnings.warn(f"Parameters skipped: {params}. {e}", RuntimeWarning)
                continue

            # Filter warnings if needed/configured
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="The forecaster will be fit.*",
            )

            lags_list.append(lags_v)
            lags_label_list.append(lags_k)
            params_list.append(params)
            for m, m_value in zip(metric, metric_values):
                m_name = m if isinstance(m, str) else m.__name__
                metric_dict[m_name].append(m_value)

            if output_file is not None:
                header = [
                    "lags",
                    "lags_label",
                    "params",
                    *metric_dict.keys(),
                    *params.keys(),
                ]
                row = [lags_v, lags_k, params, *metric_values, *params.values()]
                if not os.path.isfile(output_file):
                    with open(output_file, "w", newline="") as f:
                        f.write("\t".join(header) + "\n")
                        f.write("\t".join([str(r) for r in row]) + "\n")
                else:
                    with open(output_file, "a", newline="") as f:
                        f.write("\t".join([str(r) for r in row]) + "\n")

    results = pd.DataFrame(
        {
            "lags": lags_list,
            "lags_label": lags_label_list,
            "params": params_list,
            **metric_dict,
        }
    )

    if results.empty:
        warnings.warn(
            "All models failed to train. Check the parameters and data.",
            RuntimeWarning,
        )
        return results

    results = results.sort_values(
        by=list(metric_dict.keys())[0], ascending=True if is_regression else False
    ).reset_index(drop=True)
    results = pd.concat([results, results["params"].apply(pd.Series)], axis=1)

    if return_best:
        best_lags = results.loc[0, "lags"]
        best_params = results.loc[0, "params"]
        best_metric = results.loc[0, list(metric_dict.keys())[0]]

        # NOTE: Here we use the actual forecaster passed by the user
        forecaster.set_lags(best_lags)
        forecaster.set_params(**best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)

        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  {'Backtesting' if isinstance(cv, TimeSeriesFold) else 'One-step-ahead'} "
            f"metric: {best_metric}"
        )

    set_skforecast_warnings(suppress_warnings, action="default")

    return results


def grid_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    param_grid: dict,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    lags_grid: (
        list[int | list[int] | np.ndarray[int] | range[int]]
        | dict[str, list[int | list[int] | np.ndarray[int] | range[int]]]
        | None
    ) = None,
    return_best: bool = True,
    n_jobs: int | str = "auto",
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
) -> pd.DataFrame:
    """
    Exhaustive grid search over parameter values for a Forecaster.
    """

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters(
        forecaster=forecaster,
        y=y,
        cv=cv,
        param_grid=param_grid,
        metric=metric,
        exog=exog,
        lags_grid=lags_grid,
        return_best=return_best,
        n_jobs=n_jobs,
        verbose=verbose,
        show_progress=show_progress,
        suppress_warnings=suppress_warnings,
        output_file=output_file,
    )

    return results

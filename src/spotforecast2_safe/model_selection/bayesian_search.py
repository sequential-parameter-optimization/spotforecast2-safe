"""
Bayesian hyperparameter search functions for forecasters using Optuna.
"""

from __future__ import annotations
import logging
from typing import Callable
import warnings
from copy import deepcopy
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    warnings.warn(
        "optuna is not installed. bayesian_search_forecaster will not work.",
        ImportWarning,
    )

from spotforecast2_safe.model_selection.split_ts_cv import TimeSeriesFold
from spotforecast2_safe.model_selection.split_one_step import OneStepAheadFold
from spotforecast2_safe.model_selection.validation import (
    _backtesting_forecaster,
)
from spotforecast2_safe.forecaster.metrics import add_y_train_argument, _get_metric
from spotforecast2_safe.model_selection.utils_common import (
    check_one_step_ahead_input,
    check_backtesting_input,
    select_n_jobs_backtesting,
)
from spotforecast2_safe.model_selection.utils_metrics import (
    _calculate_metrics_one_step_ahead,
)
from spotforecast2_safe.forecaster.utils import (
    initialize_lags,
    date_to_index_position,
    set_skforecast_warnings,
)
from spotforecast2_safe.exceptions import IgnoredArgumentWarning


def bayesian_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    search_space: Callable,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: int | str = "auto",
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {},
) -> tuple[pd.DataFrame, object]:
    """
    Bayesian hyperparameter optimization for a Forecaster using Optuna.

    Performs Bayesian hyperparameter search using the Optuna library for a
    Forecaster object. Validation is done using time series backtesting with
    the provided cross-validation strategy.

    Args:
        forecaster: Forecaster model. Can be ForecasterRecursive, ForecasterDirect,
            or any compatible forecaster class.
        y: Training time series values. Must be a pandas Series with a
            datetime or numeric index.
        cv: Cross-validation strategy with information needed to split the data
            into folds. Must be an instance of TimeSeriesFold or OneStepAheadFold.
        search_space: Callable function with argument `trial` that returns
            a dictionary with parameter names (str) as keys and Trial objects
            from optuna (trial.suggest_float, trial.suggest_int,
            trial.suggest_categorical) as values. Can optionally include 'lags'
            key to search over different lag configurations.
        metric: Metric(s) to quantify model goodness of fit. Can be:
            - str: One of 'mean_squared_error', 'mean_absolute_error',
              'mean_absolute_percentage_error', 'mean_squared_log_error',
              'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'.
            - Callable: Function with arguments (y_true, y_pred) or
              (y_true, y_pred, y_train) that returns a float.
            - list: List containing multiple strings and/or Callables.
        exog: Exogenous variable(s) included as predictors. Must have the
            same number of observations as `y` and aligned so that y[i] is
            regressed on exog[i]. Default is None.
        n_trials: Number of parameter settings sampled during optimization.
            Default is 10.
        random_state: Seed for sampling reproducibility. When passing a custom
            sampler in kwargs_create_study, set the seed within the sampler
            (e.g., {'sampler': TPESampler(seed=145)}). Default is 123.
        return_best: If True, refit the forecaster using the best parameters
            found on the whole dataset at the end. Default is True.
        n_jobs: Number of parallel jobs. If -1, uses all cores. If 'auto',
            uses spotforecast.skforecast.utils.select_n_jobs_backtesting to
            automatically determine the number of jobs. Default is 'auto'.
        verbose: If True, print number of folds used for cross-validation.
            Default is False.
        show_progress: Whether to show an Optuna progress bar during
            optimization. Default is True.
        suppress_warnings: If True, suppress spotforecast warnings during
            hyperparameter search. Default is False.
        output_file: Filename or full path to save results as TSV. If None,
            results are not saved to file. Default is None.
        kwargs_create_study: Additional keyword arguments passed to
            optuna.create_study(). If not specified, direction is set to
            'minimize' and TPESampler(seed=123) is used. Default is {}.
        kwargs_study_optimize: Additional keyword arguments passed to
            study.optimize(). Default is {}.

    Returns:
        tuple[pd.DataFrame, object]: A tuple containing:
            - results: DataFrame with columns 'lags', 'params', metric values,
              and individual parameter columns. Sorted by the first metric.
            - best_trial: Best optimization result as an optuna.FrozenTrial
              object containing the best parameters and metric value.

    Raises:
        ValueError: If exog length doesn't match y length when return_best=True.
        TypeError: If cv is not an instance of TimeSeriesFold or OneStepAheadFold.
        ValueError: If metric list contains duplicate metric names.

    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            f"`exog` must have same number of samples as `y`. "
            f"length `exog`: ({len(exog)}), length `y`: ({len(y)})"
        )

    results, best_trial = _bayesian_search_optuna(
        forecaster=forecaster,
        y=y,
        cv=cv,
        exog=exog,
        search_space=search_space,
        metric=metric,
        n_trials=n_trials,
        random_state=random_state,
        return_best=return_best,
        n_jobs=n_jobs,
        verbose=verbose,
        show_progress=show_progress,
        suppress_warnings=suppress_warnings,
        output_file=output_file,
        kwargs_create_study=kwargs_create_study,
        kwargs_study_optimize=kwargs_study_optimize,
    )

    return results, best_trial


def _bayesian_search_optuna(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    search_space: Callable,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: int | str = "auto",
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {},
) -> tuple[pd.DataFrame, object]:
    """
    Bayesian search for hyperparameters of a Forecaster object using Optuna library.

    This is the internal implementation function that performs the actual Bayesian
    optimization using Optuna. It handles both TimeSeriesFold (backtesting) and
    OneStepAheadFold validation strategies.
    """

    set_skforecast_warnings(suppress_warnings, action="ignore")

    forecaster_search = deepcopy(forecaster)
    forecaster_name = type(forecaster_search).__name__
    is_regression = (
        forecaster_search.__spotforecast_tags__["forecaster_task"] == "regression"
    )
    cv_name = type(cv).__name__

    if cv_name not in ["TimeSeriesFold", "OneStepAheadFold"]:
        raise TypeError(
            f"`cv` must be an instance of `TimeSeriesFold` or `OneStepAheadFold`. "
            f"Got {type(cv)}."
        )

    if cv_name == "OneStepAheadFold":

        check_one_step_ahead_input(
            forecaster=forecaster_search,
            cv=cv,
            metric=metric,
            y=y,
            exog=exog,
            show_progress=show_progress,
            suppress_warnings=False,
        )

        cv = deepcopy(cv)
        initial_train_size = date_to_index_position(
            index=cv._extract_index(y),
            date_input=cv.initial_train_size,
            method="validation",
            date_literal="initial_train_size",
        )
        cv.set_params(
            {
                "initial_train_size": initial_train_size,
                "window_size": forecaster_search.window_size,
                "differentiation": forecaster_search.differentiation_max,
                "verbose": verbose,
            }
        )
    else:
        # TimeSeriesFold
        # NOTE: Add checking input here for consistency with grid_search?
        check_backtesting_input(
            forecaster=forecaster_search,
            cv=cv,
            y=y,
            metric=metric,
            exog=exog,
            n_jobs=n_jobs,
            show_progress=show_progress,
            suppress_warnings=suppress_warnings,
        )

    if not isinstance(metric, list):
        metric = [metric]
    metric = [
        _get_metric(metric=m) if isinstance(m, str) else add_y_train_argument(m)
        for m in metric
    ]
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] for m in metric}

    if len(metric_dict) != len(metric):
        raise ValueError("When `metric` is a `list`, each metric name must be unique.")

    if n_jobs == "auto":
        # Check refit if TimeSeriesFold
        refit = cv.refit if isinstance(cv, TimeSeriesFold) else None
        n_jobs = select_n_jobs_backtesting(forecaster=forecaster_search, refit=refit)
    elif isinstance(cv, TimeSeriesFold) and cv.refit != 1 and n_jobs != 1:
        warnings.warn(
            "If `refit` is an integer other than 1 (intermittent refit). `n_jobs` "
            "is set to 1 to avoid unexpected results during parallelization.",
            IgnoredArgumentWarning,
        )
        n_jobs = 1

    # Objective function using backtesting_forecaster
    if cv_name == "TimeSeriesFold":

        def _objective(
            trial,
            search_space=search_space,
            forecaster_search=forecaster_search,
            y=y,
            cv=cv,
            exog=exog,
            metric=metric,
            n_jobs=n_jobs,
            verbose=verbose,
        ) -> float:

            sample = search_space(trial)
            sample_params = {k: v for k, v in sample.items() if k != "lags"}
            forecaster_search.set_params(**sample_params)
            if "lags" in sample:
                forecaster_search.set_lags(sample["lags"])

            metrics, _ = _backtesting_forecaster(
                forecaster=forecaster_search,
                y=y,
                cv=cv,
                metric=metric,
                exog=exog,
                n_jobs=n_jobs,
                verbose=verbose,
                show_progress=False,
                suppress_warnings=suppress_warnings,
            )
            # _backtesting_forecaster returns DataFrame, we need list of values for the SINGLE result row
            metrics_list = metrics.iloc[0, :].to_list()

            # Store metrics in the variable `metric_values` defined outside _objective.
            metric_values.append(metrics_list)

            # Return the first metric (optimized one)
            return metrics_list[0]

    else:

        def _objective(
            trial,
            search_space=search_space,
            forecaster_search=forecaster_search,
            y=y,
            cv=cv,
            exog=exog,
            metric=metric,
        ) -> float:

            sample = search_space(trial)
            sample_params = {k: v for k, v in sample.items() if k != "lags"}
            forecaster_search.set_params(**sample_params)
            if "lags" in sample:
                forecaster_search.set_lags(sample["lags"])

            X_train, y_train, X_test, y_test = (
                forecaster_search._train_test_split_one_step_ahead(
                    y=y, initial_train_size=cv.initial_train_size, exog=exog
                )
            )

            metrics_list = _calculate_metrics_one_step_ahead(
                forecaster=forecaster_search,
                metrics=metric,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            # Store all metrics in the variable `metric_values` defined outside _objective.
            metric_values.append(metrics_list)

            return metrics_list[0]

    if "direction" not in kwargs_create_study.keys():
        kwargs_create_study["direction"] = "minimize" if is_regression else "maximize"

    if show_progress:
        kwargs_study_optimize["show_progress_bar"] = True

    if output_file is not None:
        # Redirect optuna logging to file
        optuna.logging.disable_default_handler()
        logger = logging.getLogger("optuna")
        logger.setLevel(logging.INFO)
        for handler in logger.handlers.copy():
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
        handler = logging.FileHandler(output_file, mode="w")
        logger.addHandler(handler)
    else:
        logging.getLogger("optuna").setLevel(logging.WARNING)
        optuna.logging.disable_default_handler()

    # `metric_values` will be modified inside _objective function.
    # It is a trick to extract multiple values from _objective since
    # only the optimized value can be returned.
    metric_values = []

    study = optuna.create_study(**kwargs_create_study)

    if "sampler" not in kwargs_create_study.keys():
        study.sampler = TPESampler(seed=random_state)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Choices for a categorical distribution should be*",
        )
        study.optimize(_objective, n_trials=n_trials, **kwargs_study_optimize)
        best_trial = study.best_trial
        search_space_best = search_space(best_trial)

    if output_file is not None:
        handler.close()

    if search_space_best.keys() != best_trial.params.keys():
        raise ValueError(
            f"Some of the key values do not match the search_space key names.\n"
            f"  Search Space keys  : {list(search_space_best.keys())}\n"
            f"  Trial objects keys : {list(best_trial.params.keys())}."
        )

    lags_list = []
    params_list = []

    # Optuna does not guarantee order of trials in get_trials() matches execution order
    # strictly if parallel? But here n_jobs is for backtesting, study.optimize is sequential usually unless specified.
    # Wait, study.optimize with n_jobs > 1? NO, bayesian_search_forecaster argument `n_jobs` is passed to `_backtesting_forecaster` or `cv` parallelization.
    # Optuna itself is running sequentially here (study.optimize call default n_jobs=1).
    # So `metric_values` append order should match `study.get_trials()` order IF optuna preserves that.
    # Optuna trials are stored in ID order usually.
    # To be safe, we should rely on `trial.number`.
    # But `metric_values` is a list appended during execution.
    # If optuna runs sequentially, it matches trial creation order.

    for i, trial in enumerate(study.get_trials()):
        estimator_params = {k: v for k, v in trial.params.items() if k != "lags"}
        lags = trial.params.get(
            "lags",
            forecaster_search.lags if hasattr(forecaster_search, "lags") else None,
        )
        params_list.append(estimator_params)
        lags_list.append(lags)

        # We assume metric_values[i] corresponds to trial i.
        # This is true for sequential optimization.
        for m, m_values in zip(metric, metric_values[i]):
            m_name = m if isinstance(m, str) else m.__name__
            metric_dict[m_name].append(m_values)

    lags_list = [
        initialize_lags(forecaster_name=forecaster_name, lags=lag)[0]
        for lag in lags_list
    ]

    results = pd.DataFrame({"lags": lags_list, "params": params_list, **metric_dict})

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
            f"  {'Backtesting' if cv_name == 'TimeSeriesFold' else 'One-step-ahead'} "
            f"metric: {best_metric}"
        )

    set_skforecast_warnings(suppress_warnings, action="default")

    return results, best_trial

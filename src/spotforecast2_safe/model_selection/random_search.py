"""Random search hyperparameter optimization for forecasters."""

from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from spotforecast2_safe.model_selection.split_ts_cv import TimeSeriesFold
from spotforecast2_safe.model_selection.split_one_step import OneStepAheadFold
from spotforecast2_safe.model_selection.grid_search import (
    _evaluate_grid_hyperparameters,
)


def random_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    param_distributions: dict,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    lags_grid: (
        list[int | list[int] | np.ndarray[int] | range[int]]
        | dict[str, list[int | list[int] | np.ndarray[int] | range[int]]]
        | None
    ) = None,
    n_iter: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: int | str = "auto",
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
) -> pd.DataFrame:
    """Random search over parameter distributions for a Forecaster.

    Performs random sampling of parameter settings from distributions for a
    Forecaster object. Validation is done using time series backtesting with
    the provided cross-validation strategy. This is more efficient than grid
    search when exploring large parameter spaces.

    Args:
        forecaster: Forecaster model (ForecasterRecursive or ForecasterDirect).
        y: Training time series.
        cv: Cross-validation strategy (TimeSeriesFold or OneStepAheadFold)
            with information needed to split the data into folds.
        param_distributions: Dictionary with parameter names (str) as keys
            and distributions or lists of parameters to try as values.
            Use scipy.stats distributions for continuous parameters.
        metric: Metric(s) to quantify model goodness of fit. If str:
            'mean_squared_error', 'mean_absolute_error',
            'mean_absolute_percentage_error', 'mean_squared_log_error',
            'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'.
            If Callable: Function with arguments (y_true, y_pred, y_train)
            that returns a float. If list: Multiple strings and/or Callables.
        exog: Exogenous variable(s) included as predictors. Must have the
            same number of observations as y and aligned so that y[i] is
            regressed on exog[i]. Default is None.
        lags_grid: Lists of lags to try. Can be int, lists, numpy ndarray,
            or range objects. If dict, keys are used as labels in results
            DataFrame. Default is None.
        n_iter: Number of parameter settings sampled per lags configuration.
            Trades off runtime vs solution quality. Default is 10.
        random_state: Seed for random sampling for reproducible output.
            Default is 123.
        return_best: If True, refit the forecaster using best parameters
            on the whole dataset. Default is True.
        n_jobs: Number of jobs to run in parallel. If -1, uses all cores.
            If 'auto', uses select_n_jobs_backtesting. Default is 'auto'.
        verbose: If True, print number of folds used for cv. Default is False.
        show_progress: Whether to show a progress bar. Default is True.
        suppress_warnings: If True, suppress spotforecast warnings during
            hyperparameter search. Default is False.
        output_file: Filename or full path to save results as TSV. If None,
            results are not saved to file. Default is None.

    Returns:
        Results for each parameter combination with columns: lags (lags
        configuration), lags_label (descriptive label), params (parameters
        configuration), metric (metric value), and additional columns with
        param=value pairs.

    Examples:
        Basic random search with continuous parameter distributions:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.linear_model import Ridge
        >>> from scipy.stats import uniform
        >>> from spotforecast2.forecaster.recursive import ForecasterRecursive
        >>> from spotforecast2.model_selection import TimeSeriesFold
        >>> from spotforecast2.model_selection.random_search import random_search_forecaster
        >>>
        >>> # Create sample data
        >>> np.random.seed(123)
        >>> y = pd.Series(np.random.randn(50), name='y')
        >>>
        >>> # Set up forecaster and cross-validation
        >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
        >>> cv = TimeSeriesFold(steps=3, initial_train_size=20, refit=False)
        >>>
        >>> # Define parameter distributions with scipy.stats
        >>> param_distributions = {
        ...     'estimator__alpha': uniform(0.1, 10.0)  # Uniform between 0.1 and 10.1
        ... }
        >>>
        >>> # Run random search
        >>> results = random_search_forecaster(
        ...     forecaster=forecaster,
        ...     y=y,
        ...     cv=cv,
        ...     param_distributions=param_distributions,
        ...     metric='mean_squared_error',
        ...     n_iter=5,
        ...     random_state=42,
        ...     return_best=False,
        ...     verbose=False,
        ...     show_progress=False
        ... )
        >>>
        >>> # Check results
        >>> print(results.shape[0])
        5
        >>> print('estimator__alpha' in results.columns)
        True
        >>> print('mean_squared_error' in results.columns)
        True
    """

    param_grid = list(
        ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state)
    )

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

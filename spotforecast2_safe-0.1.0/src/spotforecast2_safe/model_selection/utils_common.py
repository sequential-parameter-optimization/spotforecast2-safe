"""Common validation and initialization utilities for model selection."""

from __future__ import annotations
from typing import Callable
import warnings
import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.exceptions import NotFittedError
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin
from sklearn.pipeline import Pipeline

from spotforecast2_safe.forecaster.utils import check_interval, date_to_index_position


class OneStepAheadValidationWarning(UserWarning):
    """
    Warning used when validation is performed with one-step-ahead predictions.
    """

    pass


def initialize_lags_grid(
    forecaster: object,
    lags_grid: (
        list[int | list[int] | np.ndarray[int] | range[int]]
        | dict[str, list[int | list[int] | np.ndarray[int] | range[int]]]
        | None
    ) = None,
) -> tuple[dict[str, int], str]:
    """
    Initialize lags grid and lags label for model selection.

    Args:
        forecaster: Forecaster model. ForecasterRecursive, ForecasterDirect,
            ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate.
        lags_grid: Lists of lags to try, containing int, lists, numpy ndarray, or range
            objects. If `dict`, the keys are used as labels in the `results`
            DataFrame, and the values are used as the lists of lags to try.

    Returns:
        tuple: (lags_grid, lags_label)
            - lags_grid (dict): Dictionary with lags configuration for each iteration.
            - lags_label (str): Label for lags representation in the results object.

    Examples:
        >>> from spotforecast2_safe.model_selection.utils_common import initialize_lags_grid
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from sklearn.linear_model import LinearRegression
        >>> forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        >>> lags_grid = [2, 4]
        >>> lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
        >>> print(lags_grid)
        {'2': 2, '4': 4}
        >>> print(lags_label)
        values
    """

    if not isinstance(lags_grid, (list, dict, type(None))):
        raise TypeError(
            f"`lags_grid` argument must be a list, dict or None. "
            f"Got {type(lags_grid)}."
        )

    lags_label = "values"
    if isinstance(lags_grid, list):
        lags_grid = {f"{lags}": lags for lags in lags_grid}
    elif lags_grid is None:
        lags = [int(lag) for lag in forecaster.lags]  # Required since numpy 2.0
        lags_grid = {f"{lags}": lags}
    else:
        lags_label = "keys"

    return lags_grid, lags_label


def check_backtesting_input(
    forecaster: object,
    cv: object,
    metric: str | Callable | list[str | Callable],
    add_aggregated_metric: bool = True,
    y: pd.Series | None = None,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame] = None,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    interval: float | list[float] | tuple[float] | str | object | None = None,
    interval_method: str = "bootstrapping",
    alpha: float | None = None,
    n_boot: int = 250,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = True,
    random_state: int = 123,
    return_predictors: bool = False,
    freeze_params: bool = True,
    n_jobs: int | str = "auto",
    show_progress: bool = True,
    suppress_warnings: bool = False,
) -> None:
    """
    This is a helper function to check most inputs of backtesting functions in
    modules `model_selection`.

    Args:
        forecaster: Forecaster model.
        cv: TimeSeriesFold object with the information needed to split the data into folds.
        metric: Metric used to quantify the goodness of fit of the model.
        add_aggregated_metric: If `True`, the aggregated metrics (average, weighted average and pooling)
            over all levels are also returned (only multiseries).
        y: Training time series for uni-series forecasters.
        series: Training time series for multi-series forecasters.
        exog: Exogenous variables.
        interval: Specifies whether probabilistic predictions should be estimated and the
            method to use. The following options are supported:

            - If `float`, represents the nominal (expected) coverage (between 0 and 1).
            For instance, `interval=0.95` corresponds to `[2.5, 97.5]` percentiles.
            - If `list` or `tuple`: Sequence of percentiles to compute, each value must
            be between 0 and 100 inclusive. For example, a 95% confidence interval can
            be specified as `interval = [2.5, 97.5]` or multiple percentiles (e.g. 10,
            50 and 90) as `interval = [10, 50, 90]`.
            - If 'bootstrapping' (str): `n_boot` bootstrapping predictions will be generated.
            - If scipy.stats distribution object, the distribution parameters will
            be estimated for each prediction.
            - If None, no probabilistic predictions are estimated.
        interval_method: Technique used to estimate prediction intervals. Available options:

            - 'bootstrapping': Bootstrapping is used to generate prediction
            intervals.
            - 'conformal': Employs the conformal prediction split method for
            interval estimation.
        alpha: The confidence intervals used in ForecasterStats are (1 - alpha) %.
        n_boot: Number of bootstrapping iterations to perform when estimating prediction
            intervals.
        use_in_sample_residuals: If `True`, residuals from the training data are used as proxy of prediction
            error to create prediction intervals.  If `False`, out_sample_residuals
            are used if they are already stored inside the forecaster.
        use_binned_residuals: If `True`, residuals are selected based on the predicted values
            (binned selection).
            If `False`, residuals are selected randomly.
        random_state: Seed for the random number generator to ensure reproducibility.
        return_predictors: If `True`, the predictors used to make the predictions are also returned.
        n_jobs: The number of jobs to run in parallel. If `-1`, then the number of jobs is
            set to the number of cores. If 'auto', `n_jobs` is set using the function
            select_n_jobs_fit_forecaster.
        freeze_params: Determines whether to freeze the model parameters after the first fit
            for estimators that perform automatic model selection.

            - If `True`, the model parameters found during the first fit (e.g., order
            and seasonal_order for Arima, or smoothing parameters for Ets) are reused
            in all subsequent refits. This avoids re-running the automatic selection
            procedure in each fold and reduces runtime.
            - If `False`, automatic model selection is performed independently in each
            refit, allowing parameters to adapt across folds. This increases runtime
            and adds a `params` column to the output with the parameters selected per
            fold.
        show_progress: Whether to show a progress bar.
        suppress_warnings: If `True`, spotforecast warnings will be suppressed during the backtesting
            process.

    Returns:
        None

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.model_selection.utils_common import check_backtesting_input
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from spotforecast2_safe.model_selection import TimeSeriesFold
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.metrics import mean_squared_error
        >>> y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        >>> cv = TimeSeriesFold(
        ...     steps=3,
        ...     initial_train_size=5,
        ...     gap=0,
        ...     refit=False,
        ...     fixed_train_size=False,
        ...     allow_incomplete_fold=True
        ... )
        >>> check_backtesting_input(
        ...     forecaster=forecaster,
        ...     cv=cv,
        ...     metric=mean_squared_error,
        ...     y=y
        ... )
    """

    forecaster_name = type(forecaster).__name__
    cv_name = type(cv).__name__

    if cv_name != "TimeSeriesFold":
        raise TypeError(f"`cv` must be a 'TimeSeriesFold' object. Got '{cv_name}'.")

    steps = cv.steps
    initial_train_size = cv.initial_train_size
    gap = cv.gap
    allow_incomplete_fold = cv.allow_incomplete_fold
    refit = cv.refit

    forecasters_uni = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterStats",
        "ForecasterEquivalentDate",
        "ForecasterRecursiveClassifier",
    ]
    forecasters_direct = [
        "ForecasterDirect",
        "ForecasterDirectMultiVariate",
        "ForecasterRnn",
    ]
    forecasters_multi_no_dict = [
        "ForecasterDirectMultiVariate",
        "ForecasterRnn",
    ]
    forecasters_multi_dict = ["ForecasterRecursiveMultiSeries"]
    # NOTE: ForecasterStats has interval but not with bootstrapping or conformal
    forecasters_boot_conformal = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveMultiSeries",
        "ForecasterDirectMultiVariate",
        "ForecasterEquivalentDate",
    ]
    forecasters_return_predictors = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveMultiSeries",
        "ForecasterDirectMultiVariate",
        "ForecasterRecursiveClassifier",
    ]

    if forecaster_name in forecasters_uni:
        if not isinstance(y, pd.Series):
            raise TypeError("`y` must be a pandas Series.")
        data_name = "y"
        data_length = len(y)

    elif forecaster_name in forecasters_multi_no_dict:
        if not isinstance(series, pd.DataFrame):
            raise TypeError("`series` must be a pandas DataFrame.")
        data_name = "series"
        data_length = len(series)

    elif forecaster_name in forecasters_multi_dict:

        # NOTE: Checks are not need as they are done in the function
        # `check_preprocess_series` that is used before `check_backtesting_input`
        # in the backtesting function.

        data_name = "series"
        data_length = max([len(series[serie]) for serie in series])

    if exog is not None:
        if forecaster_name in forecasters_multi_dict:
            # NOTE: Checks are not need as they are done in the function
            # `check_preprocess_exog_multiseries` that is used before
            # `check_backtesting_input` in the backtesting function.
            pass
        else:
            if not isinstance(exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    f"`exog` must be a pandas Series, DataFrame or None. Got {type(exog)}."
                )

    if hasattr(forecaster, "differentiation"):
        if forecaster.differentiation_max != cv.differentiation:
            if forecaster_name == "ForecasterRecursiveMultiSeries" and isinstance(
                forecaster.differentiation, dict
            ):
                raise ValueError(
                    f"When using a dict as `differentiation` in ForecasterRecursiveMultiSeries, "
                    f"the `differentiation` included in the cv ({cv.differentiation}) must be "
                    f"the same as the maximum `differentiation` included in the forecaster "
                    f"({forecaster.differentiation_max}). Set the same value "
                    f"for both using the `differentiation` argument."
                )
            else:
                raise ValueError(
                    f"The differentiation included in the forecaster "
                    f"({forecaster.differentiation_max}) differs from the differentiation "
                    f"included in the cv ({cv.differentiation}). Set the same value "
                    f"for both using the `differentiation` argument."
                )

    if not isinstance(metric, (str, Callable, list)):
        raise TypeError(
            f"`metric` must be a string, a callable function, or a list containing "
            f"multiple strings and/or callables. Got {type(metric)}."
        )

    if forecaster_name == "ForecasterEquivalentDate" and isinstance(
        forecaster.offset, pd.tseries.offsets.DateOffset
    ):
        # NOTE: Checks when initial_train_size is not None cannot be done here
        # because the forecaster is not fitted yet and we don't know the
        # window_size since pd.DateOffset is not a fixed window size.
        if initial_train_size is None:
            raise ValueError(
                f"`initial_train_size` must be an integer greater than "
                f"the `window_size` of the forecaster ({forecaster.window_size}) "
                f"and smaller than the length of `{data_name}` ({data_length}) or "
                f"a date within this range of the index."
            )
    elif initial_train_size is not None:
        if forecaster_name in forecasters_uni:
            index = cv._extract_index(y)
        else:
            index = cv._extract_index(series)

        initial_train_size = date_to_index_position(
            index=index,
            date_input=initial_train_size,
            method="validation",
            date_literal="initial_train_size",
        )
        if (
            initial_train_size < forecaster.window_size
            or initial_train_size >= data_length
        ):
            raise ValueError(
                f"If `initial_train_size` is an integer, it must be greater than "
                f"the `window_size` of the forecaster ({forecaster.window_size}) "
                f"and smaller than the length of `{data_name}` ({data_length}). If "
                f"it is a date, it must be within this range of the index."
            )
        if allow_incomplete_fold:
            # At least one observation after the gap to allow incomplete fold
            if data_length <= initial_train_size + gap:
                raise ValueError(
                    f"`{data_name}` must have more than `initial_train_size + gap` "
                    f"observations to create at least one fold.\n"
                    f"    Time series length: {data_length}\n"
                    f"    Required > {initial_train_size + gap}\n"
                    f"    initial_train_size: {initial_train_size}\n"
                    f"    gap: {gap}\n"
                )
        else:
            # At least one complete fold
            if data_length < initial_train_size + gap + steps:
                raise ValueError(
                    f"`{data_name}` must have at least `initial_train_size + gap + steps` "
                    f"observations to create a minimum of one complete fold "
                    f"(allow_incomplete_fold=False).\n"
                    f"    Time series length: {data_length}\n"
                    f"    Required >= {initial_train_size + gap + steps}\n"
                    f"    initial_train_size: {initial_train_size}\n"
                    f"    gap: {gap}\n"
                    f"    steps: {steps}\n"
                )
    else:
        if forecaster_name in ["ForecasterStats", "ForecasterEquivalentDate"]:
            raise ValueError(
                f"When using {forecaster_name}, `initial_train_size` must be an "
                f"integer smaller than the length of `{data_name}` ({data_length})."
            )
        else:
            if not forecaster.is_fitted:
                raise NotFittedError(
                    "`forecaster` must be already trained if no `initial_train_size` "
                    "is provided."
                )
            if refit:
                raise ValueError(
                    "`refit` is only allowed when `initial_train_size` is not `None`."
                )

    if forecaster_name == "ForecasterStats" and cv.skip_folds is not None:
        raise ValueError(
            "`skip_folds` is not allowed for ForecasterStats. Set it to `None`."
        )

    if not isinstance(add_aggregated_metric, bool):
        raise TypeError("`add_aggregated_metric` must be a boolean: `True`, `False`.")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise TypeError(f"`n_boot` must be an integer greater than 0. Got {n_boot}.")
    if not isinstance(use_in_sample_residuals, bool):
        raise TypeError("`use_in_sample_residuals` must be a boolean: `True`, `False`.")
    if not isinstance(use_binned_residuals, bool):
        raise TypeError("`use_binned_residuals` must be a boolean: `True`, `False`.")
    if not isinstance(random_state, (int, np.integer)) or random_state < 0:
        raise TypeError(
            f"`random_state` must be an integer greater than 0. Got {random_state}."
        )
    if not isinstance(return_predictors, bool):
        raise TypeError("`return_predictors` must be a boolean: `True`, `False`.")
    if not isinstance(freeze_params, bool):
        raise TypeError("`freeze_params` must be a boolean: `True`, `False`.")
    if not isinstance(n_jobs, int) and n_jobs != "auto":
        raise TypeError(f"`n_jobs` must be an integer or `'auto'`. Got {n_jobs}.")
    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean: `True`, `False`.")
    if not isinstance(suppress_warnings, bool):
        raise TypeError("`suppress_warnings` must be a boolean: `True`, `False`.")

    if interval is not None or alpha is not None:

        if forecaster_name in forecasters_boot_conformal:

            if interval_method == "conformal":
                if not isinstance(interval, (float, list, tuple)):
                    raise TypeError(
                        f"When `interval_method` is 'conformal', `interval` must "
                        f"be a float or a list/tuple defining a symmetric interval. "
                        f"Got {type(interval)}."
                    )
            elif interval_method == "bootstrapping":
                if not isinstance(interval, (float, list, tuple, str)) and (
                    not hasattr(interval, "_pdf")
                    or not callable(getattr(interval, "fit", None))
                ):
                    raise TypeError(
                        f"When `interval_method` is 'bootstrapping', `interval` "
                        f"must be a float, a list or tuple of floats, a "
                        f"scipy.stats distribution object (with methods `_pdf` and "
                        f"`fit`) or the string 'bootstrapping'. Got {type(interval)}."
                    )
                if isinstance(interval, (list, tuple)):
                    for i in interval:
                        if not isinstance(i, (int, float)):
                            raise TypeError(
                                f"`interval` must be a list or tuple of floats. "
                                f"Got {type(i)} in {interval}."
                            )
                    if len(interval) == 2:
                        check_interval(interval=interval)
                    else:
                        for q in interval:
                            if (q < 0.0) or (q > 100.0):
                                raise ValueError(
                                    "When `interval` is a list or tuple, all values must be "
                                    "between 0 and 100 inclusive."
                                )
                elif isinstance(interval, str):
                    if interval != "bootstrapping":
                        raise ValueError(
                            f"When `interval` is a string, it must be 'bootstrapping'."
                            f"Got {interval}."
                        )
            else:
                raise ValueError(
                    f"`interval_method` must be 'bootstrapping' or 'conformal'. "
                    f"Got {interval_method}."
                )
        else:
            if forecaster_name == "ForecasterRecursiveClassifier":
                raise ValueError(
                    f"`interval` is not supported for {forecaster_name}. Class "
                    f"probabilities are returned by default during backtesting, "
                    f"set `interval=None`."
                )
            check_interval(interval=interval, alpha=alpha)

    if return_predictors and forecaster_name not in forecasters_return_predictors:
        raise ValueError(
            f"`return_predictors` is only allowed for forecasters of type "
            f"{forecasters_return_predictors}. Got {forecaster_name}."
        )

    if forecaster_name in forecasters_direct and forecaster.max_step < steps + gap:
        raise ValueError(
            f"When using a {forecaster_name}, the combination of steps "
            f"+ gap ({steps + gap}) cannot be greater than the `steps` parameter "
            f"declared when the forecaster is initialized ({forecaster.max_step})."
        )


def check_one_step_ahead_input(
    forecaster: object,
    cv: object,
    metric: str | Callable | list[str | Callable],
    y: pd.Series | None = None,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame] = None,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    show_progress: bool = True,
    suppress_warnings: bool = False,
) -> None:
    """
    This is a helper function to check most inputs of hyperparameter tuning
    functions in modules `model_selection` when using a `OneStepAheadFold`.

    Args:
        forecaster: Forecaster model.
        cv: OneStepAheadFold object with the information needed to split the data into folds.
        metric: Metric used to quantify the goodness of fit of the model.
        y: Training time series for uni-series forecasters.
        series: Training time series for multi-series forecasters.
        exog: Exogenous variables.
        show_progress: Whether to show a progress bar.
        suppress_warnings: If `True`, spotforecast warnings will be suppressed during the hyperparameter
            search.

    Returns:
        None

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.model_selection.utils_common import check_one_step_ahead_input
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from spotforecast2_safe.model_selection import OneStepAheadFold
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.metrics import mean_squared_error
        >>> y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        >>> cv = OneStepAheadFold(
        ...     initial_train_size=5,
        ...     return_all_predictions=False
        ... )
        >>> check_one_step_ahead_input(
        ...     forecaster=forecaster,
        ...     cv=cv,
        ...     metric=mean_squared_error,
        ...     y=y
        ... )
    """

    forecaster_name = type(forecaster).__name__
    cv_name = type(cv).__name__

    if cv_name != "OneStepAheadFold":
        raise TypeError(f"`cv` must be a 'OneStepAheadFold' object. Got '{cv_name}'.")

    initial_train_size = cv.initial_train_size

    forecasters_one_step_ahead = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveClassifier",
        "ForecasterRecursiveMultiSeries",
        "ForecasterDirectMultiVariate",
    ]
    if forecaster_name not in forecasters_one_step_ahead:
        raise TypeError(
            f"Only forecasters of type {forecasters_one_step_ahead} are allowed "
            f"when using `cv` of type `OneStepAheadFold`. Got {forecaster_name}."
        )

    forecasters_uni = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveClassifier",
    ]
    forecasters_multi_no_dict = [
        "ForecasterDirectMultiVariate",
    ]
    forecasters_multi_dict = ["ForecasterRecursiveMultiSeries"]

    if forecaster_name in forecasters_uni:
        if not isinstance(y, pd.Series):
            raise TypeError(f"`y` must be a pandas Series. Got {type(y)}")
        data_name = "y"
        data_length = len(y)

    elif forecaster_name in forecasters_multi_no_dict:
        if not isinstance(series, pd.DataFrame):
            raise TypeError(f"`series` must be a pandas DataFrame. Got {type(series)}")
        data_name = "series"
        data_length = len(series)

    elif forecaster_name in forecasters_multi_dict:

        # NOTE: Checks are not need as they are done in the function
        # `check_preprocess_series` that is used before `check_one_step_ahead_input`
        # in the backtesting function.

        data_name = "series"
        data_length = max([len(series[serie]) for serie in series])

    if exog is not None:
        if forecaster_name in forecasters_multi_dict:
            # NOTE: Checks are not need as they are done in the function
            # `check_preprocess_exog_multiseries` that is used before
            # `check_backtesting_input` in the backtesting function.
            pass
        else:
            if not isinstance(exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    f"`exog` must be a pandas Series, DataFrame or None. Got {type(exog)}."
                )

    if hasattr(forecaster, "differentiation"):
        if forecaster.differentiation_max != cv.differentiation:
            if forecaster_name == "ForecasterRecursiveMultiSeries" and isinstance(
                forecaster.differentiation, dict
            ):
                raise ValueError(
                    f"When using a dict as `differentiation` in ForecasterRecursiveMultiSeries, "
                    f"the `differentiation` included in the cv ({cv.differentiation}) must be "
                    f"the same as the maximum `differentiation` included in the forecaster "
                    f"({forecaster.differentiation_max}). Set the same value "
                    f"for both using the `differentiation` argument."
                )
            else:
                raise ValueError(
                    f"The differentiation included in the forecaster "
                    f"({forecaster.differentiation_max}) differs from the differentiation "
                    f"included in the cv ({cv.differentiation}). Set the same value "
                    f"for both using the `differentiation` argument."
                )

    if not isinstance(metric, (str, Callable, list)):
        raise TypeError(
            f"`metric` must be a string, a callable function, or a list containing "
            f"multiple strings and/or callables. Got {type(metric)}."
        )

    if forecaster_name in forecasters_uni:
        index = cv._extract_index(y)
    else:
        index = cv._extract_index(series)

    initial_train_size = date_to_index_position(
        index=index,
        date_input=initial_train_size,
        method="validation",
        date_literal="initial_train_size",
    )
    if initial_train_size < forecaster.window_size or initial_train_size >= data_length:
        raise ValueError(
            f"If `initial_train_size` is an integer, it must be greater than "
            f"the `window_size` of the forecaster ({forecaster.window_size}) "
            f"and smaller than the length of `{data_name}` ({data_length}). If "
            f"it is a date, it must be within this range of the index."
        )

    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean: `True`, `False`.")
    if not isinstance(suppress_warnings, bool):
        raise TypeError("`suppress_warnings` must be a boolean: `True`, `False`.")

    if not suppress_warnings:
        warnings.warn(
            "One-step-ahead predictions are used for faster model comparison, but they "
            "may not fully represent multi-step prediction performance. It is recommended "
            "to backtest the final model for a more accurate multi-step performance "
            "estimate.",
            OneStepAheadValidationWarning,
        )


def select_n_jobs_backtesting(forecaster: object, refit: bool | int) -> int:
    """
    Select the optimal number of jobs to use in the backtesting process. This
    selection is based on heuristics and is not guaranteed to be optimal.

    The number of jobs is chosen as follows:

    - If `refit` is an integer, then `n_jobs = 1`. This is because parallelization doesn't
    work with intermittent refit.
    - If forecaster is 'ForecasterRecursive' and estimator is a linear estimator,
    then `n_jobs = 1`.
    - If forecaster is 'ForecasterRecursive' and estimator is not a linear
    estimator then `n_jobs = cpu_count() - 1`.
    - If forecaster is 'ForecasterDirect' or 'ForecasterDirectMultiVariate'
    and `refit = True`, then `n_jobs = cpu_count() - 1`.
    - If forecaster is 'ForecasterDirect' or 'ForecasterDirectMultiVariate'
    and `refit = False`, then `n_jobs = 1`.
    - If forecaster is 'ForecasterRecursiveMultiSeries', then `n_jobs = cpu_count() - 1`.
    - If forecaster is 'ForecasterStats' or 'ForecasterEquivalentDate',
    then `n_jobs = 1`.
    - If estimator is a `LGBMRegressor(n_jobs=1)`, then `n_jobs = cpu_count() - 1`.
    - If estimator is a `LGBMRegressor` with internal n_jobs != 1, then `n_jobs = 1`.
    This is because `lightgbm` is highly optimized for gradient boosting and
    parallelizes operations at a very fine-grained level, making additional
    parallelization unnecessary and potentially harmful due to resource contention.

    Args:
        forecaster: Forecaster model.
        refit: If the forecaster is refitted during the backtesting process.

    Returns:
        int: The number of jobs to run in parallel.

    Examples:
        >>> from spotforecast2_safe.model_selection.utils_common import select_n_jobs_backtesting
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from sklearn.linear_model import LinearRegression
        >>> forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        >>> select_n_jobs_backtesting(forecaster, refit=True)
        1
    """

    forecaster_name = type(forecaster).__name__

    if forecaster_name == "ForecasterStats":
        n_jobs = 1
        return n_jobs

    if isinstance(forecaster.estimator, Pipeline):
        estimator = forecaster.estimator[-1]
    else:
        estimator = forecaster.estimator

    refit = False if refit == 0 else refit
    if not isinstance(refit, bool) and refit != 1:
        n_jobs = 1
    else:
        if forecaster_name in {"ForecasterRecursive", "ForecasterRecursiveClassifier"}:
            if isinstance(estimator, (LinearModel, LinearClassifierMixin)):
                n_jobs = 1
            elif type(estimator).__name__ in {"LGBMRegressor", "LGBMClassifier"}:
                n_jobs = cpu_count() - 1 if estimator.n_jobs == 1 else 1
            else:
                n_jobs = cpu_count() - 1
        elif forecaster_name in {"ForecasterDirect", "ForecasterDirectMultiVariate"}:
            # Parallelization is applied during the fitting process.
            n_jobs = 1
        elif forecaster_name in {"ForecasterRecursiveMultiSeries"}:
            if type(estimator).__name__ == "LGBMRegressor":
                n_jobs = cpu_count() - 1 if estimator.n_jobs == 1 else 1
            else:
                n_jobs = cpu_count() - 1
        elif forecaster_name in {"ForecasterEquivalentDate"}:
            n_jobs = 1
        else:
            n_jobs = 1

    return n_jobs

import pandas as pd
from sklearn.linear_model import LinearRegression
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.model_selection import TimeSeriesFold
from spotforecast2_safe.model_selection.validation import _backtesting_forecaster
import numpy as np
from spotforecast2_safe.model_selection.split_one_step import OneStepAheadFold
from spotforecast2_safe.model_selection.validation import (
    backtesting_forecaster_one_step,
)


def test_backtesting_forecaster_example():
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    cv = TimeSeriesFold(steps=2, initial_train_size=6)
    metric_values, backtest_predictions = _backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric="mean_squared_error",
        show_progress=False,
    )
    # Check output types and shapes
    assert isinstance(metric_values, pd.DataFrame)
    assert isinstance(backtest_predictions, pd.DataFrame)
    assert not metric_values.empty
    assert not backtest_predictions.empty


def test_backtesting_forecaster_one_step():
    y = pd.Series(np.random.randn(100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    cv = OneStepAheadFold(initial_train_size=20, window_size=5)
    metric_values, backtest_predictions = backtesting_forecaster_one_step(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric="mean_squared_error",
        exog=None,
        interval=0.95,
        interval_method="bootstrapping",
        n_boot=20,
        use_in_sample_residuals=True,
        use_binned_residuals=False,
        random_state=42,
        return_predictors=False,
        n_jobs=1,
        verbose=False,
        show_progress=False,
        suppress_warnings=True,
    )
    assert not metric_values.empty
    assert not backtest_predictions.empty

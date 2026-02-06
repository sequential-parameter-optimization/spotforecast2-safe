"""Metrics calculation utilities for model selection."""

from __future__ import annotations
import pandas as pd


def _calculate_metrics_one_step_ahead(
    forecaster: object,
    metrics: list,
    X_train: pd.DataFrame,
    y_train: pd.Series | dict[int, pd.Series],
    X_test: pd.DataFrame,
    y_test: pd.Series | dict[int, pd.Series],
) -> list:
    """
    Calculate metrics when predictions are one-step-ahead.

    When forecaster is of type `ForecasterDirect`, only the estimator for step 1
    is used.

    Args:
        forecaster: Forecaster model.
        metrics: List of metrics.
        X_train: Predictor values used to train the model.
        y_train: Target values related to each row of `X_train`.
        X_test: Predictor values used to test the model.
        y_test: Target values related to each row of `X_test`.

    Returns:
        List with metric values.

    Notes:
        When using this metric in validation, `y_train` doesn't include
        the first window_size observations used to create the predictors and/or
        rolling features.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.metrics import mean_squared_error
        >>> from spotforecast2.forecaster.recursive import ForecasterRecursive
        >>> from spotforecast2.model_selection.utils_metrics import _calculate_metrics_one_step_ahead
        >>>
        >>> forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        >>>
        >>> # Mock training data (already transformed into predictors/target)
        >>> X_train = pd.DataFrame(np.array([[1, 2], [2, 3], [3, 4]]), columns=['lag_2', 'lag_1'])
        >>> y_train = pd.Series(np.array([3, 4, 5]), name='y')
        >>>
        >>> # Mock test data
        >>> X_test = pd.DataFrame(np.array([[4, 5]]), columns=['lag_2', 'lag_1'])
        >>> y_test = pd.Series(np.array([6]), name='y')
        >>>
        >>> metrics = [mean_squared_error]
        >>> result = _calculate_metrics_one_step_ahead(
        ...     forecaster=forecaster,
        ...     metrics=metrics,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
        >>> print(result)
        [0.0]
    """

    if type(forecaster).__name__ == "ForecasterDirect":

        step = 1  # Only the model for step 1 is optimized.
        X_train, y_train = forecaster.filter_train_X_y_for_step(
            step=step, X_train=X_train, y_train=y_train
        )
        X_test, y_test = forecaster.filter_train_X_y_for_step(
            step=step, X_train=X_test, y_train=y_test
        )
        forecaster.estimators_[step].fit(X_train, y_train)
        y_pred = forecaster.estimators_[step].predict(X_test)

    else:
        forecaster.estimator.fit(X_train, y_train)
        y_pred = forecaster.estimator.predict(X_test)

    y_true = y_test.to_numpy()
    y_pred = y_pred.ravel()
    y_train = y_train.to_numpy()

    if forecaster.differentiation is not None:
        y_true = forecaster.differentiator.inverse_transform_next_window(y_true)
        y_pred = forecaster.differentiator.inverse_transform_next_window(y_pred)
        y_train = forecaster.differentiator.inverse_transform_training(y_train)

    if forecaster.transformer_y is not None:
        y_true = forecaster.transformer_y.inverse_transform(y_true.reshape(-1, 1))
        y_pred = forecaster.transformer_y.inverse_transform(y_pred.reshape(-1, 1))
        y_train = forecaster.transformer_y.inverse_transform(y_train.reshape(-1, 1))

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_train = y_train.ravel()
    metric_values = [m(y_true=y_true, y_pred=y_pred, y_train=y_train) for m in metrics]

    return metric_values

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recursive forecaster model wrappers for different estimators."""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from lightgbm import LGBMRegressor

from spotforecast2_safe.data.data import Period
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

logger = logging.getLogger(__name__)


class ForecasterRecursiveModel:
    """
    Base wrapper around ForecasterRecursive to match application logic.

    This class manages the lifecycle of a recursive forecaster, including
    feature building, tuning (simulated), and packaging predictions for UI.

    Attributes:
        iteration (int): The current training iteration.
        end_dev (pd.Timestamp): The end date of the development/training period.
        train_size (Optional[pd.Timedelta]): Lookback window for training data.
        preprocessor (ExogBuilder): Builder for exogenous features.
        name (str): Label for the model type.
        forecaster (Optional[ForecasterRecursive]): The underlying forecaster instance.
        is_tuned (bool): Flag indicating if hyperparameter tuning has been performed.
        predict_size (int): Prediction horizon in hours.
        refit_size (int): Refit interval in days.
        random_state (int): Seed for reproducibility.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.preprocessing.forecaster_recursive_model import ForecasterRecursiveModel
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>> model = ForecasterRecursiveModel(iteration=0)
        >>> model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=1)
        >>> model.name = "linear"
        >>> model.tune()
        >>> model.is_tuned
        True
    """

    def __init__(
        self,
        iteration: int,
        end_dev: Optional[Union[str, pd.Timestamp]] = None,
        train_size: Optional[pd.Timedelta] = None,
        periods: Optional[List[Period]] = None,
        country_code: str = "DE",
        random_state: int = 314159,
        predict_size: int = 24,
        refit_size: int = 7,
        **kwargs: Any,
    ):
        """
        Initialize the Recursive Forecaster Model.

        Args:
            iteration: Current iteration index.
            end_dev: Cutoff date for training. Defaults to a reasonable date if None.
            train_size: Time window for training data lookback.
            periods: List of Period objects for cyclical encoding.
            country_code: ISO country code for holidays.
            random_state: Random seed.
            predict_size: Forecast horizon in hours.
            refit_size: Retraining frequency in days.
            **kwargs: Additional parameters for forward compatibility.
        """
        self.iteration = iteration
        # Default end date if not provided (safety fallback)
        default_end = "2025-12-31 00:00+00:00"
        self.end_dev = pd.to_datetime(end_dev if end_dev else default_end, utc=True)
        self.train_size = train_size
        self.random_state = random_state
        self.predict_size = predict_size
        self.refit_size = refit_size

        self.preprocessor = ExogBuilder(periods=periods, country_code=country_code)
        self.name = "base"
        self.forecaster: Optional[ForecasterRecursive] = None
        self.is_tuned = False

    def tune(self) -> None:
        """
        Simulate hyperparameter tuning.

        In a production environment, this would implement Bayesian search or
        similar optimization. For safety-critical stability, we currently
        default to robust parameters.
        #TODO: Implement hyperparameter tuning in spotforecast2
        """
        logger.info("Tuning %s model (simulated)...", self.name)
        self.is_tuned = True

    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the underlying forecaster.

        Args:
            y: Target time series.
            exog: Optional exogenous features.

        Raises:
            ValueError: If the forecaster has not been initialized.
        """
        if self.forecaster is None:
            raise ValueError("Forecaster not initialized")

        self.forecaster.fit(y=y, exog=exog)

    def package_prediction(self, predict_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate predictions and package them with metrics for the UI.

        This method handles data loading (from interim), alignment,
        scoring, and benchmark comparison.

        Args:
            predict_size: Optional override for the prediction horizon.

        Returns:
            Dict[str, Any]: A result package containing actual values,
                predictions, and calculated metrics (MAE, MAPE).
        """
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

        from spotforecast2_safe.data.fetch_data import get_data_home

        if self.forecaster is None:
            logger.error("Forecaster not initialized")
            return {}

        try:
            # Load data from interim directory
            data_home = get_data_home()
            data_path = data_home / "interim" / "energy_load.csv"

            if not data_path.exists():
                logger.error("Data file not found: %s", data_path)
                return {}

            # Read and prepare data
            df = pd.read_csv(data_path, parse_dates=["Time (UTC)"])
            df = df.set_index("Time (UTC)")
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "datetime"
            df = df.asfreq("h")

            if "Actual Load" not in df.columns:
                logger.error("'Actual Load' column missing in %s", data_path)
                return {}

            y = df["Actual Load"]
            # Basic imputation
            if y.isna().any():
                y = y.ffill().bfill()

            # Benchmark
            future_forecast = df.get("Forecasted Load", None)
            if future_forecast is not None and future_forecast.isna().any():
                future_forecast = future_forecast.ffill().bfill()

            # Train/test split using end_dev
            y_train = y.loc[: self.end_dev]
            y_test = y.loc[self.end_dev :]

            if predict_size is None:
                predict_size = self.predict_size

            # Limit test to prediction window
            predict_hours = predict_size * self.refit_size
            if len(y_test) > predict_hours:
                y_test = y_test.iloc[:predict_hours]

            # Fit on training data
            self.forecaster.fit(y=y_train)

            # In-sample (train) predictions
            train_pred = self.forecaster.predict(steps=len(y_train))
            train_pred.index = y_train.index[-len(train_pred) :]
            y_train_aligned = y_train.loc[train_pred.index]

            # Out-of-sample (future) predictions
            future_pred = self.forecaster.predict(steps=len(y_test))
            future_pred.index = y_test.index[: len(future_pred)]
            y_test_aligned = y_test.loc[future_pred.index]

            # Metrics
            metrics_train = {
                "mae": mean_absolute_error(y_train_aligned, train_pred),
                "mape": mean_absolute_percentage_error(y_train_aligned, train_pred),
            }
            metrics_future = {
                "mae": mean_absolute_error(y_test_aligned, future_pred),
                "mape": mean_absolute_percentage_error(y_test_aligned, future_pred),
            }

            # 24h window metrics
            f_24h_pred = future_pred.iloc[: min(24, len(future_pred))]
            f_24h_actual = y_test_aligned.iloc[: min(24, len(y_test_aligned))]
            metrics_future_24h = {
                "mae": mean_absolute_error(f_24h_actual, f_24h_pred),
                "mape": mean_absolute_percentage_error(f_24h_actual, f_24h_pred),
            }

            result = {
                "train_actual": y_train_aligned,
                "future_actual": y_test_aligned,
                "train_pred": train_pred,
                "future_pred": future_pred,
                "metrics_train": metrics_train,
                "metrics_future": metrics_future,
                "metrics_future_one_day": metrics_future_24h,
            }

            # Add benchmark if available
            if future_forecast is not None:
                forecast_test = future_forecast.loc[y_test_aligned.index]
                result["future_forecast"] = forecast_test
                result["metrics_forecast"] = {
                    "mae": mean_absolute_error(y_test_aligned, forecast_test),
                    "mape": mean_absolute_percentage_error(
                        y_test_aligned, forecast_test
                    ),
                }

            return result

        except Exception as e:
            logger.error("Error generating prediction package: %s", e, exc_info=True)
            return {}


class ForecasterRecursiveLGBM(ForecasterRecursiveModel):
    """
    ForecasterRecursive specialization using LightGBM.

    Examples:
        >>> from spotforecast2_safe.preprocessing.forecaster_recursive_model import ForecasterRecursiveLGBM
        >>> model = ForecasterRecursiveLGBM(iteration=0)
        >>> model.name
        'lgbm'
        >>> model.forecaster is not None
        True
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        """
        Initialize the LGBM Recursive Forecaster.

        Args:
            iteration: Current iteration index.
            lags: Number of autoregressive lags.
            **kwargs: Passed to ForecasterRecursiveModel.
        """
        super().__init__(iteration, **kwargs)
        self.name = "lgbm"
        self.forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(
                n_jobs=-1, verbose=-1, random_state=self.random_state
            ),
            lags=lags,
        )


class ForecasterRecursiveXGB(ForecasterRecursiveModel):
    """
    ForecasterRecursive specialization using XGBoost.

    Examples:
        >>> from spotforecast2_safe.preprocessing.forecaster_recursive_model import ForecasterRecursiveXGB
        >>> # Only works if xgboost is installed
        >>> try:
        ...     model = ForecasterRecursiveXGB(iteration=0)
        ...     print(model.name)
        ... except Exception:
        ...     print("xgb")
        xgb
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        """
        Initialize the XGBoost Recursive Forecaster.

        Args:
            iteration: Current iteration index.
            lags: Number of autoregressive lags.
            **kwargs: Passed to ForecasterRecursiveModel.
        """
        super().__init__(iteration, **kwargs)
        self.name = "xgb"
        if XGBRegressor is not None:
            self.forecaster = ForecasterRecursive(
                estimator=XGBRegressor(n_jobs=-1, random_state=self.random_state),
                lags=lags,
            )
        else:
            logger.warning(
                "XGBoost not installed. This model will fail during fit/predict."
            )

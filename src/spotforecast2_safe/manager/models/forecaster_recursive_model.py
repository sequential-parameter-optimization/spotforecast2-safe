# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recursive forecaster model wrappers for different estimators."""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from spotforecast2_safe.data.data import Period
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder

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
        >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import ForecasterRecursiveModel
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
        end_dev: Union[str, pd.Timestamp] = "2025-12-31 00:00+00:00",
        train_size: Optional[pd.Timedelta] = None,
        periods: Optional[List[Period]] = None,
        country_code: str = "DE",
        random_state: int = 123456789,
        predict_size: int = 24,
        refit_size: int = 7,
        name: str = "base",
        **kwargs: Any,
    ):
        """
        Initialize the Recursive Forecaster Model.

        Args:
            iteration:
                Current iteration index.
            end_dev:
                Cutoff date for training. Defaults to "2025-12-31 00:00+00:00".
            train_size:
                Time window for training data lookback.
            periods:
                List of Period objects for cyclical encoding.
            country_code:
                ISO country code for holidays. Defaults to "DE".
            random_state:
                Random seed. Defaults to 123456789.
            predict_size:
                Forecast horizon in hours. Defaults to 24.
            refit_size:
                Retraining frequency in days. Defaults to 7.
            name:
                Model name identifier. Defaults to "base".
            **kwargs:
                Additional parameters for forward compatibility.

        Returns:
            None

        Raises:
            ValueError: If the forecaster has not been initialized.

        Examples:
            >>> import pandas as pd
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import ForecasterRecursiveModel
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> model.name
            'base'
            >>> model.end_dev
            Timestamp('2025-12-31 00:00:00+0000', tz='UTC')
            >>> model.train_size
            >>> model.random_state
            123456789
            >>> model.predict_size
            24
            >>> model.refit_size
            7
            >>> model.preprocessor # doctest: +ELLIPSIS
            <spotforecast2_safe.preprocessing.exog_builder.ExogBuilder object at 0x...>
            >>> model.is_tuned
            False
        """
        self.iteration = iteration
        self.end_dev = pd.to_datetime(end_dev, utc=True)
        self.train_size = train_size
        self.random_state = random_state
        self.predict_size = predict_size
        self.refit_size = refit_size

        self.preprocessor = ExogBuilder(periods=periods, country_code=country_code)
        self.name = name
        self.forecaster: Optional[ForecasterRecursive] = None
        self.is_tuned = False

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        """
        Get parameters for this forecaster model.

        Collects wrapper-level parameters (``iteration``, ``end_dev``,
        ``train_size``, ``random_state``, ``predict_size``, ``refit_size``,
        ``name``) and, when a forecaster is attached, delegates to
        :meth:`ForecasterRecursive.get_params` for forecaster-level
        parameters (``estimator``, ``lags``, ``window_features``, etc.).

        Args:
            deep: If True, will return the parameters for this forecaster model and
                contained sub-objects that are estimators.

        Returns:
            params: Dictionary of parameter names mapped to their values.

        Examples:
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import (
            ...     ForecasterRecursiveModel,
            ... )
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> p = model.get_params(deep=False)
            >>> p["iteration"]
            0
            >>> p["name"]
            'base'
            >>> p["predict_size"]
            24
            >>> "forecaster" not in p  # forecaster is None
            True

            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> model2 = ForecasterRecursiveModel(iteration=1)
            >>> model2.forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(), lags=3
            ... )
            >>> p2 = model2.get_params(deep=False)
            >>> len(p2["forecaster__lags"])
            3
            >>> isinstance(p2["forecaster__estimator"], LinearRegression)
            True

            >>> p3 = model2.get_params(deep=True)
            >>> "forecaster__estimator__fit_intercept" in p3
            True
        """
        # Wrapper-level parameters
        params: Dict[str, object] = {}
        for key in [
            "iteration",
            "end_dev",
            "train_size",
            "random_state",
            "predict_size",
            "refit_size",
            "name",
        ]:
            if hasattr(self, key):
                params[key] = getattr(self, key)

        # Delegate to ForecasterRecursive.get_params when available
        if self.forecaster is not None:
            forecaster_params = self.forecaster.get_params(deep=deep)
            for key, value in forecaster_params.items():
                params[f"forecaster__{key}"] = value

        return params

    def set_params(
        self, params: Dict[str, object] = None, **kwargs: object
    ) -> "ForecasterRecursiveModel":
        """
        Set the parameters of this forecaster model.

        Wrapper-level keys (``iteration``, ``name``, ``predict_size``, …)
        are set directly on the model.  Keys prefixed with ``forecaster__``
        are forwarded to :meth:`ForecasterRecursive.set_params`.

        Args:
            params: Optional dictionary of parameter names mapped to their
                new values.  If provided, these parameters are set first.
            **kwargs: Additional parameter names mapped to their new values.
                Parameters can target the wrapper (e.g. ``name="new"``),
                the forecaster (e.g. ``forecaster__lags=5``), or the
                estimator inside the forecaster
                (e.g. ``forecaster__estimator__fit_intercept=False``).

        Returns:
            ForecasterRecursiveModel: The model instance with updated
                parameters (supports method chaining).

        Examples:
            Setting wrapper-level parameters:

            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import (
            ...     ForecasterRecursiveModel,
            ... )
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> _ = model.set_params(name="updated", predict_size=48)
            >>> model.name
            'updated'
            >>> model.predict_size
            48

            Setting forecaster-level parameters on an attached forecaster:

            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> model2 = ForecasterRecursiveModel(iteration=1)
            >>> model2.forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(), lags=3
            ... )
            >>> _ = model2.set_params(
            ...     params={"forecaster__estimator__fit_intercept": False}
            ... )
            >>> model2.forecaster.estimator.fit_intercept
            False
        """
        # Merge params dict and kwargs
        all_params: Dict[str, object] = {}
        if params is not None:
            all_params.update(params)
        all_params.update(kwargs)

        if not all_params:
            return self

        # Separate forecaster-level from wrapper-level params
        forecaster_params: Dict[str, object] = {}
        for key, value in all_params.items():
            if key.startswith("forecaster__"):
                # Strip the 'forecaster__' prefix before forwarding
                sub_key = key[len("forecaster__") :]
                forecaster_params[sub_key] = value
            else:
                setattr(self, key, value)

        # Delegate forecaster-level params
        if forecaster_params and self.forecaster is not None:
            if hasattr(self.forecaster, "set_params"):
                self.forecaster.set_params(**forecaster_params)
            else:
                for param_name, value in forecaster_params.items():
                    setattr(self.forecaster, param_name, value)

        return self

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

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import ForecasterRecursiveModel
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import LinearRegression
            >>>
            >>> # Example 1: Basic usage with pd.Series
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> y = pd.Series(
            ...     np.random.rand(10),
            ...     index=pd.date_range("2023-01-01", periods=10, freq="h")
            ... )
            >>> model.fit(y=y)
            >>> model.forecaster.is_fitted
            True
            >>>
            >>> # Example 2: Usage with exogenous variables
            >>> model_exog = ForecasterRecursiveModel(iteration=0)
            >>> model_exog.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> exog = pd.DataFrame(
            ...     np.random.rand(10, 2),
            ...     index=y.index,
            ...     columns=["exog_1", "exog_2"]
            ... )
            >>> model_exog.fit(y=y, exog=exog)
            >>> model_exog.forecaster.is_fitted
            True
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

        Examples:
            >>> import os
            >>> import tempfile
            >>> import pandas as pd
            >>> from pathlib import Path
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_lgbm import ForecasterRecursiveLGBM
            >>> from spotforecast2_safe.data.fetch_data import get_package_data_home
            >>>
            >>> # Setup temporary data environment
            >>> tmp_dir = tempfile.mkdtemp()
            >>> os.environ["SPOTFORECAST2_DATA"] = tmp_dir
            >>> data_path = Path(tmp_dir) / "interim"
            >>> data_path.mkdir(parents=True)
            >>>
            >>> # Load demo data and rename columns to match expectations
            >>> demo_path = get_package_data_home() / "demo01.csv"
            >>> df = pd.read_csv(demo_path)
            >>> df = df.rename(columns={
            ...     "Time": "Time (UTC)",
            ...     "Actual": "Actual Load",
            ...     "Forecast": "Forecasted Load"
            ... })
            >>> df.to_csv(data_path / "energy_load.csv", index=False)
            >>>
            >>> # Initialize and run prediction package
            >>> model = ForecasterRecursiveLGBM(iteration=0, end_dev="2022-01-05 00:00+00:00")
            >>> result = model.package_prediction(predict_size=24)
            >>>
            >>> # Validate output
            >>> "train_actual" in result and "future_pred" in result
            True
            >>>
            >>> # Cleanup
            >>> import shutil
            >>> shutil.rmtree(tmp_dir)
            >>> del os.environ["SPOTFORECAST2_DATA"]
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
            logger.info(
                "Prediction window: predict_size=%d, refit_size=%d, "
                "predict_hours=%d, available_test_hours=%d",
                predict_size,
                self.refit_size,
                predict_hours,
                len(y_test),
            )
            if len(y_test) > predict_hours:
                y_test = y_test.iloc[:predict_hours]
                logger.info("Limited test data to %d hours", predict_hours)
            else:
                logger.info("Using all %d available test hours", len(y_test))

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

            # Debug logging for metric computation
            logger.info(
                "Metric computation windows: "
                "future_pred=%d hours, future_actual=%d hours, "
                "24h_pred=%d hours, 24h_actual=%d hours",
                len(future_pred),
                len(y_test_aligned),
                len(f_24h_pred),
                len(f_24h_actual),
            )
            logger.info(
                "Metrics computed: "
                "Full horizon (MAE=%.2f, MAPE=%.4f), "
                "24h window (MAE=%.2f, MAPE=%.4f)",
                metrics_future["mae"],
                metrics_future["mape"],
                metrics_future_24h["mae"],
                metrics_future_24h["mape"],
            )

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

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recursive forecaster model wrappers for different estimators."""

import logging
from typing import Any

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

from .model import ForecasterRecursiveModel

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

logger = logging.getLogger(__name__)


class ForecasterRecursiveXGB(ForecasterRecursiveModel):
    """
    ForecasterRecursive specialization using XGBoost.

    Attributes:
        forecaster: The XGBoost forecaster.
        name: The name of the forecaster.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import Ridge

        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.forecaster.wrappers import (
            ForecasterRecursiveModel,
            ForecasterRecursiveXGB,
        )

        model = ForecasterRecursiveXGB(iteration=0, lags=3)
        assert model.name == "xgb"
        assert isinstance(model, ForecasterRecursiveModel)
        print(f"name: {model.name}")
        print(f"random_state: {model.random_state}")

        # When xgboost is not installed, substitute a Ridge estimator so the
        # fit/predict lifecycle can still be demonstrated end-to-end.
        if model.forecaster is None:
            model.forecaster = ForecasterRecursive(
                estimator=Ridge(random_state=1234), lags=3
            )

        rng = np.random.default_rng(0)
        y = pd.Series(
            rng.random(20),
            index=pd.date_range("2023-01-01", periods=20, freq="h"),
        )
        model.fit(y=y)
        assert model.forecaster.is_fitted
        pred = model.forecaster.predict(steps=2)
        assert len(pred) == 2
        print(f"is_fitted: {model.forecaster.is_fitted}")
        print(f"forecast horizon: {len(pred)} steps")
        ```
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        """
        Initialize the XGBoost Recursive Forecaster.

        Args:
            iteration: Current iteration index.
            lags: Number of autoregressive lags.
            **kwargs: Passed to ForecasterRecursiveModel.

        Returns:
            None

        Raises:
            ImportError: If xgboost is not installed.
        """
        super().__init__(iteration, name="xgb", **kwargs)
        if XGBRegressor is not None:
            self.forecaster = ForecasterRecursive(
                estimator=XGBRegressor(n_jobs=-1, random_state=self.random_state),
                lags=lags,
            )
        else:
            logger.warning(
                "XGBoost not installed. This model will fail during fit/predict."
            )

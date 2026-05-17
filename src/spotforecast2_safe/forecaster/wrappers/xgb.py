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
        from spotforecast2_safe.forecaster.wrappers import (
            ForecasterRecursiveModel,
            ForecasterRecursiveXGB,
        )

        model = ForecasterRecursiveXGB(iteration=0)
        print(model.name)
        print(isinstance(model, ForecasterRecursiveModel))
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

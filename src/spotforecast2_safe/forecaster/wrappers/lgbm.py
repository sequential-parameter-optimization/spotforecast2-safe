# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recursive forecaster wrapper using LightGBM."""

import logging
from typing import Any

from lightgbm import LGBMRegressor

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

from .model import ForecasterRecursiveModel

logger = logging.getLogger(__name__)


class ForecasterRecursiveLGBM(ForecasterRecursiveModel):
    """
    ForecasterRecursive specialization using LightGBM.

    Examples:
        ```{python}
        from spotforecast2_safe.forecaster.wrappers import (
            ForecasterRecursiveLGBM,
            ForecasterRecursiveModel,
        )

        model = ForecasterRecursiveLGBM(iteration=0)
        print(model.name)
        print(isinstance(model, ForecasterRecursiveModel))
        print(model.forecaster is not None)
        ```
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        """
        Initialize the LGBM Recursive Forecaster.

        Args:
            iteration: Current iteration index.
            lags: Number of autoregressive lags.
            **kwargs: Passed to ForecasterRecursiveModel.
        """
        super().__init__(iteration, name="lgbm", **kwargs)
        self.forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(
                n_jobs=-1,
                verbose=-1,
                random_state=self.random_state,
                deterministic=True,
                force_col_wise=True,
            ),
            lags=lags,
        )

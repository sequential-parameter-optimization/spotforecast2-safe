# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recursive forecaster wrapper using CatBoost."""

import logging
from typing import Any

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

from .model import ForecasterRecursiveModel

# CatBoost is an optional dependency. It is deliberately kept off the core
# dependency list to minimise the CVE / supply-chain surface (MODEL_CARD.md),
# mirroring the XGBoost wrapper. When it is absent the import degrades to
# ``None`` and any later fit/predict call fails explicitly rather than silently.
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

logger = logging.getLogger(__name__)


class ForecasterRecursiveCatBoost(ForecasterRecursiveModel):
    """
    ForecasterRecursive specialization using CatBoost.

    CatBoost is an optional dependency. When it is not installed the wrapper
    still constructs, but leaves ``forecaster`` as ``None`` and logs a
    warning; any subsequent fit/predict call then fails explicitly instead of
    silently degrading.

    The estimator is pinned for bit-level reproducibility and to avoid
    filesystem side effects: a fixed ``random_seed``, single-threaded
    summation (``thread_count=1``) so the result does not depend on the host
    core count, and ``allow_writing_files=False`` so training never writes a
    ``catboost_info`` directory.

    Attributes:
        forecaster: The CatBoost forecaster, or ``None`` if CatBoost is not
            installed.
        name: The name of the forecaster (``"catboost"``).

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import Ridge

        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.forecaster.wrappers import (
            ForecasterRecursiveCatBoost,
            ForecasterRecursiveModel,
        )

        model = ForecasterRecursiveCatBoost(iteration=0, lags=3)
        assert model.name == "catboost"
        assert isinstance(model, ForecasterRecursiveModel)
        print(f"name: {model.name}")
        print(f"random_state: {model.random_state}")

        # When catboost is not installed, substitute a Ridge estimator so the
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
        Initialize the CatBoost Recursive Forecaster.

        Args:
            iteration: Current iteration index.
            lags: Number of autoregressive lags.
            **kwargs: Passed to ForecasterRecursiveModel.
        """
        super().__init__(iteration, name="catboost", **kwargs)
        if CatBoostRegressor is not None:
            self.forecaster = ForecasterRecursive(
                estimator=CatBoostRegressor(
                    random_seed=self.random_state,
                    thread_count=1,
                    verbose=False,
                    allow_writing_files=False,
                ),
                lags=lags,
            )
        else:
            logger.warning(
                "CatBoost not installed. This model will fail during fit/predict."
            )

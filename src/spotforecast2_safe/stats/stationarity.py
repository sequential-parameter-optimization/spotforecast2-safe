# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Augmented Dickey-Fuller stationarity test."""

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from spotforecast2_safe.preprocessing.checking import check_y


def augmented_dickey_fuller(y: pd.Series, *, autolag: str = "AIC") -> pd.Series:
    """Run an Augmented Dickey-Fuller test on a time series.

    Returns a structured ``pd.Series`` instead of printing — callers
    decide how to display the result.

    Args:
        y: Series to test.  Must be a non-empty pandas Series without
            missing values (validated via `check_y`).
        autolag: Lag-selection criterion forwarded to
            `statsmodels.tsa.stattools.adfuller`.  Defaults to ``"AIC"``.

    Returns:
        ``pd.Series`` with the index
        ``["statistic", "p_value", "n_lags", "n_obs", "critical_1%",
        "critical_5%", "critical_10%"]``.

    Raises:
        TypeError: If ``y`` is not a pandas Series.
        ValueError: If ``y`` contains missing values.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd

        from spotforecast2_safe.stats.stationarity import (
            augmented_dickey_fuller,
        )

        rng = np.random.default_rng(0)
        y = pd.Series(rng.standard_normal(500))
        out = augmented_dickey_fuller(y)
        print(out["p_value"] < 0.05)
        ```
    """
    check_y(y)

    statistic, p_value, n_lags, n_obs, critical_values, _ = adfuller(
        y.to_numpy(), autolag=autolag
    )

    return pd.Series(
        {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n_lags": int(n_lags),
            "n_obs": int(n_obs),
            "critical_1%": float(critical_values["1%"]),
            "critical_5%": float(critical_values["5%"]),
            "critical_10%": float(critical_values["10%"]),
        }
    )

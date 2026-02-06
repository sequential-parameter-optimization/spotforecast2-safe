from __future__ import annotations
import numpy as np
import pandas as pd

from spotforecast2_safe.forecaster.utils import check_optional_dependency

try:
    from statsmodels.tsa.stattools import acf, pacf
except ImportError:
    acf, pacf = None, None


def calculate_lag_autocorrelation(
    data: pd.Series | pd.DataFrame,
    n_lags: int = 50,
    last_n_samples: int | None = None,
    sort_by: str = "partial_autocorrelation_abs",
    acf_kwargs: dict[str, object] = {},
    pacf_kwargs: dict[str, object] = {},
) -> pd.DataFrame:
    """
    Calculate autocorrelation and partial autocorrelation for a time series.

    This is a wrapper around statsmodels.acf and statsmodels.pacf.

    Args:
        data: Time series to calculate autocorrelation. If a DataFrame is provided,
            it must have exactly one column.
        n_lags: Number of lags to calculate autocorrelation. Default is 50.
        last_n_samples: Number of most recent samples to use. If None, use the entire
            series. Note that partial correlations can only be computed for lags up to
            50% of the sample size. For example, if the series has 10 samples,
            n_lags must be less than or equal to 5. This parameter is useful
            to speed up calculations when the series is very long. Default is None.
        sort_by: Sort results by lag, partial_autocorrelation_abs,
            partial_autocorrelation, autocorrelation_abs or autocorrelation.
            Default is partial_autocorrelation_abs.
        acf_kwargs: Optional arguments to pass to statsmodels.tsa.stattools.acf.
            Default is {}.
        pacf_kwargs: Optional arguments to pass to statsmodels.tsa.stattools.pacf.
            Default is {}.

    Returns:
        DataFrame with columns: lag, partial_autocorrelation_abs,
            partial_autocorrelation, autocorrelation_abs, autocorrelation.

    Raises:
        TypeError: If data is not a pandas Series or DataFrame with a single column.
        ValueError: If data is a DataFrame with more than one column.
        TypeError: If n_lags is not a positive integer.
        TypeError: If last_n_samples is not None and not a positive integer.
        ValueError: If sort_by is not one of the valid options.

    Examples:
        Calculate autocorrelation for a simple Series:

        >>> import pandas as pd
        >>> from spotforecast.stats.autocorrelation import calculate_lag_autocorrelation
        >>>
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = calculate_lag_autocorrelation(data=data, n_lags=4)
        >>> result.head()
           lag  partial_autocorrelation_abs  partial_autocorrelation  autocorrelation_abs  autocorrelation
        0    1                     0.999998                 0.999998             1.000000         1.000000
        1    2                     0.000002                -0.000002             0.645497         0.645497
        2    3                     0.000002                 0.000002             0.298549         0.298549
        3    4                     0.000001                -0.000001             0.068719         0.068719

        Calculate autocorrelation using only the last 8 samples:

        >>> import pandas as pd
        >>> from spotforecast2.stats.autocorrelation import calculate_lag_autocorrelation
        >>>
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = calculate_lag_autocorrelation(
        ...     data=data,
        ...     n_lags=3,
        ...     last_n_samples=8
        ... )
        >>> result.shape
        (3, 5)

        Calculate autocorrelation from a DataFrame with a single column:

        >>> import pandas as pd
        >>> from spotforecast.stats.autocorrelation import calculate_lag_autocorrelation
        >>>
        >>> data = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        >>> result = calculate_lag_autocorrelation(data=data, n_lags=4)
        >>> result.shape
        (4, 5)

        Sort results by autocorrelation in descending order:

        >>> import pandas as pd
        >>> from spotforecast.stats.autocorrelation import calculate_lag_autocorrelation
        >>>
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = calculate_lag_autocorrelation(
        ...     data=data,
        ...     n_lags=4,
        ...     sort_by='autocorrelation'
        ... )
        >>> result[['lag', 'autocorrelation']].head()
           lag  autocorrelation
        0    1         1.000000
        1    2         0.645497
        2    3         0.298549
        3    4         0.068719

    """
    check_optional_dependency("statsmodels")

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"`data` must be a pandas Series or a DataFrame with a single column. "
            f"Got {type(data)}."
        )
    if isinstance(data, pd.DataFrame) and data.shape[1] != 1:
        raise ValueError(
            f"If `data` is a DataFrame, it must have exactly one column. "
            f"Got {data.shape[1]} columns."
        )
    if not isinstance(n_lags, int) or n_lags <= 0:
        raise TypeError(f"`n_lags` must be a positive integer. Got {n_lags}.")

    if last_n_samples is not None:
        if not isinstance(last_n_samples, int) or last_n_samples <= 0:
            raise TypeError(
                f"`last_n_samples` must be a positive integer. Got {last_n_samples}."
            )
        data = data.iloc[-last_n_samples:]

    if sort_by not in [
        "lag",
        "partial_autocorrelation_abs",
        "partial_autocorrelation",
        "autocorrelation_abs",
        "autocorrelation",
    ]:
        raise ValueError(
            "`sort_by` must be 'lag', 'partial_autocorrelation_abs', 'partial_autocorrelation', "
            "'autocorrelation_abs' or 'autocorrelation'."
        )

    series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    if series.nunique() <= 1:
        acf_values = np.full(n_lags + 1, np.nan)
        acf_values[0] = 1.0
        pacf_values = np.zeros(n_lags + 1)
        pacf_values[0] = 1.0
    else:
        pacf_values = pacf(data, nlags=n_lags, **pacf_kwargs)
        acf_values = acf(data, nlags=n_lags, **acf_kwargs)

    results = pd.DataFrame(
        {
            "lag": range(n_lags + 1),
            "partial_autocorrelation_abs": np.abs(pacf_values),
            "partial_autocorrelation": pacf_values,
            "autocorrelation_abs": np.abs(acf_values),
            "autocorrelation": acf_values,
        }
    ).iloc[1:]

    if sort_by == "lag":
        results = results.sort_values(by=sort_by, ascending=True).reset_index(drop=True)
    else:
        results = results.sort_values(by=sort_by, ascending=False).reset_index(
            drop=True
        )

    return results

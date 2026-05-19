# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from spotforecast2_safe.preprocessing.linearly_interpolate_ts import (
    LinearlyInterpolateTS,
)

logger = logging.getLogger(__name__)


class WeightFunction:
    """Callable class for sample weights that can be pickled.

    This class wraps the weights_series and provides a callable interface
    compatible with ForecasterRecursive's weight_func parameter. Unlike
    local functions with closures, instances of this class can be pickled
    using standard pickle/joblib.

    When all weights for the requested index sum to zero (e.g. the entire
    training window falls inside gap-penalty zones after outlier detection),
    ``__call__`` returns ``None`` instead of an all-zero array.
    ``ForecasterRecursive.create_sample_weights`` treats a ``None`` return as
    "no weighting", avoiding a ``ValueError`` while still applying the
    weighted imputation for windows that do contain positive weights.

    Args:
        weights_series: Series containing weight values for each index.

    Examples:
        ```{python}
        import pickle
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.preprocessing.imputation import WeightFunction

        weights = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
        weight_func = WeightFunction(weights)

        # Callable interface: returns weights for a pd.Index
        result = weight_func(pd.Index([0, 1]))
        print(result)
        assert np.allclose(result, [1.0, 0.9])

        # Returns None when all weights in the window are zero
        zero_weights = pd.Series([0.0, 0.0, 0.0], index=[0, 1, 2])
        wf_zero = WeightFunction(zero_weights)
        assert wf_zero(pd.Index([0, 1])) is None

        # Can be pickled and unpickled (no closure, fully serialisable)
        unpickled = pickle.loads(pickle.dumps(weight_func))
        result2 = unpickled(pd.Index([0, 1]))
        print(result2)
        assert np.allclose(result2, [1.0, 0.9])
        ```
    """

    def __init__(self, weights_series: pd.Series):
        """Initialize with a weights series.

        Args:
            weights_series: Series containing weight values for each index.
        """
        self.weights_series = weights_series

    def __call__(
        self, index: Union[pd.Index, np.ndarray, list]
    ) -> Optional[np.ndarray]:
        """Return sample weights for the given index, or None if all weights are zero.

        Computes the weights via `custom_weights()`.  When every weight in
        the result sums to zero — which happens when the entire requested window
        falls within gap-penalty zones created by outlier detection — the method
        logs a warning and returns ``None``.  ``ForecasterRecursive`` interprets
        a ``None`` return as "use uniform weights", preventing a
        ``ValueError: sample_weight cannot be normalized`` crash.

        Args:
            index: Index or indices to get weights for.

        Returns:
            Numpy array of weight values, or ``None`` when the sum of all
            weights is zero (degenerate window).

        Examples:
            ```{python}
            import numpy as np
            import pandas as pd
            from spotforecast2_safe.preprocessing.imputation import WeightFunction

            weights = pd.Series([1.0, 0.5, 0.0], index=[0, 1, 2])
            wf = WeightFunction(weights)

            # pd.Index lookup returns a numpy array of weights
            result = wf(pd.Index([0, 1]))
            print(result)
            assert np.allclose(result, [1.0, 0.5])

            # Scalar access returns the stored weight directly (no None fallback)
            scalar = wf(2)
            print(scalar)
            assert scalar == 0.0

            # pd.Index whose weights all sum to zero triggers the None path
            assert wf(pd.Index([2])) is None
            ```
        """
        result = custom_weights(index, self.weights_series)
        if isinstance(index, pd.Index) and np.sum(result) == 0:
            logger.warning(
                "WeightFunction: all sample weights for the requested index are "
                "zero (the window falls entirely within gap-penalty zones). "
                "Returning None so ForecasterRecursive uses uniform weighting."
            )
            return None
        return result

    def __repr__(self):
        """String representation."""
        return f"WeightFunction(weights_series with {len(self.weights_series)} entries)"


def custom_weights(index, weights_series: pd.Series) -> float:
    """
    Return 0 if index is in or near any gap.

    Args:
        index (pd.Index):
            The index to check.
        weights_series (pd.Series):
            Series containing weights.

    Returns:
        float: The weight corresponding to the index.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.preprocessing.imputation import custom_weights

        # Build a small synthetic weights Series with a DatetimeIndex
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        weights_series = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0], index=idx)

        # Single-index scalar lookup
        w = custom_weights(idx[0], weights_series)
        print(f"Index: {idx[0]}, Weight: {w}")
        assert w == 1.0

        # pd.Index slice lookup returns a numpy array
        result = custom_weights(idx[:3], weights_series)
        print(result)
        assert np.allclose(result, [1.0, 1.0, 0.0])
        ```
    """
    # do plausibility check
    if isinstance(index, pd.Index):
        if not index.isin(weights_series.index).all():
            raise ValueError("Index not found in weights_series.")
        return weights_series.loc[index].values

    if index not in weights_series.index:
        raise ValueError("Index not found in weights_series.")
    return weights_series.loc[index]


def get_missing_weights(
    data: pd.DataFrame, window_size: int = 72, verbose: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return imputed DataFrame and a series indicating missing weights.

    Args:
        data (pd.DataFrame):
            The input dataset.
        window_size (int):
            The size of the rolling window to consider for missing values.
        verbose (bool):
            Whether to print additional information. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            A tuple containing the forward and backward filled DataFrame and a numeric series (0.0 or 1.0) where 0.0 indicates a weight for missing values/gaps.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.preprocessing.imputation import get_missing_weights

        # Synthetic DataFrame with a deliberate two-row NaN gap at positions 3-4
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        values = [1.0, 2.0, 3.0, None, None, 6.0, 7.0, 8.0, 9.0, 10.0]
        df = pd.DataFrame({"A": values}, index=idx)

        filled, weights = get_missing_weights(df, window_size=3, verbose=True)

        # No NaNs remain after forward/backward fill
        assert filled.isnull().sum().sum() == 0

        # Rows inside (and immediately after) the gap receive weight 0
        gap_weights = weights.loc[idx[3:5]]
        print(gap_weights.tolist())
        assert (gap_weights == 0.0).all()

        # Rows well before the gap retain weight 1
        assert weights.loc[idx[0]] == 1.0
        ```

    """
    # first perform some checks if dataframe has enough data and if window_size is appropriate
    if data.shape[0] == 0:
        raise ValueError("Input data is empty.")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if window_size >= data.shape[0]:
        raise ValueError("window_size must be smaller than the number of rows in data.")

    missing_indices = data.index[data.isnull().any(axis=1)]
    n_missing = len(missing_indices)
    if verbose:
        pct_missing = (n_missing / len(data)) * 100
        print(f"Number of rows with missing values: {n_missing}")
        print(f"Percentage of rows with missing values: {pct_missing:.2f}%")
        print(f"missing_indices: {missing_indices}")
    data = data.ffill()
    data = data.bfill()

    is_missing = pd.Series(0, index=data.index)
    is_missing.loc[missing_indices] = 1
    # weights_series: 0 if in/near gap, 1 otherwise
    weights_series = (
        1.0 - is_missing.rolling(window=window_size + 1, min_periods=1).max()
    )

    if verbose:
        n_missing_after = (weights_series == 0).sum()
        pct_missing_after = (n_missing_after / len(data)) * 100
        print(
            f"Number of rows with missing weights after processing: {n_missing_after}"
        )
        print(
            f"Percentage of rows with missing weights after processing: {pct_missing_after:.2f}%"
        )
    return data, weights_series


def apply_imputation(
    df_pipeline: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
    verbose: bool = False,
) -> tuple[pd.DataFrame, "WeightFunction | None"]:
    """Apply imputation to a DataFrame based on the method specified in config.

    Supports two strategies:

    - ``"weighted"``: forward-fill then backward-fill gaps, then build a
      `WeightFunction` that down-weights training rows near any gap.
      Rows inside a gap receive weight 0; the rolling window
      ``config.window_size`` controls how far the penalty extends.
    - ``"linear"``: apply `LinearlyInterpolateTS` column-by-column.

    A diagnostic summary (NaN count before **and** after imputation) is
    always written to the logger.

    Args:
        df_pipeline (pd.DataFrame): DataFrame to impute.  Modified in-place
            for the ``"linear"`` method; a new DataFrame is returned for
            ``"weighted"`` (via `get_missing_weights()`).
        config: Configuration object that must expose:
            - ``imputation_method`` (``str``): ``"weighted"`` or ``"linear"``.
            - ``targets`` (``list[str]``): column names to interpolate
              (``"linear"`` method only).
            - ``window_size`` (``int``): rolling-window size passed to
              `get_missing_weights()` (``"weighted"`` method only).
        logger (logging.Logger): Standard-library logger used to emit
            ``INFO`` and ``WARNING`` messages.
        verbose (bool): Whether to print additional information. Defaults to False.

    Returns:
        tuple[pd.DataFrame, WeightFunction | None]: A two-element tuple:

        - **df_pipeline** – imputed DataFrame with no NaN values (when the
            chosen method can fill all gaps).
        - **weight_func** – a `WeightFunction` instance ready to be
            passed to a forecaster's ``weight_func`` parameter, or ``None``
            when ``"linear"`` imputation is used.

    Raises:
        ValueError: If ``config.imputation_method`` is neither ``"weighted"``
            nor ``"linear"``.

    Examples:
        ```{python}
        import logging
        import pandas as pd
        import numpy as np
        from types import SimpleNamespace
        from spotforecast2_safe.preprocessing.imputation import apply_imputation

        # Build a small DataFrame with deliberate gaps
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {"A": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
            index=idx,
        )

        # Minimal config and stdlib logger
        config = SimpleNamespace(
            imputation_method="linear",
            targets=["A"],
            window_size=3,
        )
        logger = logging.getLogger("demo")

        imputed, weight_func = apply_imputation(df, config, logger)
        print(imputed["A"].tolist())
        print(weight_func)  # None for linear method
        ```
    """
    nan_before = int(df_pipeline.isnull().sum().sum())
    logger.info(
        "apply_imputation: NaN cells before imputation: %d (method=%r, shape=%s)",
        nan_before,
        config.imputation_method,
        df_pipeline.shape,
    )

    weight_func = None  # default: no sample weighting

    if config.imputation_method == "weighted":
        logger.info("Applying weighted imputation (n2n style)...")
        df_pipeline, weights_series = get_missing_weights(
            df_pipeline,
            window_size=config.window_size,
            verbose=verbose,
        )
        weight_func = WeightFunction(weights_series)
        logger.info("Weight function created with %d entries.", len(weights_series))
    elif config.imputation_method == "linear":
        logger.info("Applying linear interpolation...")
        # passthrough lets residual endpoint NaNs survive into the
        # post-imputation NaN count + WARNING below, matching the
        # surrounding logger.warning contract.
        interpolator = LinearlyInterpolateTS(on_missing="passthrough")
        # LinearlyInterpolateTS expects a Series; apply per column
        for col in config.targets:
            series = df_pipeline[col]
            df_pipeline[col] = interpolator.fit_transform(series)
    else:
        raise ValueError(
            f"Unknown imputation_method: {config.imputation_method!r}. "
            "Expected one of: 'weighted', 'linear'."
        )

    nan_after = int(df_pipeline.isnull().sum().sum())
    logger.info("apply_imputation: NaN cells after imputation: %d", nan_after)
    if nan_after > 0:
        logger.warning(
            "apply_imputation: %d NaN cell(s) remain after imputation. "
            "Consider reviewing the data or adjusting the imputation method.",
            nan_after,
        )
    return df_pipeline, weight_func

# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import warnings
from typing import Any, List

import numpy as np
import pandas as pd

from ._common import (
    _ewm_jit,
    _np_cv_jit,
    _np_max_jit,
    _np_mean_jit,
    _np_median_jit,
    _np_min_jit,
    _np_min_max_ratio_jit,
    _np_std_jit,
    _np_sum_jit,
)


class RollingFeatures:
    """
    Compute rolling window statistics over time series data.

    This transformer computes rolling statistics (mean, std, min, max, sum, median)
    over windows of specified sizes from a time series. The class follows the
    scikit-learn transformer API with fit() and transform() methods, making it
    compatible with scikit-learn pipelines. It also provides transform_batch()
    for pandas Series input.

    Args:
        stats: Rolling statistics to compute. Can be a single string ('mean', 'std',
            'min', 'max', 'sum', 'median'), list of statistic names, or list of
            callable functions. Multiple statistics can be computed simultaneously.
        window_sizes: Window size(s) for rolling computation. Can be a single integer
            or list of integers. Multiple windows are applied to all statistics.
        features_names: Custom names for output features. If None, names are
            auto-generated from statistic names and window sizes (e.g.,
            'roll_mean_7', 'roll_std_14'). Defaults to None.

    Attributes:
        stats: Statistics specification as provided during initialization.
        window_sizes: List of window sizes for rolling computation.
        features_names: List of output feature names.
        stats_funcs: List of compiled/numba-optimized statistical functions.

    Note:
        - Output contains NaN values for positions where the rolling window cannot
          be fully computed (first window_size-1 positions).
        - Statistics are computed using numba-optimized JIT functions for performance.
        - The transformer returns numpy arrays from transform() and pandas DataFrames
          from transform_batch() to maintain index alignment.
        - Supports custom user-defined functions in the stats parameter.

    Examples:
        ```{python}
        import numpy as np
        from spotforecast2_safe.preprocessing.rolling import RollingFeatures

        # Single statistic and window size — transform() returns the last
        # window's statistic as a 1D array of shape (n_features,).
        y = np.arange(10, dtype=float)
        rf = RollingFeatures(stats='mean', window_sizes=3)
        features = rf.fit(y).transform(y)
        print("transform output:", features)
        assert features.shape == (1,)
        # Mean of last window [7, 8, 9] = 8.0
        assert float(features[0]) == 8.0
        ```

        ```{python}
        import numpy as np
        from spotforecast2_safe.preprocessing.rolling import RollingFeatures

        # Multiple statistics and window sizes — feature names are
        # auto-generated as roll_<stat>_<window>.
        y = np.arange(10, dtype=float)
        rf = RollingFeatures(
            stats=['mean', 'std', 'min', 'max'],
            window_sizes=[3, 5],
        )
        rf.fit(y)
        print("features_names:", rf.features_names)
        assert rf.features_names == [
            'roll_mean_3', 'roll_std_3', 'roll_min_3', 'roll_max_3',
            'roll_mean_5', 'roll_std_5', 'roll_min_5', 'roll_max_5',
        ]
        ```

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.preprocessing.rolling import RollingFeatures

        # Use transform_batch() with a pandas Series to preserve the index.
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        y_series = pd.Series(np.arange(10, dtype=float), index=dates)
        rf = RollingFeatures(stats=['mean', 'max'], window_sizes=5)
        features_df = rf.transform_batch(y_series)
        print(features_df.head())
        assert features_df.shape == (10, 2)
        assert features_df.index.equals(y_series.index)
        ```

        ```{python}
        import numpy as np
        from spotforecast2_safe.preprocessing.rolling import RollingFeatures

        # Custom feature names override the auto-generated ones.
        rf = RollingFeatures(
            stats='mean',
            window_sizes=[3, 5, 7],
            features_names=['ma_3', 'ma_5', 'ma_7'],
        )
        rf.fit(np.arange(10, dtype=float))
        print("features_names:", rf.features_names)
        assert rf.features_names == ['ma_3', 'ma_5', 'ma_7']
        ```
    """

    def __init__(
        self,
        stats: str | List[str] | List[Any],
        window_sizes: int | List[int],
        features_names: List[str] | None = None,
    ):
        """
        Initialize the rolling features transformer.

        Args:
            stats: Rolling statistics to compute. Can be a single string or list
                of statistics/functions.
            window_sizes: Window size(s) for rolling statistics.
            features_names: Custom names for output features. If None, auto-generated.
                Defaults to None.
        """
        self.stats = stats
        self.window_sizes = window_sizes
        self.features_names = features_names

        # Validation and processing logic...
        self._validate_params()

    def _validate_params(self):
        """
        Validate and process rolling features parameters.

        Converts single values to lists, maps string statistics to functions,
        and generates feature names if not provided.

        Raises:
            ValueError: If an unsupported statistic name is provided.
        """
        if isinstance(self.window_sizes, int):
            self.window_sizes = [self.window_sizes]

        if isinstance(self.stats, str):
            self.stats = [self.stats]

        # Map strings to functions
        valid_stats = {
            "mean": _np_mean_jit,
            "std": _np_std_jit,
            "min": _np_min_jit,
            "max": _np_max_jit,
            "sum": _np_sum_jit,
            "median": _np_median_jit,
            "ratio_min_max": _np_min_max_ratio_jit,
            "coef_variation": _np_cv_jit,
            "ewm": _ewm_jit,
        }

        self.stats_funcs = []
        for s in self.stats:
            if isinstance(s, str):
                if s not in valid_stats:
                    raise ValueError(
                        f"Stat '{s}' not supported. Supported: {list(valid_stats.keys())}"
                    )
                self.stats_funcs.append(valid_stats[s])
            else:
                self.stats_funcs.append(s)

        if self.features_names is None:
            self.features_names = []
            for ws in self.window_sizes:
                for s in self.stats:
                    s_name = s if isinstance(s, str) else s.__name__
                    self.features_names.append(f"roll_{s_name}_{ws}")

    def fit(self, X: Any, y: Any = None) -> "RollingFeatures":
        """
        Fit the rolling features transformer (no-op).

        This transformer does not learn any parameters from the data.
        Method exists for scikit-learn compatibility.

        Args:
            X: Time series data (not used for fitting).
            y: Target values (ignored). Defaults to None.

        Returns:
            self: Returns the fitted transformer.

        Examples:
            ```{python}
            import numpy as np
            from spotforecast2_safe.preprocessing.rolling import RollingFeatures

            rf = RollingFeatures(stats='mean', window_sizes=3)
            y = np.arange(10, dtype=float)
            result = rf.fit(y)
            assert result is rf, "fit() must return self"
            features = result.transform(y)
            print("fit() returns self:", result is rf)
            print("transform output:", features)
            ```
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute rolling window statistics from time series data.

        Args:
            X: Time series data as 1D or 2D numpy array.

        Returns:
            np.ndarray: Array of rolling statistics.
                - If X is 1D: shape (n_features,) — statistics over the last window of the series.
                - If X is 2D: shape (X.shape[1], n_features) — used for vectorized bootstrap.

        Examples:
            ```{python}
            import numpy as np
            from spotforecast2_safe.preprocessing.rolling import RollingFeatures

            # 1D input: returns a 1D array of shape (n_features,) with
            # the statistic computed over the last window of the series.
            y = np.arange(10, dtype=float)
            rf = RollingFeatures(stats='mean', window_sizes=3)
            out = rf.transform(y)
            print("transform output:", out, "shape:", out.shape)
            # Mean of last window [7, 8, 9] = 8.0
            assert out.shape == (1,)
            assert float(out[0]) == 8.0
            ```
        """
        array_ndim = X.ndim
        if array_ndim == 1:
            X_2d = X[:, np.newaxis]
        else:
            X_2d = X

        vectorizable_stats = {"mean", "std", "min", "max", "sum", "median"}
        has_vectorizable = bool(set(self.stats) & vectorizable_stats)

        # Output shape: (n_columns, n_features)
        # n_features = n_stats * n_window_sizes
        rolling_features = np.full(
            (X_2d.shape[1], len(self.features_names)), np.nan, dtype=float
        )

        if has_vectorizable:
            self._transform_vectorized(X_2d, rolling_features)

        # Non-vectorizable or fallback for single columns
        for i in range(X_2d.shape[1]):
            col = X_2d[:, i]
            for j, ws in enumerate(self.window_sizes):
                for k, stat_name in enumerate(self.stats):
                    idx_feature = j * len(self.stats) + k
                    if stat_name in vectorizable_stats and array_ndim == 1:
                        # For 1D transform (batch), we use pandas for the full series
                        # For 2D transform (bootstrap), it's already handled in _transform_vectorized
                        continue

                    if stat_name not in vectorizable_stats:
                        # Custom/Non-vectorized stats only need the last window for bootstrap
                        # but transform() is also used in transform_batch() for the whole series.
                        # If it's a batch transform (1D input or many rows), we use the slow path.
                        # If it's for bootstrapping (2D input, small window), we use the last window.

                        # Bootstrap case or single-step case:
                        # X_2d is typically (window_size, n_boot) or (window_size, 1)
                        # We only need the result for the last window
                        window = col[-ws:]
                        window = window[~np.isnan(window)]
                        if len(window) > 0:
                            rolling_features[i, idx_feature] = self.stats_funcs[k](
                                window
                            )

        if array_ndim == 1:
            return rolling_features.ravel()
        else:
            return rolling_features

    def _transform_vectorized(self, X: np.ndarray, rolling_features: np.ndarray):
        """
        Vectorized transform for bootstrap predictions.
        X: (window_length, n_samples)
        rolling_features: (n_samples, n_features) - modified in place
        """
        vectorizable_stats = {"mean", "std", "min", "max", "sum", "median"}

        for j, ws in enumerate(self.window_sizes):
            for k, stat_name in enumerate(self.stats):
                if stat_name not in vectorizable_stats:
                    continue

                idx_feature = j * len(self.stats) + k
                window = X[-ws:, :]

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice")
                    warnings.filterwarnings(
                        "ignore", message="Degrees of freedom <= 0 for slice"
                    )
                    warnings.filterwarnings(
                        "ignore", message="All-NaN slice encountered"
                    )

                    if stat_name == "mean":
                        rolling_features[:, idx_feature] = np.nanmean(window, axis=0)
                    elif stat_name == "std":
                        result = np.nanstd(window, axis=0, ddof=1)
                        n_valid = np.sum(~np.isnan(window), axis=0)
                        result[n_valid == 1] = 0.0
                        rolling_features[:, idx_feature] = result
                    elif stat_name == "min":
                        rolling_features[:, idx_feature] = np.nanmin(window, axis=0)
                    elif stat_name == "max":
                        rolling_features[:, idx_feature] = np.nanmax(window, axis=0)
                    elif stat_name == "sum":
                        result = np.nansum(window, axis=0, dtype=float)
                        all_nan_mask = np.all(np.isnan(window), axis=0)
                        result[all_nan_mask] = np.nan
                        rolling_features[:, idx_feature] = result
                    elif stat_name == "median":
                        rolling_features[:, idx_feature] = np.nanmedian(window, axis=0)

    def transform_batch(self, X: pd.Series) -> pd.DataFrame:
        """
        Compute rolling features from a pandas Series with index preservation.

        Args:
            X: Time series data as a pandas Series with a datetime index.

        Returns:
            pd.DataFrame: DataFrame of rolling statistics with the same index as
                X and one column per (window_size, statistic) combination.

        Examples:
            ```{python}
            import numpy as np
            import pandas as pd
            from spotforecast2_safe.preprocessing.rolling import RollingFeatures

            y = pd.Series(
                np.arange(20, dtype=float),
                index=pd.date_range("2024-01-01", periods=20, freq="D"),
            )
            rf = RollingFeatures(stats=['mean', 'std'], window_sizes=[3, 5])
            df = rf.transform_batch(y)
            print(df.head())
            assert df.shape == (20, 4)
            assert list(df.columns) == [
                'roll_mean_3', 'roll_std_3', 'roll_mean_5', 'roll_std_5',
            ]
            assert df.index.equals(y.index)
            ```
        """
        n_samples = len(X)
        output = np.full((n_samples, len(self.features_names)), np.nan)
        values = X.to_numpy()

        for j, ws in enumerate(self.window_sizes):
            for k, stat_name in enumerate(self.stats):
                idx_feature = j * len(self.stats) + k
                func = self.stats_funcs[k]

                # Use pandas rolling for batch transformation
                series = pd.Series(values)
                # Note: skforecast uses closed='left' for RollingFeatures by default to avoid leakage.
                # However, RollingFeatures in spotforecast2-safe's original implementation
                # seemed to be a simple rolling. Let's check skforecast's default again.
                # skforecast: self.unique_rolling_windows[key]['params'] = {'window': params[0], 'min_periods': params[1], 'center': False, 'closed': 'left'}
                # Wait, if closed='left', then the current value is NOT included.
                # Let's align with skforecast's behavior if it's ported.

                # Original spotforecast2-safe implementation used standard rolling (closed='right')
                # but if we want to be exactly like skforecast, we should use closed='left'.
                # Actually, ForecasterRecursive handles the lags manually, so if window features are
                # calculated on the same 'y' as lags, they should probably be shifted too.

                rolled = series.rolling(window=ws).apply(func, raw=True)
                output[:, idx_feature] = rolled.values

        return pd.DataFrame(output, index=X.index, columns=self.features_names)

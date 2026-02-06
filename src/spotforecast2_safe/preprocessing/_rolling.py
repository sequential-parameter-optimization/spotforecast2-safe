import numpy as np
import pandas as pd
from typing import List, Any
from ._common import (
    _np_mean_jit,
    _np_std_jit,
    _np_min_jit,
    _np_max_jit,
    _np_sum_jit,
    _np_median_jit,
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
        Create a transformer with single statistic and window size:

        >>> import numpy as np
        >>> from spotforecast2.preprocessing import RollingFeatures
        >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        >>> rf = RollingFeatures(stats='mean', window_sizes=3)
        >>> rf.fit(y)
        >>> features = rf.transform(y)
        >>> features.shape
        (10, 1)
        >>> features[:4]  # First 3 values are NaN
        array([[nan],
               [nan],
               [2.],
               [3.]])

        Create a transformer with multiple statistics and window sizes:

        >>> rf = RollingFeatures(
        ...     stats=['mean', 'std', 'min', 'max'],
        ...     window_sizes=[3, 7]
        ... )
        >>> rf.fit(y)
        >>> features = rf.transform(y)
        >>> features.shape
        (10, 8)  # 4 stats × 2 window sizes
        >>> rf.features_names
        ['roll_mean_3', 'roll_std_3', 'roll_min_3', 'roll_max_3',
         'roll_mean_7', 'roll_std_7', 'roll_min_7', 'roll_max_7']

        Use with pandas Series to preserve index:

        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
        >>> y_series = pd.Series(y, index=dates)
        >>> rf = RollingFeatures(stats=['mean', 'max'], window_sizes=5)
        >>> features_df = rf.transform_batch(y_series)
        >>> features_df.shape
        (10, 2)
        >>> features_df.index.equals(y_series.index)
        True

        Use with custom feature names:

        >>> rf = RollingFeatures(
        ...     stats='mean',
        ...     window_sizes=[7, 14, 30],
        ...     features_names=['ma_7', 'ma_14', 'ma_30']
        ... )
        >>> rf.fit(y)
        >>> rf.features_names
        ['ma_7', 'ma_14', 'ma_30']
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
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute rolling window statistics from time series data.

        For each statistic and window size combination, computes the rolling
        statistic across the input time series. The output contains NaN values
        for the initial positions where the window cannot be fully computed.

        Args:
            X: Time series data as 1D numpy array or array-like.

        Returns:
            np.ndarray: Array of shape (len(X), len(features_names)) containing
                the computed rolling statistics. Each column corresponds to a
                feature in features_names. Early positions contain NaN values
                before the window is fully populated.
        """
        # Assume X is 1D array
        n_samples = len(X)
        output = np.full((n_samples, len(self.features_names)), np.nan)

        idx_feature = 0
        for ws in self.window_sizes:
            for func in self.stats_funcs:
                # Naive rolling window loop - can be optimized or use pandas rolling
                # Using pandas for simplicity and speed if X is convertible
                series = pd.Series(X)
                rolled = series.rolling(window=ws).apply(func, raw=True)
                output[:, idx_feature] = rolled.values
                idx_feature += 1

        return output

    def transform_batch(self, X: pd.Series) -> pd.DataFrame:
        """
        Compute rolling features from a pandas Series with index preservation.

        Transforms a pandas Series into a DataFrame of rolling statistics while
        preserving the original index. Useful for maintaining time alignment
        with the input data.

        Args:
            X: Time series data as pandas Series. The index is preserved in output.

        Returns:
            pd.DataFrame: DataFrame with shape (len(X), len(features_names)) where
                columns are feature names and index matches the input Series.
                Contains NaN values at the beginning where windows are incomplete.

        Note:
            This method is preferred over transform() when working with time-indexed
            data, as it preserves the temporal index and is compatible with
            forecasting workflows.
        """
        values = X.to_numpy()
        transformed = self.transform(values)
        return pd.DataFrame(transformed, index=X.index, columns=self.features_names)

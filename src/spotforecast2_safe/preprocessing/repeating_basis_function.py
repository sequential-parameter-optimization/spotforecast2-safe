# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Repeating Basis Function transformer for cyclical features."""

from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RepeatingBasisFunction(BaseEstimator, TransformerMixin):
    """
    Transformer that encodes cyclical features using repeating radial basis functions.

    This transformer places Gaussian basis functions across the specified input range
    and wraps them around to handle periodicity (e.g., day of year, hour of day).
    It is a simplified implementation to avoid external dependencies like scikit-lego.

    Attributes:
        n_periods (int): Number of basis functions to place.
        column (str): Name of the column in the input DataFrame/Series to transform.
        input_range (Tuple[int, int]): The range of the input values (min, max).
        remainder (str): Policy for remaining columns (currently only 'drop' is supported).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.preprocessing.repeating_basis_function import RepeatingBasisFunction
        >>> X = pd.DataFrame({"hour": [0, 6, 12, 18, 23]})
        >>> rbf = RepeatingBasisFunction(n_periods=4, column="hour", input_range=(0, 23))
        >>> features = rbf.fit_transform(X)
        >>> features.shape
        (5, 4)
    """

    def __init__(
        self,
        n_periods: int,
        column: str,
        input_range: Tuple[int, int],
        remainder: str = "drop",
    ):
        """
        Initialize the RepeatingBasisFunction transformer.

        Args:
            n_periods: Number of basis functions.
            column: Name of the column to transform.
            input_range: Min and max values of the periodic feature.
            remainder: How to handle other columns. Defaults to "drop".
        """
        self.n_periods = n_periods
        self.column = column
        self.input_range = input_range
        self.remainder = remainder

    def fit(self, X: Any, y: Any = None) -> "RepeatingBasisFunction":
        """
        Fitted transformer (no-op).

        Args:
            X: Input data.
            y: Ignored.

        Returns:
            self: The fitted transformer.
        """
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        Transform the input data into RBF features.

        Args:
            X: Input DataFrame or Series containing the column to transform.

        Returns:
            np.ndarray: Array of transformed features with shape (n_samples, n_periods).

        Raises:
            ValueError: If the specified column is not found in the input.
        """
        # Allow passing just the column series if X is not a DataFrame
        if isinstance(X, pd.Series):
            vals = X.values
        elif isinstance(X, pd.DataFrame) and self.column in X.columns:
            vals = X[self.column].values
        else:
            raise ValueError(f"Column {self.column} not found in input")

        # Normalize to [0, 1] relative to input range
        vals_norm = (vals - self.input_range[0]) / (
            self.input_range[1] - self.input_range[0]
        )

        features = []
        for i in range(self.n_periods):
            mu = i / self.n_periods
            # Gaussian with wraparound handling for cyclic
            diff = np.abs(vals_norm - mu)
            diff = np.minimum(diff, 1 - diff)  # cyclic distance
            # sigma estimated as 1 / n_periods for reasonable overlap
            sigma = 1 / self.n_periods
            val = np.exp(-(diff**2) / (2 * sigma**2))
            features.append(val)

        return np.stack(features, axis=1)

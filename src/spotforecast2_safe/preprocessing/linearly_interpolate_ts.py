# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Linear interpolation transformer for time series data."""

from dataclasses import dataclass
from typing import Any, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class LinearlyInterpolateTS(BaseEstimator, TransformerMixin):
    """
    Transformer that applies linear interpolation to time series data.

    This transformer fills missing values using linear interpolation and
    forward-fills any remaining gaps (e.g., at the end of the series).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.preprocessing.linearly_interpolate_ts import LinearlyInterpolateTS
        >>> s = pd.Series([1.0, np.nan, 3.0, np.nan])
        >>> interpolator = LinearlyInterpolateTS()
        >>> s_filled = interpolator.fit_transform(s)
        >>> s_filled.tolist()
        [1.0, 2.0, 3.0, 3.0]
    """

    def fit(self, X: Any, y: Any = None) -> "LinearlyInterpolateTS":
        """
        Fitted transformer (no-op).

        Args:
            X: Input data.
            y: Ignored.

        Returns:
            self: The fitted transformer.
        """
        return self

    def transform(
        self, X: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Transform the input data by applying linear interpolation.

        Args:
            X: Input Series or DataFrame to interpolate.

        Returns:
            Union[pd.Series, pd.DataFrame]: Interpolated data.
        """
        return self.apply(X)

    def apply(
        self, y: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply linear interpolation and forward-fill.

        Args:
            y: Input Series or DataFrame.

        Returns:
            Union[pd.Series, pd.DataFrame]: Interpolated and ffilled data.
        """
        y_filled = y.interpolate(method="linear")
        y_filled = y_filled.astype("float").ffill()
        return y_filled

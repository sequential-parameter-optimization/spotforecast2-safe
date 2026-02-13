# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from ._common import _check_X_numpy_ndarray_1d


class TimeSeriesDifferentiator(BaseEstimator, TransformerMixin):
    """
    Transforms a time series into a differenced time series.

    Args:
        order (int, optional): Order of differentiation. Defaults to 1.
        initial_values (list, numpy ndarray, optional): Values to be used for the inverse transformation (reverting differentiation).
            If None, the first `order` values of the training data `X` are stored during `fit`.

    Attributes:
        initial_values_ (list): Values stored for inverse transformation.
        last_values_ (list): Last values of the differenced time series.
    """

    def __init__(self, order: int = 1, initial_values: list | np.ndarray | None = None):
        self.order = order
        self.initial_values = initial_values

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def fit(self, X: np.ndarray, y: object = None) -> object:
        """
        Store initial values if not provided.
        """
        if self.order < 1:
            raise ValueError("`order` must be a positive integer.")

        if self.initial_values is None:
            if len(X) < self.order:
                raise ValueError(
                    f"The time series must have at least {self.order} values "
                    f"to compute the differentiation of order {self.order}."
                )
            self.initial_values_ = list(X[: self.order])
        else:
            if len(self.initial_values) != self.order:
                raise ValueError(
                    f"The length of `initial_values` must be equal to the order "
                    f"of differentiation ({self.order})."
                )
            self.initial_values_ = list(self.initial_values)

        self.last_values_ = X[-self.order :]

        return self

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def fit_transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Fit and transform.
        """
        return self.fit(X).transform(X)

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Compute the differences.
        """
        if not hasattr(self, "initial_values_") and self.initial_values is not None:
            self.fit(X)
        elif not hasattr(self, "initial_values_"):
            check_is_fitted(self, ["initial_values_"])

        X_diff = np.diff(X, n=self.order)
        # Pad with NaNs to keep same length
        X_diff = np.concatenate([np.full(self.order, np.nan), X_diff])

        # Update last values seen (for next window inverse)
        self.last_values_ = X[-self.order :]

        return X_diff

    def inverse_transform_next_window(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform for the next window of predictions.
        """
        check_is_fitted(self, ["initial_values_", "last_values_"])

        if self.order == 1:
            result = np.cumsum(X) + self.last_values_[-1]
        else:
            # Recursive or iterative approach for higher orders
            # Simplified: Assuming order 1 is sufficient for now or throwing error
            raise NotImplementedError(
                "inverse_transform_next_window not implemented for order > 1"
            )

        return result

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def inverse_transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Revert the differences.
        """
        check_is_fitted(self, ["initial_values_"])

        # X contains the differenced series (with NaNs at the beginning potentially)
        # remove NaNs at the start corresponding to order
        X_clean = X[self.order :]

        if len(X_clean) == 0:
            # Just return initial values if only NaNs were passed
            return np.array(self.initial_values_)

        result = list(self.initial_values_)

        if self.order == 1:
            current_value = result[-1]
            restored = []
            for diff_val in X_clean:
                current_value += diff_val
                restored.append(current_value)
            result.extend(restored)
        else:
            # Recursive reconstruction for higher orders logic check
            # For order > 1, np.diff does repeated diffs.
            # To invert, we need to do repeated cumsum.
            # But we need appropriate initial values for each level of integration.
            # This is a simplified version.

            raise NotImplementedError(
                "Inverse transform for order > 1 is currently not fully implemented in this port."
            )

        return np.array(result)

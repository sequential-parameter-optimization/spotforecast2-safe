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
        window_size (int, optional): The window size used by the forecaster. Defaults to None.
        initial_values (list, numpy ndarray, optional): Values to be used for the inverse transformation (reverting differentiation).
            If None, the first `order` values of the training data `X` are stored during `fit`.

    Attributes:
        initial_values_ (list): Values stored for inverse transformation.
        last_values_ (list): Last values of the differenced time series.
        pre_train_values_ (list): First training values for inverse transformation of training data.

    Examples:
        ```{python}
        import numpy as np
        from spotforecast2_safe.preprocessing import TimeSeriesDifferentiator

        rng = np.random.default_rng(0)
        y = rng.integers(10, 100, size=10).astype(float)
        diff = TimeSeriesDifferentiator(order=1)
        y_diff = diff.fit_transform(y)
        assert y_diff.shape == y.shape
        assert np.isnan(y_diff[0])
        y_back = diff.inverse_transform(y_diff)
        np.testing.assert_array_almost_equal(y_back, y)
        print(f"original : {y[:4]}")
        print(f"differenced (first 4): {y_diff[:4]}")
        print(f"recovered : {y_back[:4]}")
        ```
    """

    def __init__(
        self,
        order: int = 1,
        window_size: int | None = None,
        initial_values: list | np.ndarray | None = None,
    ):
        self.order = order
        self.window_size = window_size
        self.initial_values = initial_values

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def fit(self, X: np.ndarray, y: object = None) -> object:
        """
        Store initial values if not provided.

        Args:
            X (np.ndarray): 1D time series array to fit on.
            y (object, optional): Ignored. Present for sklearn API compatibility.
                Defaults to None.

        Returns:
            TimeSeriesDifferentiator: Fitted transformer (self).

        Raises:
            ValueError: If `order` is less than 1 or `X` has fewer than `order` values.
            TypeError: If `window_size` is not an integer.

        Examples:
            ```{python}
            import numpy as np
            from spotforecast2_safe.preprocessing import TimeSeriesDifferentiator

            y = np.array([10.0, 12.0, 11.0, 14.0, 13.0, 15.0])
            diff = TimeSeriesDifferentiator(order=1)
            fitted = diff.fit(y)
            assert fitted.initial_values_ == [10.0]
            assert fitted.last_values_ == [15.0]
            print(f"initial_values_: {fitted.initial_values_}")
            print(f"last_values_: {fitted.last_values_}")
            ```
        """
        if self.order < 1:
            raise ValueError("`order` must be a positive integer.")
        if self.order > 1:
            # Fail fast: only first-order differentiation is implemented in this
            # port (inverse_transform raises NotImplementedError for order > 1).
            # Rejecting here, rather than letting fit succeed and crashing during
            # prediction, honors the fail-safe-over-silent-failure contract.
            raise NotImplementedError(
                "Only first-order differentiation (order=1) is supported in "
                f"spotforecast2-safe; got order={self.order}. Higher orders are "
                "intentionally unimplemented in this port."
            )

        if self.window_size is not None:
            if not isinstance(self.window_size, int):
                raise TypeError(
                    f"Parameter `window_size` must be an integer greater than 0. "
                    f"Found {type(self.window_size)}."
                )
            if self.window_size < 1:
                raise ValueError(
                    f"Parameter `window_size` must be an integer greater than 0. "
                    f"Found {self.window_size}."
                )

        self.initial_values_ = []
        self.pre_train_values_ = []
        self.last_values_ = []

        if self.initial_values is None:
            if len(X) < self.order:
                raise ValueError(
                    f"The time series must have at least {self.order} values "
                    f"to compute the differentiation of order {self.order}."
                )

            # Logic similar to Skforecast:
            # We iterate to capture initial values, pre_train_values, and last_values
            # Skforecast does this iteratively. The original spotforecast implementation
            # was simplified. Integrating window_size support requires the iterative approach
            # or careful indexing.

            # Re-implementing skforecast logic for robustness with window_size

            current_X = X.copy()
            for i in range(self.order):
                self.initial_values_.append(current_X[0])
                if self.window_size is not None:
                    # Skforecast logic: self.pre_train_values.append(X[self.window_size - self.order])
                    # But wait, skforecast loop updates X_diff. A recursive implementation.

                    # If we follow skforecast exactly:
                    # if i == 0: X_diff = diff(X); initial = X[0]; pre = X[ws-order]; last = X[-1]
                    # else: X_diff = diff(previous_diff); initial = prev_diff[0]; pre = prev_diff[ws-order]; last = prev_diff[-1]

                    # Current X is the series being differentiated in this step
                    if len(current_X) > (self.window_size - self.order):
                        self.pre_train_values_.append(
                            current_X[self.window_size - self.order]
                        )
                    else:
                        # Fallback if X is smaller than window_size (shouldn't happen during training if window_size is correct)
                        self.pre_train_values_.append(np.nan)

                self.last_values_.append(current_X[-1])
                current_X = np.diff(current_X, n=1)

        else:
            if len(self.initial_values) != self.order:
                raise ValueError(
                    f"The length of `initial_values` must be equal to the order "
                    f"of differentiation ({self.order})."
                )
            self.initial_values_ = list(self.initial_values)
            # If initial_values provided, we can still compute last_values_ from X if we assume X is the training data
            # But usually initial_values are provided when loading/restoring.
            # For now, let's keep the user provided initial values.
            # But we still need last_values_ for next window.
            # Assuming X is the training data passed to fit()

            self.last_values_ = list(X[-self.order :])

        return self

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def fit_transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Fit and transform the time series in one step.

        Args:
            X (np.ndarray): 1D time series array.
            y (object, optional): Ignored. Present for sklearn API compatibility.
                Defaults to None.

        Returns:
            np.ndarray: Differenced series with `order` leading NaNs.

        Examples:
            ```{python}
            import numpy as np
            from spotforecast2_safe.preprocessing import TimeSeriesDifferentiator

            y = np.array([5.0, 7.0, 6.0, 9.0, 8.0])
            diff = TimeSeriesDifferentiator(order=1)
            y_diff = diff.fit_transform(y)
            assert y_diff.shape == y.shape
            assert np.isnan(y_diff[0])
            assert y_diff[1] == 2.0  # 7 - 5
            print(f"fit_transform output: {y_diff}")
            ```
        """
        return self.fit(X).transform(X)

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Compute the differences.

        Args:
            X (np.ndarray): 1D time series array to difference.
            y (object, optional): Ignored. Present for sklearn API compatibility.
                Defaults to None.

        Returns:
            np.ndarray: Differenced array of the same length as `X`, with `order`
                leading NaN values.

        Examples:
            ```{python}
            import numpy as np
            from spotforecast2_safe.preprocessing import TimeSeriesDifferentiator

            y_train = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
            y_new = np.array([15.0, 21.0, 28.0])
            diff = TimeSeriesDifferentiator(order=1)
            diff.fit(y_train)
            y_new_diff = diff.transform(y_new)
            assert y_new_diff.shape == y_new.shape
            assert np.isnan(y_new_diff[0])
            assert y_new_diff[1] == 6.0  # 21 - 15
            print(f"transform output: {y_new_diff}")
            ```
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
        Invert differencing for the next prediction window.

        Reconstructs original-scale values from a differenced prediction array
        by cumulative-summing the differences and adding the last observed value
        (`last_values_[-1]`) stored during the previous `transform` call.

        Args:
            X (np.ndarray): 1D array of differenced predictions (no leading NaNs).

        Returns:
            np.ndarray: Predictions in the original scale.

        Raises:
            NotImplementedError: If `order` is greater than 1.

        Examples:
            ```{python}
            import numpy as np
            from spotforecast2_safe.preprocessing import TimeSeriesDifferentiator

            y_train = np.array([10.0, 12.0, 11.0, 14.0, 13.0])
            diff = TimeSeriesDifferentiator(order=1)
            diff.fit(y_train)
            # last_values_ is [13.0] after fit; simulate differenced predictions
            y_pred_diff = np.array([1.0, -1.0, 2.0])
            y_pred_orig = diff.inverse_transform_next_window(y_pred_diff)
            # cumsum([1, -1, 2]) + 13 = [14, 13, 15]
            np.testing.assert_array_almost_equal(y_pred_orig, [14.0, 13.0, 15.0])
            print(f"inverse_transform_next_window: {y_pred_orig}")
            ```
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

        Reconstructs the original time series from the differenced series produced
        by `transform`. The first `order` NaN values are stripped before
        reconstruction; the `initial_values_` stored during `fit` anchor the
        cumulative sum.

        Args:
            X (np.ndarray): Differenced 1D array as returned by `transform` (with
                `order` leading NaNs).
            y (object, optional): Ignored. Present for sklearn API compatibility.
                Defaults to None.

        Returns:
            np.ndarray: Reconstructed time series of length `len(X)`.

        Raises:
            NotImplementedError: If `order` is greater than 1.

        Examples:
            ```{python}
            import numpy as np
            from spotforecast2_safe.preprocessing import TimeSeriesDifferentiator

            rng = np.random.default_rng(42)
            y = rng.integers(1, 20, size=8).astype(float)
            diff = TimeSeriesDifferentiator(order=1)
            y_diff = diff.fit_transform(y)
            y_back = diff.inverse_transform(y_diff)
            # round-trip must recover the full original series
            np.testing.assert_array_almost_equal(y_back, y)
            print(f"original : {y}")
            print(f"recovered: {y_back}")
            ```
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

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def inverse_transform_training(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Revert the differentiation for training data.

        Uses `pre_train_values_` (populated only when `window_size` is set during
        `fit`) as the anchor value for the cumulative sum, producing the
        original-scale training targets aligned with the forecaster's feature
        matrix.

        Args:
            X (np.ndarray): Differenced training series with `order` leading NaNs,
                as returned by `transform`.
            y (object, optional): Ignored. Present for sklearn API compatibility.
                Defaults to None.

        Returns:
            np.ndarray: Reconstructed training target values.

        Raises:
            ValueError: If `window_size` was not set before fitting (i.e.,
                `pre_train_values_` is empty).
            NotImplementedError: If `order` is greater than 1.

        Examples:
            ```{python}
            import numpy as np
            from spotforecast2_safe.preprocessing import TimeSeriesDifferentiator

            y = np.array([10.0, 12.0, 11.0, 14.0, 13.0, 15.0, 16.0, 14.0])
            window_size = 3
            diff = TimeSeriesDifferentiator(order=1, window_size=window_size)
            diff.fit(y)
            y_diff = diff.transform(y)
            y_train_back = diff.inverse_transform_training(y_diff)
            assert y_train_back.ndim == 1
            print(f"pre_train_values_: {diff.pre_train_values_}")
            print(f"inverse_transform_training output: {y_train_back}")
            ```
        """
        if not hasattr(self, "pre_train_values_") or not self.pre_train_values_:
            raise ValueError(
                "The `window_size` parameter must be set before fitting the "
                "transformer to revert the differentiation of the training "
                "time series."
            )

        # Implementation of inverse_transform_training logic
        # For order 1
        if self.order == 1:
            X_clean = X[self.order :]
            # Reconstruct
            # pre_train_values_ contains the value right BEFORE the training window starts?
            # skforecast: X_undiff = np.insert(X, 0, self.pre_train_values[-1])
            #             X_undiff = np.cumsum(X_undiff)
            #             X_undiff = X_undiff[self.order:]

            X_undiff = np.insert(X_clean, 0, self.pre_train_values_[-1])
            X_undiff = np.cumsum(X_undiff)
            # The skforecast logic seems to insert, cumsum, then slice.
            # If X_clean is the differentiated training data (which generates y_train),
            # we need the value just before it to start cumsum.

            # Simplified for order=1 for now to match safety/robustness needs.
            return X_undiff

        else:
            raise NotImplementedError(
                "inverse_transform_training not implemented for order > 1"
            )

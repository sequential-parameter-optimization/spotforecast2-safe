# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``forecaster.metrics``.

The module is a parity-locked skforecast port whose examples previously ran only
at doc-render time, leaving its fail-safe type/length guards unexercised by the
pytest suite. These tests cover each metric's happy path and every explicit
``TypeError`` / ``ValueError`` branch.
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.forecaster.metrics import (
    _get_metric,
    add_y_train_argument,
    calculate_coverage,
    create_mean_pinball_loss,
    crps_from_predictions,
    crps_from_quantiles,
    mean_absolute_scaled_error,
    root_mean_squared_scaled_error,
    symmetric_mean_absolute_percentage_error,
)


class TestGetMetric:
    def test_returns_callable_for_known_metric(self):
        mse = _get_metric("mean_squared_error")
        err = mse(np.array([1, 2, 3]), np.array([1.1, 1.9, 3.2]))
        assert err > 0

    def test_accepts_y_train_kwarg(self):
        # _get_metric wraps via add_y_train_argument, so y_train is accepted.
        mae = _get_metric("mean_absolute_error")
        err = mae(np.array([1, 2, 3]), np.array([1, 2, 3]), y_train=None)
        assert err == 0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Allowed metrics are"):
            _get_metric("not_a_metric")


class TestAddYTrainArgument:
    def test_adds_argument_when_absent(self):
        def my_metric(y_true, y_pred):
            return float(np.mean(np.abs(y_true - y_pred)))

        enhanced = add_y_train_argument(my_metric)
        assert "y_train" in __import__("inspect").signature(enhanced).parameters
        assert enhanced(np.array([1, 2]), np.array([1, 2]), y_train=None) == 0

    def test_returns_func_unchanged_when_present(self):
        def metric_with_train(y_true, y_pred, y_train=None):
            return 0.0

        assert add_y_train_argument(metric_with_train) is metric_with_train


class TestMeanAbsoluteScaledError:
    def test_happy_path(self):
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        mase = mean_absolute_scaled_error(
            np.array([9, 10, 11]), np.array([8.8, 10.2, 10.9]), y_train
        )
        assert mase < 1.0

    def test_y_train_list_of_arrays(self):
        mase = mean_absolute_scaled_error(
            np.array([9, 10]),
            np.array([9, 10]),
            [np.array([1, 2, 3]), np.array([4, 6, 8])],
        )
        assert mase == 0.0

    def test_y_true_wrong_type_raises(self):
        with pytest.raises(TypeError, match="`y_true` must be"):
            mean_absolute_scaled_error([1, 2], np.array([1, 2]), np.array([1, 2, 3]))

    def test_y_pred_wrong_type_raises(self):
        with pytest.raises(TypeError, match="`y_pred` must be"):
            mean_absolute_scaled_error(np.array([1, 2]), [1, 2], np.array([1, 2, 3]))

    def test_y_train_wrong_type_raises(self):
        with pytest.raises(TypeError, match="`y_train` must be"):
            mean_absolute_scaled_error(np.array([1, 2]), np.array([1, 2]), "abc")

    def test_y_train_list_bad_element_raises(self):
        with pytest.raises(TypeError, match="each element must be"):
            mean_absolute_scaled_error(np.array([1, 2]), np.array([1, 2]), [1, 2, 3])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mean_absolute_scaled_error(
                np.array([1, 2, 3]), np.array([1, 2]), np.array([1, 2, 3])
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one element"):
            mean_absolute_scaled_error(np.array([]), np.array([]), np.array([1, 2, 3]))


class TestRootMeanSquaredScaledError:
    def test_happy_path(self):
        rmsse = root_mean_squared_scaled_error(
            np.array([9, 10, 11]),
            np.array([8.8, 10.2, 10.9]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        )
        assert rmsse < 1.0

    def test_y_true_wrong_type_raises(self):
        with pytest.raises(TypeError, match="`y_true` must be"):
            root_mean_squared_scaled_error(
                [1, 2], np.array([1, 2]), np.array([1, 2, 3])
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            root_mean_squared_scaled_error(
                np.array([1, 2, 3]), np.array([1, 2]), np.array([1, 2, 3])
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one element"):
            root_mean_squared_scaled_error(
                np.array([]), np.array([]), np.array([1, 2, 3])
            )


class TestCrpsFromPredictions:
    def test_happy_path_non_negative(self):
        crps = crps_from_predictions(5.0, np.array([4.5, 5.1, 4.9, 5.3, 4.7]))
        assert crps >= 0

    def test_y_pred_not_1d_raises(self):
        with pytest.raises(TypeError, match="1D numpy array"):
            crps_from_predictions(5.0, np.array([[1.0, 2.0]]))

    def test_y_true_wrong_type_raises(self):
        with pytest.raises(TypeError, match="float or integer"):
            crps_from_predictions("x", np.array([1.0, 2.0]))


class TestCrpsFromQuantiles:
    def test_happy_path_non_negative(self):
        crps = crps_from_quantiles(
            5.0,
            np.array([4.0, 4.5, 5.0, 5.5, 6.0]),
            np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
        )
        assert crps >= 0

    def test_y_true_wrong_type_raises(self):
        with pytest.raises(TypeError, match="float or integer"):
            crps_from_quantiles("x", np.array([1.0]), np.array([0.5]))

    def test_pred_quantiles_not_1d_raises(self):
        with pytest.raises(TypeError, match="`pred_quantiles` must be a 1D"):
            crps_from_quantiles(5.0, np.array([[1.0]]), np.array([0.5]))

    def test_quantile_levels_not_1d_raises(self):
        with pytest.raises(TypeError, match="`quantile_levels` must be a 1D"):
            crps_from_quantiles(5.0, np.array([1.0]), np.array([[0.5]]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must be equal"):
            crps_from_quantiles(5.0, np.array([1.0, 2.0]), np.array([0.5]))


class TestCalculateCoverage:
    def test_full_coverage(self):
        cov = calculate_coverage(
            np.array([1, 2, 3, 4, 5]),
            np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
            np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        )
        assert cov == 1.0

    def test_partial_coverage(self):
        cov = calculate_coverage(
            np.array([0, 2, 10]), np.array([1, 1, 1]), np.array([3, 3, 3])
        )
        assert cov == pytest.approx(1 / 3)

    def test_y_true_wrong_type_raises(self):
        with pytest.raises(TypeError, match="`y_true` must be a 1D"):
            calculate_coverage([1, 2], np.array([0, 1]), np.array([2, 3]))

    def test_lower_bound_wrong_type_raises(self):
        with pytest.raises(TypeError, match="`lower_bound` must be a 1D"):
            calculate_coverage(np.array([1, 2]), [0, 1], np.array([2, 3]))

    def test_upper_bound_wrong_type_raises(self):
        with pytest.raises(TypeError, match="`upper_bound` must be a 1D"):
            calculate_coverage(np.array([1, 2]), np.array([0, 1]), [2, 3])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            calculate_coverage(np.array([1, 2, 3]), np.array([0, 1]), np.array([2, 3]))


class TestCreateMeanPinballLoss:
    def test_happy_path_non_negative(self):
        loss = create_mean_pinball_loss(alpha=0.5)
        value = loss(np.array([1, 2, 3, 4, 5]), np.array([1.1, 1.9, 3.2, 3.8, 5.1]))
        assert value >= 0

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_alpha_out_of_range_raises(self, alpha):
        with pytest.raises(ValueError, match="between 0 and 1"):
            create_mean_pinball_loss(alpha=alpha)


class TestSmape:
    def test_happy_path_in_range(self):
        result = symmetric_mean_absolute_percentage_error(
            np.array([100, 200, 0]), np.array([110, 180, 10])
        )
        assert 0 <= result <= 200

    def test_handles_zero_denominator(self):
        # y_true == y_pred == 0 for an element: denominator is 0, contributes 0.
        result = symmetric_mean_absolute_percentage_error(
            np.array([0, 4]), np.array([0, 4])
        )
        assert result == 0.0

    def test_accepts_pandas_series(self):
        result = symmetric_mean_absolute_percentage_error(
            pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])
        )
        assert result == 0.0

    def test_y_true_wrong_type_raises(self):
        with pytest.raises(TypeError, match="`y_true` must be"):
            symmetric_mean_absolute_percentage_error([1, 2], np.array([1, 2]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            symmetric_mean_absolute_percentage_error(
                np.array([1, 2, 3]), np.array([1, 2])
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one element"):
            symmetric_mean_absolute_percentage_error(np.array([]), np.array([]))

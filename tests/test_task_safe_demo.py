# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Real tests for the ``task_safe_demo`` task and the helpers it relies on.

Scope is deliberately narrow: the behaviour that actually belongs to the demo
task — metric computation (``calculate_metrics``), CLI boolean parsing
(``parse_bool``), forecast-horizon validation, and the fail-fast missing-data
guard in ``main`` — rather than re-asserting numpy/pandas/sklearn behaviour.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.manager.demo_metrics import calculate_metrics
from spotforecast2_safe.tasks.task_safe_demo import main as demo_main
from spotforecast2_safe.utils.parse import parse_bool


class TestCalculateMetrics:
    """Test metrics calculation functionality."""

    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics when prediction equals actual (MAE=0, MSE=0)."""
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calculate_metrics(actual, predicted)

        assert result["MAE"] == 0.0
        assert result["MSE"] == 0.0

    def test_calculate_metrics_constant_offset(self):
        """Test metrics with constant error offset."""
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])  # Off by 1

        result = calculate_metrics(actual, predicted)

        assert result["MAE"] == 1.0
        assert result["MSE"] == 1.0

    def test_calculate_metrics_raises_on_nan(self):
        """Test that NaN values in either series raise ValueError."""
        actual = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        predicted = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])

        with pytest.raises(ValueError, match="NaN"):
            calculate_metrics(actual, predicted)


class TestParseBool:
    """CLI boolean parsing — exercises the real ``parse_bool`` (not a re-impl).

    The real contract raises ``argparse.ArgumentTypeError`` (not ``ValueError``)
    on bad input, which the previous local re-implementations got wrong.
    """

    @pytest.mark.parametrize(
        "value",
        ["true", "True", "TRUE", "t", "T", "yes", "YES", "1", "  true  "],
    )
    def test_true_variants(self, value):
        assert parse_bool(value) is True

    @pytest.mark.parametrize(
        "value",
        ["false", "False", "FALSE", "f", "F", "no", "NO", "0", "  false  "],
    )
    def test_false_variants(self, value):
        assert parse_bool(value) is False

    def test_invalid_value_raises_argument_type_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_bool("maybe")


class TestForecastHorizonValidation:
    """Forecast horizon is validated fail-fast at the pipeline boundary."""

    def test_invalid_forecast_horizon(self):
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            n2n_predict_with_covariates,
        )

        with pytest.raises(ValueError):
            n2n_predict_with_covariates(forecast_horizon=-1)

        with pytest.raises(ValueError):
            n2n_predict_with_covariates(forecast_horizon=0)


class TestDemoMainFailFast:
    """``main`` must refuse to run when the ground-truth file is absent."""

    def test_missing_ground_truth_returns_nonzero(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        assert not missing.exists()

        # Fail-fast: returns non-zero before any compute, no exception raised.
        assert demo_main(force_train=False, data_path=missing) == 1

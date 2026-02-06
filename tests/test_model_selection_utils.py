"""
Tests for spotforecast2_safe.model_selection.utils_common module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

from spotforecast2_safe.model_selection.utils_common import (
    initialize_lags_grid,
    check_backtesting_input,
    check_one_step_ahead_input,
    select_n_jobs_backtesting,
    OneStepAheadValidationWarning,
)
from spotforecast2_safe.model_selection import TimeSeriesFold, OneStepAheadFold


class TestInitializeLagsGrid:
    """Tests for initialize_lags_grid."""

    def test_initialize_lags_grid_list(self):
        """Test with list of lags."""
        forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        lags_grid = [2, 4]
        grid, label = initialize_lags_grid(forecaster, lags_grid)
        assert label == "values"
        assert grid == {"2": 2, "4": 4}

    def test_initialize_lags_grid_dict(self):
        """Test with dict of lags."""
        forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        lags_grid = {"lags_2": 2, "lags_4": 4}
        grid, label = initialize_lags_grid(forecaster, lags_grid)
        assert label == "keys"
        assert grid == lags_grid

    def test_initialize_lags_grid_none(self):
        """Test with None (uses forecaster lags)."""
        forecaster = ForecasterRecursive(LinearRegression(), lags=[1, 2])
        grid, label = initialize_lags_grid(forecaster, None)
        assert label == "values"
        # Forecaster lags might be array or list depending on initialization
        # The function converts forecaster.lags to list of ints
        assert grid == {"[1, 2]": [1, 2]}

    def test_initialize_lags_grid_invalid_type(self):
        """Test with invalid input type."""
        forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        with pytest.raises(
            TypeError, match="`lags_grid` argument must be a list, dict or None"
        ):
            initialize_lags_grid(forecaster, "invalid")


class TestCheckBacktestingInput:
    """Tests for check_backtesting_input."""

    @pytest.fixture
    def setup_data(self):
        y = pd.Series(np.arange(50), name="y")
        cv = TimeSeriesFold(
            steps=5,
            initial_train_size=20,
            gap=0,
            refit=False,
            fixed_train_size=False,
            allow_incomplete_fold=True,
        )
        forecaster = ForecasterRecursive(LinearRegression(), lags=5)
        forecaster.fit(
            y=y.iloc[:20]
        )  # Fit to avoid NotFittedError if initial_train_size checking logic needs it
        return y, cv, forecaster

    def test_check_backtesting_input_valid(self, setup_data):
        """Test with valid inputs."""
        y, cv, forecaster = setup_data
        check_backtesting_input(
            forecaster=forecaster, cv=cv, metric=mean_squared_error, y=y
        )

    def test_check_backtesting_input_invalid_cv(self, setup_data):
        """Test with invalid CV object."""
        y, _, forecaster = setup_data
        with pytest.raises(TypeError, match="`cv` must be a 'TimeSeriesFold' object"):
            check_backtesting_input(
                forecaster=forecaster, cv="invalid_cv", metric=mean_squared_error, y=y
            )

    def test_check_backtesting_input_invalid_y(self, setup_data):
        """Test with invalid y."""
        _, cv, forecaster = setup_data
        with pytest.raises(TypeError, match="`y` must be a pandas Series"):
            check_backtesting_input(
                forecaster=forecaster, cv=cv, metric=mean_squared_error, y=[1, 2, 3]
            )

    def test_check_backtesting_input_invalid_metric(self, setup_data):
        """Test with invalid metric."""
        y, cv, forecaster = setup_data
        with pytest.raises(
            TypeError, match="`metric` must be a string, a callable function, or a list"
        ):
            check_backtesting_input(forecaster=forecaster, cv=cv, metric=123, y=y)

    def test_check_backtesting_input_initial_train_size_too_small(self, setup_data):
        """Test initial_train_size smaller than window_size."""
        y, _, forecaster = setup_data
        # Forecaster lags=5 -> window_size=5
        cv = TimeSeriesFold(
            steps=5,
            initial_train_size=4,  # < 5
            refit=False,
            allow_incomplete_fold=True,
        )
        with pytest.raises(ValueError, match="must be greater than the `window_size`"):
            check_backtesting_input(
                forecaster=forecaster, cv=cv, metric=mean_squared_error, y=y
            )

    def test_check_backtesting_input_initial_train_size_too_large(self, setup_data):
        """Test initial_train_size larger than data."""
        y, _, forecaster = setup_data
        cv = TimeSeriesFold(
            steps=5,
            initial_train_size=len(y) + 1,
            refit=False,
            allow_incomplete_fold=True,
        )
        with pytest.raises(ValueError, match="smaller than the length of `y`"):
            check_backtesting_input(
                forecaster=forecaster, cv=cv, metric=mean_squared_error, y=y
            )


class TestCheckOneStepAheadInput:
    """Tests for check_one_step_ahead_input."""

    @pytest.fixture
    def setup_data(self):
        y = pd.Series(np.arange(50), name="y")
        cv = OneStepAheadFold(initial_train_size=20, return_all_indexes=False)
        forecaster = ForecasterRecursive(LinearRegression(), lags=5)
        return y, cv, forecaster

    def test_check_one_step_ahead_input_valid(self, setup_data):
        """Test with valid inputs."""
        y, cv, forecaster = setup_data
        # Expect warning unless suppressed
        with pytest.warns(OneStepAheadValidationWarning):
            check_one_step_ahead_input(
                forecaster=forecaster, cv=cv, metric=mean_squared_error, y=y
            )

    def test_check_one_step_ahead_input_suppress_warnings(self, setup_data):
        """Test with suppressed warnings."""
        import warnings

        y, cv, forecaster = setup_data
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check_one_step_ahead_input(
                forecaster=forecaster,
                cv=cv,
                metric=mean_squared_error,
                y=y,
                suppress_warnings=True,
            )
            # Filter for OneStepAheadValidationWarning
            relevant_warnings = [
                w
                for w in record
                if issubclass(w.category, OneStepAheadValidationWarning)
            ]
            assert len(relevant_warnings) == 0

    def test_check_one_step_ahead_input_invalid_cv(self, setup_data):
        """Test with invalid CV."""
        y, _, forecaster = setup_data
        with pytest.raises(TypeError, match="`cv` must be a 'OneStepAheadFold' object"):
            check_one_step_ahead_input(
                forecaster=forecaster, cv="invalid", metric=mean_squared_error, y=y
            )


class TestSelectNJobsBacktesting:
    """Tests for select_n_jobs_backtesting."""

    def test_select_n_jobs_linear_model(self):
        """Test Recursive with Linear model -> 1 job."""
        forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        n_jobs = select_n_jobs_backtesting(forecaster, refit=True)
        assert n_jobs == 1

    def test_select_n_jobs_refit_int(self):
        """Test refit as int -> 1 job."""
        forecaster = ForecasterRecursive(LinearRegression(), lags=2)
        n_jobs = select_n_jobs_backtesting(forecaster, refit=2)
        assert n_jobs == 1

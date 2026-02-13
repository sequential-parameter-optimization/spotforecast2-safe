# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


class TestBinningInSampleResidualsRecursive:
    """Test suite for _binning_in_sample_residuals method in ForecasterRecursive."""

    def test_deterministic_behavior_with_random_state(self):
        """
        Test that random_state parameter ensures deterministic residual sampling.
        """
        np.random.seed(42)
        y = pd.Series(
            data=np.random.randn(200) * 10 + 50,
            index=pd.date_range(start="2022-01-01", periods=200, freq="D"),
        )

        # First run
        forecaster1 = ForecasterRecursive(
            estimator=LinearRegression(),
            lags=3,
            binner_kwargs={"n_bins": 3, "random_state": 123},
        )
        forecaster1.fit(y=y, store_in_sample_residuals=True)
        residuals1 = forecaster1.in_sample_residuals_.copy()
        bins1 = {
            k: v.copy() for k, v in forecaster1.in_sample_residuals_by_bin_.items()
        }

        # Second run with same random_state
        forecaster2 = ForecasterRecursive(
            estimator=LinearRegression(),
            lags=3,
            binner_kwargs={"n_bins": 3, "random_state": 123},
        )
        forecaster2.fit(y=y, store_in_sample_residuals=True)
        residuals2 = forecaster2.in_sample_residuals_.copy()
        bins2 = {
            k: v.copy() for k, v in forecaster2.in_sample_residuals_by_bin_.items()
        }

        # Assert exact reproducibility
        np.testing.assert_array_equal(residuals1, residuals2)
        assert bins1.keys() == bins2.keys()
        for k in bins1.keys():
            np.testing.assert_array_equal(bins1[k], bins2[k])

    def test_max_sample_per_bin_enforcement(self):
        """
        Test that residuals per bin are limited to max_sample = 10_000 // n_bins.
        """
        # Create large dataset to trigger downsampling
        np.random.seed(42)
        y = pd.Series(
            data=np.random.randn(20_000) * 10 + 50,
            index=pd.date_range(start="2022-01-01", periods=20_000, freq="D"),
        )

        n_bins = 4
        max_sample_per_bin = 10_000 // n_bins  # 2500

        forecaster = ForecasterRecursive(
            estimator=LinearRegression(),
            lags=3,
            binner_kwargs={"n_bins": n_bins, "random_state": 123},
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Verify each bin respects max_sample limit
        for bin_idx, residuals in forecaster.in_sample_residuals_by_bin_.items():
            assert len(residuals) <= max_sample_per_bin

    def test_total_residuals_limited_to_10000(self):
        """
        Test that total in_sample_residuals_ is limited to 10,000 samples.
        """
        # Create large dataset
        np.random.seed(42)
        y = pd.Series(
            data=np.random.randn(20_000) * 10 + 50,
            index=pd.date_range(start="2022-01-01", periods=20_000, freq="D"),
        )

        forecaster = ForecasterRecursive(
            estimator=LinearRegression(),
            lags=3,
            binner_kwargs={"n_bins": 5, "random_state": 123},
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Verify total residuals limited to 10,000
        assert len(forecaster.in_sample_residuals_) <= 10_000

    def test_store_false_only_stores_intervals(self):
        """
        Test that store_in_sample_residuals=False stores intervals but not residuals.
        """
        y = pd.Series(
            data=np.arange(50, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=50, freq="D"),
        )

        forecaster = ForecasterRecursive(
            estimator=LinearRegression(),
            lags=3,
            binner_kwargs={"n_bins": 3, "random_state": 123},
        )
        forecaster.fit(y=y, store_in_sample_residuals=False)

        # Should have intervals (dict with 3 bins)
        assert hasattr(forecaster, "binner_intervals_")
        assert len(forecaster.binner_intervals_) == 3

        # Attributes exist but should be None when store=False
        assert forecaster.in_sample_residuals_ is None
        assert forecaster.in_sample_residuals_by_bin_ is None

    def test_probabilistic_mode_default_behavior(self):
        """
        Test that when binner_kwargs is None, default binning is used and
        _probabilistic_mode is 'binned'.
        """
        y = pd.Series(
            data=np.arange(50, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=50, freq="D"),
        )

        forecaster = ForecasterRecursive(
            estimator=LinearRegression(), lags=3, binner_kwargs=None
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        assert forecaster._probabilistic_mode == "binned"
        assert forecaster.binner_intervals_ is not None
        assert forecaster.in_sample_residuals_by_bin_ is not None
        assert forecaster.in_sample_residuals_ is not None

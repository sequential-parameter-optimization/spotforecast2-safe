# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2_safe task scripts."""

import unittest
from unittest.mock import MagicMock, patch
import sys


class TestTaskSafeDemo(unittest.TestCase):
    """Tests for task_safe_demo.py."""

    @patch("spotforecast2_safe.tasks.task_safe_demo.n2n_predict")
    @patch("spotforecast2_safe.tasks.task_safe_demo.n2n_predict_with_covariates")
    @patch("spotforecast2_safe.tasks.task_safe_demo.agg_predict")
    def test_main_returns_zero_on_success(self, mock_agg, mock_cov, mock_n2n):
        """Test that main() returns 0 on successful execution."""
        from spotforecast2_safe.tasks.task_safe_demo import main, DemoConfig
        import pandas as pd

        # Mock return values
        mock_predictions = pd.DataFrame({"col1": [1, 2, 3]})
        mock_n2n.return_value = (mock_predictions, {})
        mock_cov.return_value = (mock_predictions, {}, {})
        mock_agg.return_value = pd.Series([1, 2, 3])

        # Create a mock data file
        with patch("pandas.read_csv") as mock_csv:
            mock_csv.return_value = pd.DataFrame(
                {"col1": range(100), "col2": range(100)},
                index=pd.date_range("2020-01-01", periods=100, freq="h"),
            )
            # Should not raise
            result = main(force_train=False)
            self.assertEqual(result, 0)


class TestTaskSafeN2O1CovDf(unittest.TestCase):
    """Tests for task_safe_n_to_1_with_covariates_and_dataframe.py."""

    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.n2n_predict_with_covariates"
    )
    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.agg_predict"
    )
    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.fetch_data"
    )
    def test_main_returns_zero_on_success(self, mock_fetch, mock_agg, mock_cov):
        """Test that main() returns 0 on successful execution."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )
        import pandas as pd

        # Mock return values
        mock_predictions = pd.DataFrame({"col1": [1, 2, 3]})
        mock_fetch.return_value = pd.DataFrame(
            {"col1": range(100)},
            index=pd.date_range("2020-01-01", periods=100, freq="h"),
        )
        mock_cov.return_value = (mock_predictions, {}, {})
        mock_agg.return_value = pd.Series([1, 2, 3])

        # Should complete without error (returns None)
        result = main(verbose=False)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

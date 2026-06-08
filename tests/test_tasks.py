# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2_safe task scripts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class TestTaskSafeDemo(unittest.TestCase):
    """Tests for task_safe_demo.py."""

    def test_main_returns_zero_on_success(self):
        """Test that main() returns 0 on successful execution."""
        from spotforecast2_safe.tasks.task_safe_demo import main

        # Create temporary test data file
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "demo11.csv"

            # Create mock data with columns matching what the forecasters return
            test_data = pd.DataFrame(
                {f"col{i}": range(100) for i in range(11)},
                index=pd.date_range("2020-01-01", periods=100, freq="h"),
            )
            test_data.to_csv(data_path)

            # Mock the forecasting functions
            mock_predictions = pd.DataFrame(
                {f"col{i}": [1.0, 2.0, 3.0] for i in range(11)},
                index=pd.date_range("2020-01-01", periods=3, freq="h"),
            )

            with (
                patch(
                    "spotforecast2_safe.tasks.task_safe_demo.n2n_predict"
                ) as mock_n2n,
                patch(
                    "spotforecast2_safe.tasks.task_safe_demo.n2n_predict_with_covariates"
                ) as mock_cov,
                patch(
                    "spotforecast2_safe.tasks.task_safe_demo.agg_predict"
                ) as mock_agg,
            ):
                mock_n2n.return_value = (mock_predictions, {})
                mock_cov.return_value = (mock_predictions, {}, {})
                mock_agg.return_value = pd.Series(
                    [1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="h")
                )

                result = main(force_train=False, data_path=data_path)
                self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()

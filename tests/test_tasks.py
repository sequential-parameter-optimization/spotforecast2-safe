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


class TestTaskSafeN2O1CovDf(unittest.TestCase):
    """Tests for task_safe_n_to_1_with_covariates_and_dataframe.py (ConfigMulti-driven)."""

    def test_main_exits_0_on_help(self):
        """main(["--help"]) must exit with code 0."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.run_pipeline"
    )
    def test_main_with_empty_argv_calls_run_pipeline(self, mock_run):
        """main([]) builds a config from defaults and delegates to run_pipeline."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        mock_run.return_value = pd.DataFrame({"forecast": [1.0, 2.0, 3.0]})
        main([])
        mock_run.assert_called_once()

    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.run_pipeline"
    )
    def test_main_forecast_horizon_flag(self, mock_run):
        """--forecast_horizon flag is translated to config.predict_size."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})
        main(["--forecast_horizon", "48"])
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["config"].predict_size, 48)

    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.run_pipeline",
        side_effect=RuntimeError("pipeline error"),
    )
    def test_main_exits_1_on_failure(self, _mock_run):
        """main exits with code 1 when run_pipeline raises."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()

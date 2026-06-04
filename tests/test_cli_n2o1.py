# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI smoke tests for the spotforecast-safe-n2o1-cov-df entry point.

These tests verify that:
- ``--help`` exits with code 0 (argparse contract).
- ``main(["--help"])`` exits with code 0 (programmatic argv path).
- An invalid flag value causes argparse to exit with code 2.
- ``main([])`` with a patched ``run_pipeline`` completes without error.

The tests do NOT run the full pipeline (that requires model training and
cached data); they exercise only the CLI argument-parsing layer.  The single
real end-to-end test lives in
``tests/test_task_safe_n_to_1_with_covariates.py::TestRunPipelineEndToEnd``.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

MOD = "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe"


class TestCliHelp:
    """``--help`` must produce a usage message and exit 0."""

    def test_help_via_subprocess(self):
        """Invoke the CLI via subprocess to exercise the full console-script path."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe",
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "forecast_horizon" in result.stdout

    def test_help_via_main_argv(self):
        """``main(["--help"])`` exits 0 without touching the pipeline."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_invalid_flag_exits_2(self):
        """An unrecognised flag must cause argparse to exit with code 2."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        with pytest.raises(SystemExit) as exc_info:
            main(["--no_such_flag"])
        assert exc_info.value.code == 2


class TestCliArgvParsing:
    """CLI flag parsing routes to the correct ``ConfigMulti`` fields."""

    @patch(f"{MOD}.run_pipeline")
    def test_empty_argv_runs_without_error(self, mock_run):
        """``main([])`` uses all defaults and calls ``run_pipeline`` once."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        mock_run.return_value = pd.DataFrame({"forecast": [1.0, 2.0]})
        main([])
        mock_run.assert_called_once()

    @patch(f"{MOD}.run_pipeline")
    def test_forecast_horizon_flag(self, mock_run):
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})
        main(["--forecast_horizon", "6"])
        _, kwargs = mock_run.call_args
        assert kwargs["config"].predict_size == 6

    @patch(f"{MOD}.run_pipeline")
    def test_lags_flag_expands_range(self, mock_run):
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})
        main(["--lags", "5"])
        _, kwargs = mock_run.call_args
        assert kwargs["config"].lags_consider == list(range(1, 6))

    @patch(f"{MOD}.run_pipeline")
    def test_weights_flag(self, mock_run):
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})
        main(["--weights", "1.0", "-1.0", "0.5"])
        _, kwargs = mock_run.call_args
        assert kwargs["config"].agg_weights == [1.0, -1.0, 0.5]

    @patch(f"{MOD}.run_pipeline")
    def test_log_dir_becomes_cache_home(self, mock_run, tmp_path: Path):
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})
        main(["--log_dir", str(tmp_path)])
        _, kwargs = mock_run.call_args
        assert kwargs["cache_home"] == tmp_path

    @patch(f"{MOD}.run_pipeline", side_effect=RuntimeError("boom"))
    def test_pipeline_error_exits_1(self, _mock_run):
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

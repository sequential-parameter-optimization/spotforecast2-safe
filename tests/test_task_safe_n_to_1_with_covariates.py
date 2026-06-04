# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the N-to-1-with-covariates task module (ConfigMulti-driven).

These tests exercise the thin entry-point in
``spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe``
by verifying:

- ``run_pipeline`` delegates to ``runner.run`` with the right arguments.
- ``run_pipeline`` with ``config=None`` builds the demo10 config.
- ``_build_config_from_cli`` maps CLI flags to ``ConfigMulti`` fields correctly.
- A real end-to-end run on synthetic data returns a non-empty forecast DataFrame.
- Fail-safe: invalid input raises ``TypeError``/``ValueError``.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
    _build_config_from_cli,
    main,
    run_pipeline,
)

MOD = "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 500, n_cols: int = 3) -> pd.DataFrame:
    """Return a small hourly DataFrame with a DatetimeIndex suitable for the pipeline.

    The pipeline's ``prepare_data`` step expects a DataFrame with a
    ``DatetimeIndex`` (or a ``DatetimeIndex`` named ``index_name``).  Passing
    a pre-reset DataFrame (with "DateTime" as a regular column) causes a
    ``ValueError`` when ``reset_index`` is called internally.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    data = rng.uniform(0, 100, size=(n, n_cols))
    df = pd.DataFrame(data, index=idx, columns=[f"C{i}" for i in range(n_cols)])
    df.index.name = "DateTime"
    return df


def _minimal_config(n_cols: int = 3, predict_size: int = 4) -> ConfigMulti:
    """Return a fast, offline-safe ``ConfigMulti`` for synthetic-data tests."""
    return ConfigMulti(
        predict_size=predict_size,
        train_size=pd.Timedelta(days=14),
        use_exogenous_features=False,
        use_outlier_detection=False,
        imputation_method="linear",
        agg_weights=[1.0] * n_cols,
    )


# ---------------------------------------------------------------------------
# run_pipeline: routing tests (runner.run patched)
# ---------------------------------------------------------------------------


class TestRunPipelineRouting:
    """``run_pipeline`` must delegate correctly to ``runner.run``."""

    @patch(f"{MOD}.run")
    def test_run_called_with_task_lazy(self, mock_run):
        mock_run.return_value = pd.DataFrame({"forecast": [1.0, 2.0, 3.0, 4.0]})
        cfg = _minimal_config()
        df = _make_synthetic_df()

        run_pipeline(config=cfg, dataframe=df, project_name="proj", cache_home="/tmp")

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["task"] == "lazy"
        assert kwargs["project_name"] == "proj"
        assert kwargs["cache_home"] == "/tmp"
        assert kwargs["dataframe"] is df

    @patch(f"{MOD}.run")
    @patch(f"{MOD}.make_demo10_config")
    @patch("pandas.read_csv")
    def test_none_config_uses_demo10_config(self, mock_read_csv, mock_demo10, mock_run):
        """When config=None, make_demo10_config() is invoked."""
        fake_cfg = _minimal_config()
        mock_demo10.return_value = fake_cfg
        # _make_synthetic_df already returns a DatetimeIndex DF
        fake_df = _make_synthetic_df()
        mock_read_csv.return_value = fake_df
        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})

        run_pipeline(config=None)

        mock_demo10.assert_called_once()
        mock_run.assert_called_once()

    @patch(f"{MOD}.run")
    def test_returns_dataframe_from_runner(self, mock_run):
        expected = pd.DataFrame(
            {"forecast": [10.0, 20.0, 30.0, 40.0]},
            index=pd.date_range("2020-01-01", periods=4, freq="h"),
        )
        mock_run.return_value = expected
        cfg = _minimal_config()
        df = _make_synthetic_df()

        result = run_pipeline(config=cfg, dataframe=df)

        assert result is expected


# ---------------------------------------------------------------------------
# run_pipeline: fail-safe behaviour
# ---------------------------------------------------------------------------


class TestRunPipelineFailSafe:
    """Invalid input must raise explicitly; nothing is silently swallowed."""

    def test_wrong_config_type_raises_type_error(self):
        with pytest.raises(TypeError, match="ConfigMulti"):
            run_pipeline(config="not_a_config")

    def test_wrong_config_type_int_raises_type_error(self):
        with pytest.raises(TypeError, match="ConfigMulti"):
            run_pipeline(config=42)

    @patch(f"{MOD}.run", side_effect=ValueError("bad data"))
    def test_pipeline_failure_propagates(self, _mock_run):
        cfg = _minimal_config()
        df = _make_synthetic_df()
        with pytest.raises(ValueError, match="bad data"):
            run_pipeline(config=cfg, dataframe=df)


# ---------------------------------------------------------------------------
# run_pipeline: real end-to-end on synthetic data
# ---------------------------------------------------------------------------


class TestRunPipelineEndToEnd:
    """At least one test exercises the real pipeline on synthetic data."""

    def test_returns_non_empty_forecast_of_correct_length(self, tmp_path: Path):
        """Full real pipeline run (no mocks) on a small synthetic DataFrame."""
        n = 500
        predict_size = 4
        n_cols = 3
        df = _make_synthetic_df(n=n, n_cols=n_cols)
        cfg = _minimal_config(n_cols=n_cols, predict_size=predict_size)

        result = run_pipeline(
            config=cfg,
            dataframe=df,
            project_name="e2e_test",
            cache_home=str(tmp_path),
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == predict_size
        assert "forecast" in result.columns


# ---------------------------------------------------------------------------
# _build_config_from_cli: flag mapping
# ---------------------------------------------------------------------------


class TestBuildConfigFromCli:
    """``_build_config_from_cli`` must map every CLI flag to the right ConfigMulti field."""

    def _args(self, **kwargs) -> argparse.Namespace:
        defaults = dict(
            forecast_horizon=24,
            lags=24,
            train_ratio=0.8,
            contamination=0.01,
            window_size=72,
            latitude=51.5136,
            longitude=7.4653,
            timezone="UTC",
            country_code="DE",
            state="NW",
            include_weather_windows=False,
            include_holiday_features=False,
            include_holiday_adjacency_features=False,
            poly_features_degree=1,
            max_poly_features=10,
            verbose=False,
            weights=None,
            log_dir=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_predict_size_mapped(self):
        cfg = _build_config_from_cli(self._args(forecast_horizon=48))
        assert cfg.predict_size == 48

    def test_lags_expanded_to_range(self):
        cfg = _build_config_from_cli(self._args(lags=5))
        assert cfg.lags_consider == list(range(1, 6))
        assert cfg.lags_consider[-1] == 5

    def test_lags_last_matches_n(self):
        """lags_consider[-1] == N preserves old ForecasterRecursive(lags=N) behaviour."""
        for n in (12, 24, 48):
            cfg = _build_config_from_cli(self._args(lags=n))
            assert cfg.lags_consider[-1] == n

    def test_train_ratio_yields_timedelta(self):
        cfg = _build_config_from_cli(self._args(train_ratio=0.5))
        assert isinstance(cfg.train_size, pd.Timedelta)
        assert cfg.train_size > pd.Timedelta(0)

    def test_train_ratio_proportional(self):
        """Larger ratio -> larger train_size."""
        cfg_large = _build_config_from_cli(self._args(train_ratio=0.9))
        cfg_small = _build_config_from_cli(self._args(train_ratio=0.5))
        assert cfg_large.train_size > cfg_small.train_size

    def test_contamination_mapped(self):
        cfg = _build_config_from_cli(self._args(contamination=0.05))
        assert cfg.contamination == 0.05

    def test_window_size_mapped(self):
        cfg = _build_config_from_cli(self._args(window_size=48))
        assert cfg.window_size == 48

    def test_latitude_mapped(self):
        cfg = _build_config_from_cli(self._args(latitude=48.1))
        assert cfg.latitude == 48.1

    def test_longitude_mapped(self):
        cfg = _build_config_from_cli(self._args(longitude=11.6))
        assert cfg.longitude == 11.6

    def test_timezone_mapped(self):
        cfg = _build_config_from_cli(self._args(timezone="Europe/Berlin"))
        assert cfg.timezone == "Europe/Berlin"

    def test_country_code_mapped(self):
        cfg = _build_config_from_cli(self._args(country_code="FR"))
        assert cfg.country_code == "FR"

    def test_state_mapped(self):
        cfg = _build_config_from_cli(self._args(state="BY"))
        assert cfg.state == "BY"

    def test_include_weather_windows_mapped(self):
        cfg = _build_config_from_cli(self._args(include_weather_windows=True))
        assert cfg.include_weather_windows is True

    def test_include_holiday_features_mapped(self):
        cfg = _build_config_from_cli(self._args(include_holiday_features=True))
        assert cfg.include_holiday_features is True

    def test_include_holiday_adjacency_features_mapped(self):
        cfg = _build_config_from_cli(
            self._args(include_holiday_adjacency_features=True)
        )
        assert cfg.include_holiday_adjacency_features is True

    def test_poly_features_degree_mapped(self):
        cfg = _build_config_from_cli(self._args(poly_features_degree=2))
        assert cfg.poly_features_degree == 2

    def test_max_poly_features_mapped(self):
        cfg = _build_config_from_cli(self._args(max_poly_features=5))
        assert cfg.max_poly_features == 5

    def test_verbose_mapped(self):
        cfg = _build_config_from_cli(self._args(verbose=True))
        assert cfg.verbose is True

    def test_weights_mapped_to_agg_weights(self):
        weights = [1.0, -1.0, 0.5]
        cfg = _build_config_from_cli(self._args(weights=weights))
        assert cfg.agg_weights == weights

    def test_weights_none_stays_none(self):
        cfg = _build_config_from_cli(self._args(weights=None))
        assert cfg.agg_weights is None


# ---------------------------------------------------------------------------
# main: CLI argv parsing
# ---------------------------------------------------------------------------


class TestMainCliParsing:
    """``main()`` must parse argv, build config, and call run_pipeline."""

    @patch(f"{MOD}.run_pipeline")
    def test_help_exits_0(self, _mock_run):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    @patch(f"{MOD}.run_pipeline")
    def test_main_with_no_argv_uses_defaults(self, mock_run):
        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})
        # Calling with empty argv list should not raise
        main([])
        mock_run.assert_called_once()

    @patch(f"{MOD}.run_pipeline")
    def test_forecast_horizon_flag_forwarded(self, mock_run):
        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})
        main(["--forecast_horizon", "48"])
        _, kwargs = mock_run.call_args
        # The config passed to run_pipeline should have predict_size=48
        assert kwargs["config"].predict_size == 48

    @patch(f"{MOD}.run_pipeline")
    def test_log_dir_becomes_cache_home(self, mock_run, tmp_path: Path):
        mock_run.return_value = pd.DataFrame({"forecast": [1.0]})
        main(["--log_dir", str(tmp_path)])
        _, kwargs = mock_run.call_args
        assert kwargs["cache_home"] == tmp_path

    @patch(f"{MOD}.run_pipeline", side_effect=RuntimeError("boom"))
    def test_pipeline_failure_exits_1(self, _mock_run):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

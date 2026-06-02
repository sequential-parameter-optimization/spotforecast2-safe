# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Real tests for the N-to-1-with-covariates task module.

These tests exercise the actual task code in
``spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe`` —
the orchestration in ``n_to_1_with_covariates`` and the fail-safe behaviour of
``main`` — rather than re-asserting numpy/pandas/sklearn library behaviour. The
heavy forecasting and aggregation calls are mocked so the orchestration logic
(argument forwarding, weight defaulting, error propagation) is verified in
isolation.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
    DEFAULT_WEIGHTS,
    main,
    n_to_1_with_covariates,
)

MOD = "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe"


@pytest.fixture
def mocked_pipeline():
    """Patch the heavy forecasting + aggregation calls with light stand-ins."""
    predictions = pd.DataFrame(
        {f"col{i}": [1.0, 2.0, 3.0] for i in range(11)},
        index=pd.date_range("2020-01-01", periods=3, freq="h"),
    )
    combined = pd.Series([1.0, 2.0, 3.0], index=predictions.index, name="combined")
    with (
        patch(f"{MOD}.n2n_predict_with_covariates") as mock_cov,
        patch(f"{MOD}.agg_predict") as mock_agg,
    ):
        mock_cov.return_value = (predictions, {"mae": 1.5}, {"n_features": 7})
        mock_agg.return_value = combined
        yield mock_cov, mock_agg, predictions, combined


class TestDefaultWeights:
    """The module-level aggregation weights are a stable contract."""

    def test_default_weights_shape(self):
        assert isinstance(DEFAULT_WEIGHTS, list)
        assert len(DEFAULT_WEIGHTS) == 11
        assert all(isinstance(w, float) for w in DEFAULT_WEIGHTS)


class TestNToOneWithCovariates:
    """Orchestration contract of ``n_to_1_with_covariates``."""

    def test_returns_four_tuple_of_pipeline_outputs(self, mocked_pipeline):
        mock_cov, mock_agg, predictions, combined = mocked_pipeline

        result = n_to_1_with_covariates(verbose=False, show_progress=False)

        assert len(result) == 4
        preds, comb, metrics, features = result
        assert preds is predictions
        assert comb is combined
        assert metrics == {"mae": 1.5}
        assert features == {"n_features": 7}
        # Aggregation is applied to the raw multi-output predictions.
        assert mock_agg.call_args.args[0] is predictions

    def test_defaults_to_module_weights_without_mutation(self, mocked_pipeline):
        _, mock_agg, _, _ = mocked_pipeline
        before = DEFAULT_WEIGHTS.copy()

        n_to_1_with_covariates(weights=None, verbose=False, show_progress=False)

        # The module-level default must never be mutated by a call.
        assert DEFAULT_WEIGHTS == before
        # agg_predict receives an equal-but-distinct copy of the defaults.
        passed = mock_agg.call_args.kwargs["weights"]
        assert passed == DEFAULT_WEIGHTS
        assert passed is not DEFAULT_WEIGHTS

    def test_custom_weights_forwarded_to_aggregation(self, mocked_pipeline):
        _, mock_agg, _, _ = mocked_pipeline
        custom = [1.0, -1.0, 0.5]

        n_to_1_with_covariates(weights=custom, verbose=False, show_progress=False)

        assert mock_agg.call_args.kwargs["weights"] == custom

    def test_estimator_forwarded_to_forecaster(self, mocked_pipeline):
        mock_cov, _, _, _ = mocked_pipeline
        estimator = MagicMock(name="estimator")

        n_to_1_with_covariates(estimator=estimator, verbose=False, show_progress=False)

        assert mock_cov.call_args.kwargs["estimator"] is estimator

    def test_forecast_parameters_forwarded(self, mocked_pipeline):
        mock_cov, _, _, _ = mocked_pipeline

        n_to_1_with_covariates(
            forecast_horizon=48,
            lags=12,
            window_size=96,
            verbose=False,
            show_progress=False,
        )

        kwargs = mock_cov.call_args.kwargs
        assert kwargs["forecast_horizon"] == 48
        assert kwargs["lags"] == 12
        assert kwargs["window_size"] == 96

    def test_extra_kwargs_forwarded(self, mocked_pipeline):
        mock_cov, _, _, _ = mocked_pipeline

        n_to_1_with_covariates(verbose=False, show_progress=False, freq="h")

        assert mock_cov.call_args.kwargs["freq"] == "h"

    def test_reraises_when_forecasting_fails(self, mocked_pipeline):
        mock_cov, mock_agg, _, _ = mocked_pipeline
        mock_cov.side_effect = RuntimeError("boom")

        # Fail-safe: the pipeline surfaces the failure instead of swallowing it.
        with pytest.raises(RuntimeError, match="boom"):
            n_to_1_with_covariates(verbose=False, show_progress=False)
        mock_agg.assert_not_called()


class TestMainFailSafe:
    """``main`` must propagate pipeline failures rather than exit silently."""

    @patch(f"{MOD}.n2n_predict_with_covariates", side_effect=RuntimeError("train fail"))
    @patch(f"{MOD}.fetch_data")
    def test_main_reraises_on_pipeline_failure(self, mock_fetch, _mock_cov):
        mock_fetch.return_value = pd.DataFrame(
            {"col0": range(50)},
            index=pd.date_range("2020-01-01", periods=50, freq="h"),
        )

        with pytest.raises(RuntimeError, match="train fail"):
            main(verbose=False)

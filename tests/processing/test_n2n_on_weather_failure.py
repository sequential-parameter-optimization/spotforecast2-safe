# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``on_weather_failure`` policy in n2n_predict_with_covariates.

Verifies the fail-safe contract documented on
`ConfigEntsoe.on_weather_failure`: ``"raise"`` (default) propagates
`WeatherFetchError`, ``"skip"`` logs a warning and continues with empty
weather features so the rest of the pipeline can still run.
"""

import tempfile
from unittest.mock import patch

import pytest

from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)
from spotforecast2_safe.weather.client import WeatherFetchError


def _raise_weather_fetch_error(*args, **kwargs):
    raise WeatherFetchError("simulated Open-Meteo outage")


def test_default_raises_when_weather_fetch_fails():
    """Default policy (``on_weather_failure='raise'``) propagates the error."""
    with patch(
        "spotforecast2_safe.processing.n2n_predict_with_covariates.get_weather_features",
        side_effect=_raise_weather_fetch_error,
    ):
        with pytest.raises(WeatherFetchError):
            n2n_predict_with_covariates(
                forecast_horizon=2,
                lags=4,
                window_size=8,
                force_train=True,
                model_dir=tempfile.mkdtemp(),
                verbose=False,
            )


def test_skip_continues_when_weather_fetch_fails(caplog):
    """``on_weather_failure='skip'`` swallows WeatherFetchError and continues."""
    with patch(
        "spotforecast2_safe.processing.n2n_predict_with_covariates.get_weather_features",
        side_effect=_raise_weather_fetch_error,
    ):
        with caplog.at_level("WARNING"):
            predictions, metadata, forecasters = n2n_predict_with_covariates(
                forecast_horizon=2,
                lags=4,
                window_size=8,
                force_train=True,
                model_dir=tempfile.mkdtemp(),
                verbose=False,
                on_weather_failure="skip",
            )

    assert predictions.shape[0] == 2
    assert "forecast_horizon" in metadata
    assert len(forecasters) >= 1
    assert any("on_weather_failure='skip'" in rec.message for rec in caplog.records)


def test_skip_only_swallows_weather_fetch_error():
    """Other exceptions from get_weather_features still propagate."""

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("unrelated bug")

    with patch(
        "spotforecast2_safe.processing.n2n_predict_with_covariates.get_weather_features",
        side_effect=raise_runtime_error,
    ):
        with pytest.raises(RuntimeError):
            n2n_predict_with_covariates(
                forecast_horizon=2,
                lags=4,
                window_size=8,
                force_train=True,
                model_dir=tempfile.mkdtemp(),
                verbose=False,
                on_weather_failure="skip",
            )

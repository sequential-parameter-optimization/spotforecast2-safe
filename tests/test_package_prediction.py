"""Comprehensive pytests for package_prediction method in ForecasterRecursiveModel."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)


@pytest.fixture
def tmp_data_extra(monkeypatch):
    """Setup a temporary data environment for package_prediction tests."""
    tmp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("SPOTFORECAST2_DATA", tmp_dir)

    interim_dir = Path(tmp_dir) / "interim"
    interim_dir.mkdir(parents=True)

    yield interim_dir

    shutil.rmtree(tmp_dir)


def create_mock_data(path: Path, columns=None):
    """Create a mock energy_load.csv file.

    Uses column names expected by ``load_timeseries`` / ``load_timeseries_forecast``.
    """
    if columns is None:
        columns = ["Actual Load", "Forecasted Load"]

    dates = pd.date_range("2022-01-01", periods=100, freq="h")
    df = pd.DataFrame(np.random.rand(100, len(columns)), index=dates, columns=columns)
    df.index.name = "Time (UTC)"
    df.to_csv(path / "energy_load.csv")
    return df


def test_package_prediction_success(tmp_data_extra):
    """Verify package_prediction returns expected structure on success."""
    create_mock_data(tmp_data_extra)

    model = ForecasterRecursiveModel(iteration=0, end_dev="2022-01-03 00:00+00:00")
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    result = model.package_prediction(predict_size=2)

    assert isinstance(result, dict)
    assert "train_actual" in result
    assert "future_actual" in result
    assert "train_pred" in result
    assert "future_pred" in result
    assert "metrics_train" in result
    assert "metrics_future" in result
    assert "metrics_future_one_day" in result
    assert "future_forecast" in result
    assert "metrics_forecast" in result


def test_package_prediction_no_forecast_column(tmp_data_extra):
    """Verify package_prediction works even if benchmark forecast is missing."""
    create_mock_data(tmp_data_extra, columns=["Actual Load"])

    model = ForecasterRecursiveModel(iteration=0, end_dev="2022-01-03 00:00+00:00")
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    result = model.package_prediction(predict_size=2)

    assert "train_actual" in result
    assert "future_forecast" not in result
    assert "metrics_forecast" not in result


def test_package_prediction_missing_actual_load(tmp_data_extra, caplog):
    """Verify package_prediction returns {} and logs error if Actual Load is missing."""
    create_mock_data(tmp_data_extra, columns=["Wrong Column"])

    model = ForecasterRecursiveModel(iteration=0)
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    with caplog.at_level("ERROR"):
        result = model.package_prediction()

    assert result == {}
    # load_timeseries raises KeyError which is caught by package_prediction's except
    assert "Error generating prediction package" in caplog.text


def test_package_prediction_custom_predict_size(tmp_data_extra):
    """Verify that predict_size override is respected."""
    create_mock_data(tmp_data_extra)

    # Default predict_size is 24, refit_size is 7 -> 168 hours
    model = ForecasterRecursiveModel(
        iteration=0, end_dev="2022-01-03 00:00+00:00", predict_size=24, refit_size=7
    )
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Override with predict_size=1 -> 7 hours
    result = model.package_prediction(predict_size=1)

    assert "future_actual" in result
    assert len(result["future_actual"]) == 7
    assert len(result["future_pred"]) == 7


def test_package_prediction_exception_handling(tmp_data_extra, monkeypatch, caplog):
    """Verify that exceptions during processing return {} and log the error."""
    create_mock_data(tmp_data_extra)

    model = ForecasterRecursiveModel(iteration=0)
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Force an exception during load_timeseries by removing the data file
    import os

    os.remove(tmp_data_extra / "energy_load.csv")

    with caplog.at_level("ERROR"):
        result = model.package_prediction()

    assert result == {}
    assert "Error generating prediction package" in caplog.text

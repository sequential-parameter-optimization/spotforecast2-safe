"""Tests for ForecasterRecursive __setstate__ compatibility."""

from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_setstate_restores_spotforecast_tags_when_missing():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    state = forecaster.__dict__.copy()

    state.pop("__spotforecast_tags__", None)
    delattr(forecaster, "__spotforecast_tags__")

    forecaster.__setstate__(state)

    assert hasattr(forecaster, "__spotforecast_tags__")
    tags = forecaster.__spotforecast_tags__
    assert tags["forecaster_name"] == "ForecasterRecursive"
    assert tags["forecasting_strategy"] == "recursive"
    assert tags["supports_exog"] is True

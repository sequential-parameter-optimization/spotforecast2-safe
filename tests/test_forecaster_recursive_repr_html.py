"""Tests for ForecasterRecursive HTML representation."""

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def _make_series(length: int = 30) -> pd.Series:
    return pd.Series(np.arange(length, dtype=float), name="y")


def test_repr_html_unfitted_contains_sections():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    html = forecaster._repr_html_()

    assert isinstance(html, str)
    assert "<style>" in html
    assert "<details open>" in html
    assert "<summary>General Information</summary>" in html
    assert "Training range:" in html
    assert "Not fitted" in html
    assert re.search(r"container-[a-zA-Z0-9]+", html) is not None


def test_repr_html_fitted_includes_training_and_exog():
    y = _make_series()
    exog = pd.DataFrame({"temp": np.arange(len(y), dtype=float)})
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)

    forecaster.fit(y=y, exog=exog)
    html = forecaster._repr_html_()

    assert "Not fitted" not in html
    assert "Exogenous included:" in html
    assert "True" in html
    assert "temp" in html
    assert "Estimator Parameters" in html

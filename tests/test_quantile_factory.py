# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the quantile-LightGBM probabilistic head factory."""

import types

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.multitask.factories import (
    DEFAULT_QUANTILES,
    predict_quantile_band,
    quantile_lgbm_forecaster_factory,
)


def _config():
    return types.SimpleNamespace(random_state=0, lags_consider=[1, 24], window_size=24)


class TestQuantileFactory:
    def test_builds_one_head_per_quantile(self):
        heads = quantile_lgbm_forecaster_factory(_config(), quantiles=[0.1, 0.5, 0.9])
        assert sorted(heads) == [0.1, 0.5, 0.9]
        assert all(isinstance(fc, ForecasterRecursive) for fc in heads.values())

    def test_objective_and_alpha_set(self):
        heads = quantile_lgbm_forecaster_factory(_config(), quantiles=[0.1, 0.9])
        for q in (0.1, 0.9):
            params = heads[q].regressor.get_params()
            assert params["objective"] == "quantile"
            assert params["alpha"] == q

    def test_default_quantiles(self):
        heads = quantile_lgbm_forecaster_factory(_config())
        assert tuple(sorted(heads)) == DEFAULT_QUANTILES

    def test_ascending_order(self):
        heads = quantile_lgbm_forecaster_factory(_config(), quantiles=[0.9, 0.1, 0.5])
        assert list(heads.keys()) == [0.1, 0.5, 0.9]

    @pytest.mark.parametrize(
        "bad",
        [[], [0.0, 0.5], [0.5, 1.0], [1.5], [0.5, 0.5]],
    )
    def test_invalid_quantiles_raise(self, bad):
        with pytest.raises(ValueError):
            quantile_lgbm_forecaster_factory(_config(), quantiles=bad)


class _StubForecaster:
    """Minimal fitted-forecaster stand-in returning fixed predictions."""

    def __init__(self, values):
        self.values = values

    def predict(self, steps, last_window=None, exog=None):
        return pd.Series(self.values[:steps], index=pd.RangeIndex(steps))


class TestPredictQuantileBand:
    def test_columns_and_index(self):
        heads = {
            0.1: _StubForecaster([1.0, 1.0]),
            0.5: _StubForecaster([2.0, 2.0]),
            0.9: _StubForecaster([3.0, 3.0]),
        }
        band = predict_quantile_band(heads, steps=2)
        assert band.columns.tolist() == ["q_0.1", "q_0.5", "q_0.9"]
        assert band.shape == (2, 3)

    def test_rearrangement_fixes_crossing(self):
        # Lower head predicts ABOVE the upper head (crossing).
        heads = {
            0.1: _StubForecaster([10.0]),
            0.5: _StubForecaster([5.0]),
            0.9: _StubForecaster([8.0]),
        }
        band = predict_quantile_band(heads, steps=1, enforce_monotonic=True)
        # Row [10, 5, 8] sorted ascending → [5, 8, 10].
        assert band["q_0.1"].iloc[0] == 5.0
        assert band["q_0.5"].iloc[0] == 8.0
        assert band["q_0.9"].iloc[0] == 10.0

    def test_no_rearrangement_keeps_raw(self):
        heads = {
            0.1: _StubForecaster([10.0]),
            0.5: _StubForecaster([5.0]),
            0.9: _StubForecaster([8.0]),
        }
        band = predict_quantile_band(heads, steps=1, enforce_monotonic=False)
        assert band["q_0.1"].iloc[0] == 10.0  # crossing preserved

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            predict_quantile_band({}, steps=1)


class TestIntegration:
    def test_fit_predict_band_is_monotonic(self):
        idx = pd.date_range("2023-01-01", periods=400, freq="h")
        y = pd.Series(
            50 + 10 * np.sin(np.arange(400) * 2 * np.pi / 24), index=idx, name="y"
        )
        heads = quantile_lgbm_forecaster_factory(_config(), quantiles=[0.1, 0.5, 0.9])
        for fc in heads.values():
            fc.fit(y=y)
        band = predict_quantile_band(heads, steps=6)
        assert (band["q_0.1"] <= band["q_0.5"] + 1e-9).all()
        assert (band["q_0.5"] <= band["q_0.9"] + 1e-9).all()
        assert band.shape == (6, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

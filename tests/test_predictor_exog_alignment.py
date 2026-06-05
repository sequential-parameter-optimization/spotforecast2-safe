# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the exog_future alignment guard in
``build_prediction_package`` (manager/predictor.py).

Incident 2026-06-05/06: ``exo_pred`` was built for a window ~36 h after the
true prediction window (``data_end`` contaminated by trailing exogenous-only
rows) and the forecaster consumed it positionally — the live forecast came out
phase-rolled by ~+9 h. ``build_prediction_package`` now realigns
``exog_future`` to ``expand_index(forecaster.last_window_.index, steps)`` and
raises when the prediction window is not covered.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.manager.predictor import build_prediction_package

N_TRAIN = 100
N_LAGS = 24
N_FUTURE = 24


@pytest.fixture
def train_idx():
    return pd.date_range("2024-01-01", periods=N_TRAIN, freq="h", tz="UTC")


@pytest.fixture
def y_train(train_idx):
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(50.0, 5.0, N_TRAIN), index=train_idx, name="load")


@pytest.fixture
def future_idx(train_idx):
    return pd.date_range(
        train_idx[-1] + pd.Timedelta(hours=1), periods=N_FUTURE, freq="h", tz="UTC"
    )


@pytest.fixture
def forecaster(train_idx, future_idx):
    """Mock forecaster with a REAL DatetimeIndex on ``last_window_`` so the
    alignment guard engages (a bare MagicMock attribute would bypass it)."""
    internal_idx = train_idx[N_LAGS:]
    n_internal = len(internal_idx)

    fc = MagicMock()
    fc.create_train_X_y.return_value = (
        np.ones((n_internal, N_LAGS)),
        pd.Series(np.full(n_internal, 50.0), index=internal_idx),
    )
    fc.estimator.predict.return_value = np.full(n_internal, 49.0)
    fc.predict.return_value = pd.Series(
        np.full(N_FUTURE, 48.0), index=future_idx, name="pred"
    )
    fc.last_window_ = pd.DataFrame(
        {"load": np.full(N_LAGS, 50.0)}, index=train_idx[-N_LAGS:]
    )
    return fc


def _exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"feat1": np.arange(len(index), dtype=float), "feat2": 1.0}, index=index
    )


class TestExogAlignmentGuard:
    def test_aligned_exog_passes_through_unchanged(
        self, forecaster, y_train, future_idx
    ):
        exog_future = _exog(future_idx)
        build_prediction_package(
            forecaster, "load", y_train, N_FUTURE, exog_future=exog_future
        )
        passed = forecaster.predict.call_args.kwargs["exog"]
        pd.testing.assert_frame_equal(passed, exog_future)

    def test_misaligned_uncovered_window_raises(self, forecaster, y_train, future_idx):
        """The incident shape: exog labelled ~36 h after the prediction window
        and NOT covering it -> hard error instead of a silent phase roll."""
        shifted = _exog(future_idx + pd.Timedelta(hours=36))
        with pytest.raises(ValueError, match="not aligned with the prediction window"):
            build_prediction_package(
                forecaster, "load", y_train, N_FUTURE, exog_future=shifted
            )

    def test_superset_frame_is_realigned(self, forecaster, y_train, future_idx):
        """A frame that covers the window but starts early is realigned (with a
        warning) instead of being consumed positionally."""
        wide_idx = pd.date_range(
            future_idx[0] - pd.Timedelta(hours=2),
            future_idx[-1],
            freq="h",
            tz="UTC",
        )
        build_prediction_package(
            forecaster, "load", y_train, N_FUTURE, exog_future=_exog(wide_idx)
        )
        passed = forecaster.predict.call_args.kwargs["exog"]
        assert passed.index.equals(future_idx)
        assert not passed.isna().any().any()

    def test_none_exog_keeps_legacy_path(self, forecaster, y_train):
        build_prediction_package(forecaster, "load", y_train, N_FUTURE)
        assert forecaster.predict.call_args.kwargs["exog"] is None

    def test_mock_last_window_bypasses_guard(self, y_train, future_idx, train_idx):
        """Forecasters whose ``last_window_`` lacks a DatetimeIndex (e.g. test
        mocks, exotic estimators) keep the legacy pass-through behaviour."""
        internal_idx = train_idx[N_LAGS:]
        fc = MagicMock()  # last_window_.index is a MagicMock, not a DatetimeIndex
        fc.create_train_X_y.return_value = (
            np.ones((len(internal_idx), N_LAGS)),
            pd.Series(np.full(len(internal_idx), 50.0), index=internal_idx),
        )
        fc.estimator.predict.return_value = np.full(len(internal_idx), 49.0)
        fc.predict.return_value = pd.Series(
            np.full(N_FUTURE, 48.0), index=future_idx, name="pred"
        )
        shifted = _exog(future_idx + pd.Timedelta(hours=36))
        build_prediction_package(fc, "load", y_train, N_FUTURE, exog_future=shifted)
        passed = fc.predict.call_args.kwargs["exog"]
        pd.testing.assert_frame_equal(passed, shifted)

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for multitask.guards.assert_no_leakage.

Uses minimal duck-typed SimpleNamespace stubs rather than a full MultiTask
instance so the tests are fast and free of heavy I/O.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from spotforecast2_safe.exceptions import LeakageError
from spotforecast2_safe.multitask.guards import assert_no_leakage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDX = pd.date_range("2026-06-01", periods=3, freq="h", tz="UTC")
FORBIDDEN = {"Forecasted Load", "Actual Load"}


def _make_mt(
    *,
    targets=("load",),
    fitted_features=("lag_1", "lag_24", "hour_sin"),
    data_with_exog_cols=("load", "lag_1"),
    exog_feature_names=("lag_1",),
    task="defaults",
) -> object:
    """Build a minimal duck-typed mt stub."""
    run_state = SimpleNamespace(targets=list(targets))
    estimator = SimpleNamespace(feature_name_=list(fitted_features))
    forecaster = SimpleNamespace(estimator=estimator)
    results = {task: {t: {"forecaster": forecaster} for t in targets}}
    data = {col: [1.0, 2.0, 3.0] for col in data_with_exog_cols}
    data_with_exog = pd.DataFrame(data, index=IDX)
    return SimpleNamespace(
        run_state=run_state,
        results=results,
        data_with_exog=data_with_exog,
        exog_feature_names=list(exog_feature_names),
    )


# ---------------------------------------------------------------------------
# Passing cases
# ---------------------------------------------------------------------------


class TestAssertNoLeakagePassing:
    def test_clean_mt_passes(self):
        """No forbidden columns anywhere -> no exception."""
        mt = _make_mt(
            fitted_features=["lag_1", "lag_24"],
            data_with_exog_cols=["load", "lag_1"],
            exog_feature_names=["lag_1"],
        )
        assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")

    def test_empty_forbidden_set_always_passes(self):
        """An empty forbidden set can never produce a violation."""
        mt = _make_mt(
            fitted_features=["Forecasted Load"],  # would leak under normal rules
            data_with_exog_cols=["load", "Forecasted Load"],
            exog_feature_names=["Forecasted Load"],
        )
        assert_no_leakage(mt, forbidden=set(), task="defaults")

    def test_multiple_targets_all_clean(self):
        """Multi-target pipeline with no forbidden columns passes."""
        run_state = SimpleNamespace(targets=["zone_A", "zone_B"])
        estimator = SimpleNamespace(feature_name_=["lag_1", "hour_sin"])
        forecaster = SimpleNamespace(estimator=estimator)
        results = {
            "defaults": {
                "zone_A": {"forecaster": forecaster},
                "zone_B": {"forecaster": forecaster},
            }
        }
        mt = SimpleNamespace(
            run_state=run_state,
            results=results,
            data_with_exog=pd.DataFrame(
                {"zone_A": [1.0], "zone_B": [2.0], "lag_1": [0.0]},
                index=IDX[:1],
            ),
            exog_feature_names=["lag_1"],
        )
        assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")


# ---------------------------------------------------------------------------
# Violation in training frame
# ---------------------------------------------------------------------------


class TestLeakInTrainingFrame:
    def test_forbidden_column_in_training_frame_raises(self):
        """Forbidden column in data_with_exog -> LeakageError naming the surface."""
        mt = _make_mt(
            fitted_features=["lag_1"],
            data_with_exog_cols=["load", "Forecasted Load"],  # LEAK
            exog_feature_names=["lag_1"],
        )
        with pytest.raises(LeakageError) as exc_info:
            assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")
        msg = str(exc_info.value)
        assert "training frame" in msg
        assert "Forecasted Load" in msg

    def test_message_contains_task_name(self):
        mt = _make_mt(
            data_with_exog_cols=["load", "Actual Load"],
            exog_feature_names=["lag_1"],
            task="spotoptim",
        )
        with pytest.raises(LeakageError) as exc_info:
            assert_no_leakage(mt, forbidden=FORBIDDEN, task="spotoptim")
        assert "spotoptim" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Violation in selected exogenous features
# ---------------------------------------------------------------------------


class TestLeakInExogFeatures:
    def test_forbidden_column_in_exog_names_raises(self):
        """Forbidden column in exog_feature_names -> LeakageError naming the surface."""
        mt = _make_mt(
            fitted_features=["lag_1"],
            data_with_exog_cols=["load", "lag_1"],
            exog_feature_names=["lag_1", "Forecasted Load"],  # LEAK
        )
        with pytest.raises(LeakageError) as exc_info:
            assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")
        msg = str(exc_info.value)
        assert "selected exogenous features" in msg
        assert "Forecasted Load" in msg


# ---------------------------------------------------------------------------
# Violation in fitted model features
# ---------------------------------------------------------------------------


class TestLeakInFittedFeatures:
    def test_forbidden_column_in_fitted_features_raises(self):
        """Forbidden column in estimator.feature_name_ -> LeakageError naming the surface."""
        mt = _make_mt(
            fitted_features=["lag_1", "Forecasted Load"],  # LEAK
            data_with_exog_cols=["load", "lag_1"],
            exog_feature_names=["lag_1"],
        )
        with pytest.raises(LeakageError) as exc_info:
            assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")
        msg = str(exc_info.value)
        assert "fitted model features" in msg
        assert "Forecasted Load" in msg


# ---------------------------------------------------------------------------
# Multiple simultaneous violations
# ---------------------------------------------------------------------------


class TestMultipleSurfaceViolations:
    def test_violations_on_all_three_surfaces_raised_together(self):
        """All three surfaces leaking -> single LeakageError naming all."""
        mt = _make_mt(
            fitted_features=["Forecasted Load"],
            data_with_exog_cols=["load", "Actual Load"],
            exog_feature_names=["Forecasted Load"],
        )
        with pytest.raises(LeakageError) as exc_info:
            assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")
        msg = str(exc_info.value)
        # At least two surfaces should be mentioned
        surfaces_mentioned = sum(
            s in msg
            for s in (
                "training frame",
                "selected exogenous features",
                "fitted model features",
            )
        )
        assert surfaces_mentioned == 3


# ---------------------------------------------------------------------------
# Unreadable fitted features -> RuntimeError
# ---------------------------------------------------------------------------


class TestUnreadableFittedFeatures:
    def test_missing_feature_name_attr_raises_runtime_error(self):
        """Estimator without feature_name_ attribute must raise RuntimeError."""
        run_state = SimpleNamespace(targets=["load"])
        estimator_no_attr = SimpleNamespace()  # no feature_name_
        forecaster = SimpleNamespace(estimator=estimator_no_attr)
        results = {"defaults": {"load": {"forecaster": forecaster}}}
        mt = SimpleNamespace(
            run_state=run_state,
            results=results,
            data_with_exog=pd.DataFrame({"load": [1.0]}, index=IDX[:1]),
            exog_feature_names=["lag_1"],
        )
        with pytest.raises(RuntimeError, match="Cannot read fitted feature names"):
            assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")

    def test_missing_task_key_raises_runtime_error(self):
        """Missing task key in results must raise RuntimeError (unreadable features)."""
        run_state = SimpleNamespace(targets=["load"])
        estimator = SimpleNamespace(feature_name_=["lag_1"])
        forecaster = SimpleNamespace(estimator=estimator)
        results = {"other_task": {"load": {"forecaster": forecaster}}}
        mt = SimpleNamespace(
            run_state=run_state,
            results=results,
            data_with_exog=pd.DataFrame({"load": [1.0]}, index=IDX[:1]),
            exog_feature_names=["lag_1"],
        )
        with pytest.raises(RuntimeError, match="Cannot read fitted feature names"):
            assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")

    def test_none_estimator_raises_runtime_error(self):
        """``None`` estimator must raise RuntimeError rather than silent skip."""
        run_state = SimpleNamespace(targets=["load"])
        forecaster = SimpleNamespace(estimator=None)
        results = {"defaults": {"load": {"forecaster": forecaster}}}
        mt = SimpleNamespace(
            run_state=run_state,
            results=results,
            data_with_exog=pd.DataFrame({"load": [1.0]}, index=IDX[:1]),
            exog_feature_names=["lag_1"],
        )
        with pytest.raises(RuntimeError, match="Cannot read fitted feature names"):
            assert_no_leakage(mt, forbidden=FORBIDDEN, task="defaults")

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``forecaster.base.ForecasterBase`` shared behaviour.

Exercised through the concrete ``ForecasterRecursive``: the repr-preprocessing
truncation branches, the ``regressor`` -> ``estimator`` pickle migration in
``__setstate__``, the deprecated ``regressor`` property FutureWarning, and
``get_tags`` / ``summary``.
"""

import pickle

import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def _fc():
    return ForecasterRecursive(estimator=Ridge(), lags=3)


class TestPreprocessRepr:
    def test_plain_estimator_params_string(self):
        params, tr, sn, en, ts = _fc()._preprocess_repr(estimator=Ridge(alpha=0.5))
        assert "alpha" in params
        assert tr is None and sn is None and en is None and ts is None

    def test_pipeline_estimator_filters_step_params(self):
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
        params, *_ = _fc()._preprocess_repr(estimator=pipe)
        assert "ridge__" in params or "scaler__" in params

    def test_training_range_truncation(self):
        import pandas as pd

        tr = {
            f"s{i}": pd.Index(pd.to_datetime(["2020-01-01", "2020-01-02"]))
            for i in range(12)
        }
        _, tr_str, *_ = _fc()._preprocess_repr(training_range_=tr)
        assert "..." in tr_str

    def test_series_names_truncation(self):
        _, _, sn, _, _ = _fc()._preprocess_repr(
            series_names_in_=[f"s{i}" for i in range(60)]
        )
        assert "..." in sn

    def test_exog_names_truncation(self):
        *_, en, _ = _fc()._preprocess_repr(exog_names_in_=[f"e{i}" for i in range(60)])
        assert "..." in en

    def test_transformer_series_dict_truncation(self):
        ts = {f"s{i}": StandardScaler() for i in range(12)}
        *_, ts_str = _fc()._preprocess_repr(transformer_series=ts)
        assert "..." in ts_str

    def test_transformer_series_non_dict(self):
        *_, ts_str = _fc()._preprocess_repr(transformer_series=StandardScaler())
        assert "StandardScaler" in ts_str


class TestFormatTextRepr:
    def test_short_text_unchanged(self):
        assert _fc()._format_text_repr("short") == "short"

    def test_long_text_is_wrapped(self):
        out = _fc()._format_text_repr("x" * 100)
        assert out.startswith("\n    ")


class TestTagsAndSummary:
    def test_get_tags_returns_dict(self):
        tags = _fc().get_tags()
        assert isinstance(tags, dict) and len(tags) > 0

    def test_summary_prints_repr(self, capsys):
        _fc().summary()
        assert "ForecasterRecursive" in capsys.readouterr().out


class TestSetStateMigration:
    def test_regressor_key_migrated_to_estimator(self):
        fc = _fc()
        state = fc.__dict__.copy()
        # Simulate an old pickle that stored the estimator under "regressor".
        state["regressor"] = state.pop("estimator")

        restored = ForecasterRecursive.__new__(ForecasterRecursive)
        restored.__setstate__(state)

        assert "estimator" in restored.__dict__
        assert "regressor" not in restored.__dict__

    def test_pickle_roundtrip(self):
        loaded = pickle.loads(pickle.dumps(_fc()))
        assert "estimator" in loaded.__dict__


class TestRegressorDeprecation:
    def test_regressor_warns_and_returns_estimator(self):
        fc = _fc()
        with pytest.warns(FutureWarning, match="regressor"):
            est = fc.regressor
        assert est is fc.estimator

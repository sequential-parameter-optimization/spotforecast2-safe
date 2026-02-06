import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing import outlier as outlier_module


class DummyIForest:
    def __init__(self, contamination=0.1, random_state=1234):
        self.contamination = contamination
        self.random_state = random_state
        self.fit_predict_calls = []

    def fit_predict(self, X):
        self.fit_predict_calls.append(X.copy())
        n = len(X)
        labels = np.ones(n, dtype=int)
        if n:
            labels[0] = -1
        return labels


def test_mark_outliers_marks_first_row_nan(monkeypatch):
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
        }
    )

    dummy = DummyIForest()

    def _factory(contamination=0.1, random_state=1234):
        dummy.contamination = contamination
        dummy.random_state = random_state
        return dummy

    monkeypatch.setattr(outlier_module, "IsolationForest", _factory)

    cleaned, labels = outlier_module.mark_outliers(
        data.copy(), contamination=0.2, random_state=42, verbose=False
    )

    assert np.isnan(cleaned.loc[0, "a"])
    assert np.isnan(cleaned.loc[0, "b"])
    assert cleaned.loc[1, "a"] == 2.0
    assert cleaned.loc[2, "b"] == 30.0

    assert labels[0] == -1
    assert (labels[1:] == 1).all()
    assert dummy.contamination == 0.2
    assert dummy.random_state == 42
    assert len(dummy.fit_predict_calls) == 2


def test_mark_outliers_verbose_prints(monkeypatch, capsys):
    data = pd.DataFrame({"a": [1.0, 2.0]})

    dummy = DummyIForest()

    def _factory(contamination=0.1, random_state=1234):
        return dummy

    monkeypatch.setattr(outlier_module, "IsolationForest", _factory)

    outlier_module.mark_outliers(data.copy(), verbose=True)
    captured = capsys.readouterr().out

    assert "Column 'a': Marked" in captured


def test_mark_outliers_returns_last_column_labels(monkeypatch):
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )

    class DummyIForestSequence:
        def __init__(self, labels):
            self.labels = labels
            self.calls = 0

        def fit_predict(self, X):
            result = self.labels[self.calls]
            self.calls += 1
            return result

    sequence = DummyIForestSequence(labels=[np.array([1, 1, 1]), np.array([-1, 1, 1])])

    def _factory(contamination=0.1, random_state=1234):
        return sequence

    monkeypatch.setattr(outlier_module, "IsolationForest", _factory)

    _, labels = outlier_module.mark_outliers(data.copy())
    assert labels.tolist() == [-1, 1, 1]


def test_mark_outliers_raises_on_non_dataframe():
    with pytest.raises(AttributeError):
        outlier_module.mark_outliers([1, 2, 3])


def test_manual_outlier_removal_both_thresholds():
    data = pd.DataFrame({"a": [10.0, 100.0, 1000.0]})

    cleaned, n_outliers = outlier_module.manual_outlier_removal(
        data.copy(),
        column="a",
        lower_threshold=50,
        upper_threshold=700,
    )

    assert n_outliers == 2
    assert np.isnan(cleaned.loc[0, "a"])
    assert cleaned.loc[1, "a"] == 100.0
    assert np.isnan(cleaned.loc[2, "a"])


def test_manual_outlier_removal_lower_only():
    data = pd.DataFrame({"a": [10.0, 100.0, 1000.0]})

    cleaned, n_outliers = outlier_module.manual_outlier_removal(
        data.copy(),
        column="a",
        lower_threshold=50,
        upper_threshold=None,
    )

    assert n_outliers == 1
    assert np.isnan(cleaned.loc[0, "a"])
    assert cleaned.loc[1, "a"] == 100.0
    assert cleaned.loc[2, "a"] == 1000.0


def test_manual_outlier_removal_upper_only():
    data = pd.DataFrame({"a": [10.0, 100.0, 1000.0]})

    cleaned, n_outliers = outlier_module.manual_outlier_removal(
        data.copy(),
        column="a",
        lower_threshold=None,
        upper_threshold=700,
    )

    assert n_outliers == 1
    assert cleaned.loc[0, "a"] == 10.0
    assert cleaned.loc[1, "a"] == 100.0
    assert np.isnan(cleaned.loc[2, "a"])


def test_manual_outlier_removal_no_thresholds_verbose(capsys):
    data = pd.DataFrame({"a": [10.0, 100.0, 1000.0]})

    cleaned, n_outliers = outlier_module.manual_outlier_removal(
        data.copy(),
        column="a",
        lower_threshold=None,
        upper_threshold=None,
        verbose=True,
    )

    captured = capsys.readouterr().out

    assert n_outliers == 0
    assert cleaned.equals(data)
    assert "No thresholds provided for a" in captured

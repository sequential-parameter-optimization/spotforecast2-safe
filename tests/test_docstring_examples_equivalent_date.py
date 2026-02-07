import numpy as np
import pandas as pd

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def _make_series(length: int) -> pd.Series:
    return pd.Series(
        data=np.arange(length, dtype=float),
        index=pd.date_range(start="2022-01-01", periods=length, freq="D"),
    )


def test_example_fit_and_predict():
    data = _make_series(14)
    forecaster = ForecasterEquivalentDate(offset=7)
    forecaster.fit(y=data)

    preds = forecaster.predict(steps=3)
    expected_index = pd.date_range(start="2022-01-15", periods=3, freq="D")
    expected_values = np.array([7.0, 8.0, 9.0])

    pd.testing.assert_index_equal(preds.index, expected_index)
    np.testing.assert_allclose(preds.to_numpy(), expected_values)


def test_example_repr_contains_expected_lines():
    data = _make_series(14)
    forecaster = ForecasterEquivalentDate(offset=7)
    forecaster.fit(y=data)

    repr_text = repr(forecaster)
    assert "ForecasterEquivalentDate" in repr_text
    assert "Offset: 7" in repr_text
    assert "Number of offsets: 1" in repr_text
    assert "Window size: 7" in repr_text


def test_example_repr_html_contains_expected_markers():
    data = _make_series(14)
    forecaster = ForecasterEquivalentDate(offset=7)
    forecaster.fit(y=data)

    html = forecaster._repr_html_()
    assert "<style>" in html
    assert "ForecasterEquivalentDate" in html


def test_example_binning_in_sample_residuals():
    data = _make_series(21)
    forecaster = ForecasterEquivalentDate(
        offset=7, binner_kwargs={"n_bins": 2, "random_state": 123}
    )
    forecaster.fit(y=data, store_in_sample_residuals=True)

    expected = np.full(14, 7.0)
    np.testing.assert_array_equal(forecaster.in_sample_residuals_, expected)

    assert set(forecaster.in_sample_residuals_by_bin_.keys()) == {0, 1}
    np.testing.assert_array_equal(
        forecaster.in_sample_residuals_by_bin_[0], np.full(7, 7.0)
    )
    np.testing.assert_array_equal(
        forecaster.in_sample_residuals_by_bin_[1], np.full(7, 7.0)
    )


def test_example_predict_interval_structure_and_preds():
    data = _make_series(14)
    forecaster = ForecasterEquivalentDate(offset=7)
    forecaster.fit(y=data, store_in_sample_residuals=True)

    intervals = forecaster.predict_interval(steps=3, interval=0.8)

    assert list(intervals.columns) == ["pred", "lower_bound", "upper_bound"]
    expected_index = pd.date_range(start="2022-01-15", periods=3, freq="D")
    pd.testing.assert_index_equal(intervals.index, expected_index)
    np.testing.assert_allclose(intervals["pred"].to_numpy(), [7.0, 8.0, 9.0])
    assert (intervals["lower_bound"] <= intervals["pred"]).all()
    assert (intervals["upper_bound"] >= intervals["pred"]).all()


def test_example_set_in_sample_residuals():
    data = _make_series(14)
    forecaster = ForecasterEquivalentDate(offset=7)
    forecaster.fit(y=data)

    forecaster.set_in_sample_residuals(y=data, random_state=123)
    assert forecaster.in_sample_residuals_.shape == (7,)


def test_example_set_out_sample_residuals():
    data = _make_series(21)
    forecaster = ForecasterEquivalentDate(offset=7)
    forecaster.fit(y=data)

    preds = forecaster.predict(steps=7)
    y_true = pd.Series(data[-7:].to_numpy(), index=preds.index)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=preds)
    assert forecaster.out_sample_residuals_.shape == (7,)


def test_example_get_tags():
    forecaster = ForecasterEquivalentDate(offset=7)
    tags = forecaster.get_tags()
    assert sorted(tags.keys())[:3] == [
        "allowed_input_types_exog",
        "allowed_input_types_series",
        "forecaster_name",
    ]


def test_example_summary_prints(capsys):
    data = _make_series(14)
    forecaster = ForecasterEquivalentDate(offset=7)
    forecaster.fit(y=data)

    forecaster.summary()
    captured = capsys.readouterr()
    assert "ForecasterEquivalentDate" in captured.out

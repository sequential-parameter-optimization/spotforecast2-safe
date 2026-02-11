import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures


def _make_series(length: int, seed: int = 0, name: str = "y") -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=length), name=name)


def _make_exog(length: int, index: pd.Index, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"temp": rng.normal(size=length)}, index=index)


def test_doc_example_basic_forecaster():
    y = _make_series(100)
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=10)
    forecaster.fit(y)
    predictions = forecaster.predict(steps=5)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 5


def test_doc_example_window_features_and_transformations():
    y = _make_series(100, seed=2)
    forecaster = ForecasterRecursive(
        estimator=RandomForestRegressor(n_estimators=100, random_state=0),
        lags=[1, 7, 30],
        window_features=[RollingFeatures(stats="mean", window_sizes=7)],
        transformer_y=StandardScaler(),
        differentiation=1,
    )
    forecaster.fit(y)
    predictions = forecaster.predict(steps=10)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 10


def test_doc_example_with_exog_variables():
    y = _make_series(100, seed=3, name="target")
    exog = _make_exog(100, index=y.index, seed=4)
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=7, forecaster_id="id")
    forecaster.fit(y, exog)

    exog_future = _make_exog(5, index=pd.RangeIndex(start=100, stop=105), seed=5)
    predictions = forecaster.predict(steps=5, exog=exog_future)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 5
    pd.testing.assert_index_equal(predictions.index, exog_future.index)


def test_doc_example_probabilistic_config():
    y = _make_series(100, seed=6)
    forecaster = ForecasterRecursive(
        estimator=GradientBoostingRegressor(random_state=0),
        lags=14,
        binner_kwargs={"n_bins": 15, "method": "linear"},
    )
    forecaster.fit(y, store_in_sample_residuals=True)
    predictions = forecaster.predict(steps=5)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 5


def test_doc_example_repr_html():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    html = forecaster._repr_html_()
    assert "<div" in html
    assert "ForecasterRecursive" in html


def test_doc_example_setstate():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    pickled = pickle.dumps(forecaster)
    unpickled_forecaster = pickle.loads(pickled)
    assert hasattr(unpickled_forecaster, "__spotforecast_tags__")


def test_doc_example_create_lags():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    y = np.arange(10)
    train_index = pd.RangeIndex(start=3, stop=10)
    X_data, y_data = forecaster._create_lags(
        y=y, X_as_pandas=True, train_index=train_index
    )

    assert isinstance(X_data, pd.DataFrame)
    assert X_data.shape == (7, 3)
    assert y_data.shape == (7,)


def test_doc_example_create_window_features():
    y = pd.Series(np.arange(30), name="y")
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    train_index = y.index[3:]
    X_train_window_features, feature_names = forecaster._create_window_features(
        y=y, train_index=train_index, X_as_pandas=True
    )

    assert isinstance(X_train_window_features[0], pd.DataFrame)
    assert X_train_window_features[0].shape[0] == len(train_index)
    assert (X_train_window_features[0].index == train_index).all()
    assert len(feature_names) == X_train_window_features[0].shape[1]


def test_doc_example_create_train_X_y():
    y = pd.Series(np.arange(30), name="y")
    exog = _make_exog(30, index=y.index, seed=7)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    (
        X_train,
        y_train,
        exog_names_in_,
        window_features_names,
        exog_names_out,
        feature_names,
        exog_dtypes_in_,
        exog_dtypes_out_,
    ) = forecaster._create_train_X_y(y=y, exog=exog)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert (
        feature_names == forecaster.lags_names + window_features_names + exog_names_out
    )
    assert exog_names_in_ == exog_names_out
    assert exog_dtypes_in_ is not None
    assert exog_dtypes_out_ is not None


def test_doc_example_create_train_X_y_public():
    y = pd.Series(np.arange(30), name="y")
    exog = _make_exog(30, index=y.index, seed=8)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    (
        X_train,
        y_train,
        exog_names_in_,
        window_features_names,
        exog_names_out,
        feature_names,
        exog_dtypes_in_,
        exog_dtypes_out_,
    ) = forecaster.create_train_X_y(y=y, exog=exog)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert (
        feature_names == forecaster.lags_names + window_features_names + exog_names_out
    )
    assert exog_names_in_ == exog_names_out
    assert exog_dtypes_in_ is not None
    assert exog_dtypes_out_ is not None


def test_doc_example_train_test_split_one_step_ahead():
    y = pd.Series(np.arange(30), name="y")
    exog = _make_exog(30, index=y.index, seed=9)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    X_train, y_train, X_test, y_test = forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=20, exog=exog
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)


def test_doc_example_get_params():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    params = forecaster.get_params()

    assert "estimator" in params
    assert "lags" in params
    assert "window_features" in params


def test_doc_example_set_params():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.set_params(estimator__fit_intercept=False)

    assert forecaster.estimator.get_params()["fit_intercept"] is False


def test_doc_example_fit():
    y = pd.Series(np.arange(30), name="y")
    exog = _make_exog(30, index=y.index, seed=10)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)

    assert forecaster.is_fitted is True
    assert forecaster.last_window_ is not None


def test_doc_example_create_predict_inputs():
    y = pd.Series(np.arange(30), name="y")
    exog = _make_exog(30, index=y.index, seed=11)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    forecaster.fit(y=y, exog=exog)
    last_window = y.iloc[-3:]
    exog_future = _make_exog(5, index=pd.RangeIndex(start=30, stop=35), seed=12)

    last_window_values, exog_values, prediction_index, exog_index = (
        forecaster._create_predict_inputs(
            steps=5, last_window=last_window, exog=exog_future, check_inputs=True
        )
    )

    assert isinstance(last_window_values, np.ndarray)
    assert isinstance(exog_values, np.ndarray)
    assert isinstance(prediction_index, pd.Index)
    assert isinstance(exog_index, pd.Index)
    pd.testing.assert_index_equal(prediction_index, exog_future.index)


def test_doc_example_recursive_predict():
    y = pd.Series(np.arange(30), name="y")
    exog = _make_exog(30, index=y.index, seed=13)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    forecaster.fit(y=y, exog=exog)
    last_window = y.iloc[-3:]
    exog_future = _make_exog(5, index=pd.RangeIndex(start=30, stop=35), seed=14)

    last_window_values, exog_values, _, _ = forecaster._create_predict_inputs(
        steps=5, last_window=last_window, exog=exog_future, check_inputs=True
    )
    predictions = forecaster._recursive_predict(
        steps=5, last_window_values=last_window_values, exog_values=exog_values
    )

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (5,)


def test_doc_example_predict():
    y = pd.Series(np.arange(30), name="y")
    exog = _make_exog(30, index=y.index, seed=15)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    forecaster.fit(y=y, exog=exog)
    last_window = y.iloc[-3:]
    exog_future = _make_exog(5, index=pd.RangeIndex(start=30, stop=35), seed=16)

    predictions = forecaster.predict(
        steps=5, last_window=last_window, exog=exog_future, check_inputs=True
    )

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 5
    pd.testing.assert_index_equal(predictions.index, exog_future.index)

import numpy as np
import pandas as pd
import pytest
from spotforecast2_safe.preprocessing.linearly_interpolate_ts import LinearlyInterpolateTS


def test_linear_interpolate_ts_series():
    """Test linear interpolation with a pandas Series."""
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    interpolator = LinearlyInterpolateTS()
    s_filled = interpolator.transform(s)
    
    expected = [1.0, 2.0, 3.0, 3.0]
    assert s_filled.tolist() == expected
    assert s_filled.dtype == "float64"


def test_linear_interpolate_ts_dataframe():
    """Test linear interpolation with a pandas DataFrame."""
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [np.nan, 10.0, 20.0]
    })
    interpolator = LinearlyInterpolateTS()
    df_filled = interpolator.transform(df)
    
    assert df_filled["a"].tolist() == [1.0, 2.0, 3.0]
    # Leading NaNs are NOT filled by ffill(), only trailing/middle ones after interpolation.
    assert np.isnan(df_filled["b"].iloc[0])
    assert df_filled["b"].iloc[1] == 10.0
    assert df_filled["b"].iloc[2] == 20.0


def test_linear_interpolate_ts_docstring_example():
    """Test the example provided in the docstring."""
    # The docstring example:
    # >>> s = pd.Series([1.0, np.nan, 3.0, np.nan])
    # >>> interpolator = LinearlyInterpolateTS()
    # >>> s_filled = interpolator.fit_transform(s)
    # >>> s_filled.tolist()
    # [1.0, 2.0, 3.0, 3.0]
    
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    interpolator = LinearlyInterpolateTS()
    s_filled = interpolator.fit_transform(s)
    assert s_filled.tolist() == [1.0, 2.0, 3.0, 3.0]


def test_linear_interpolate_ts_no_nan():
    """Test behavior with no missing values."""
    s = pd.Series([1.0, 2.0, 3.0])
    interpolator = LinearlyInterpolateTS()
    s_filled = interpolator.transform(s)
    assert s_filled.tolist() == [1.0, 2.0, 3.0]

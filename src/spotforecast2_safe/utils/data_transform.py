"""
Data transformation utilities for time series forecasting.

This module provides functions for normalizing and transforming data formats.
"""

from typing import Union
import numpy as np
import pandas as pd


def input_to_frame(
    data: Union[pd.Series, pd.DataFrame], input_name: str
) -> pd.DataFrame:
    """
    Convert input data to a pandas DataFrame.

    This function ensures consistent DataFrame format for internal processing.
    If data is already a DataFrame, it's returned as-is. If it's a Series,
    it's converted to a single-column DataFrame.

    Args:
        data: Input data as pandas Series or DataFrame.
        input_name: Name of the input data type. Accepted values are:
            - 'y': Target time series
            - 'last_window': Last window for prediction
            - 'exog': Exogenous variables

    Returns:
        DataFrame version of the input data. For Series input, uses the series
        name if available, otherwise uses a default name based on input_name.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2.utils.data_transform import input_to_frame
        >>>
        >>> # Series with name
        >>> y = pd.Series([1, 2, 3], name="sales")
        >>> df = input_to_frame(y, input_name="y")
        >>> df.columns.tolist()
        ['sales']
        >>>
        >>> # Series without name (uses default)
        >>> y_no_name = pd.Series([1, 2, 3])
        >>> df = input_to_frame(y_no_name, input_name="y")
        >>> df.columns.tolist()
        ['y']
        >>>
        >>> # DataFrame (returned as-is)
        >>> df_input = pd.DataFrame({"temp": [20, 21], "humidity": [50, 55]})
        >>> df_output = input_to_frame(df_input, input_name="exog")
        >>> df_output.columns.tolist()
        ['temp', 'humidity']
        >>>
        >>> # Exog series without name
        >>> exog = pd.Series([10, 20, 30])
        >>> df_exog = input_to_frame(exog, input_name="exog")
        >>> df_exog.columns.tolist()
        ['exog']
    """
    output_col_name = {"y": "y", "last_window": "y", "exog": "exog"}

    if isinstance(data, pd.Series):
        data = data.to_frame(
            name=data.name if data.name is not None else output_col_name[input_name]
        )

    return data


def expand_index(index: Union[pd.Index, None], steps: int) -> pd.Index:
    """
    Create a new index extending from the end of the original index.

    This function generates future indices for forecasting by extending the time
    series index by a specified number of steps. Handles both DatetimeIndex and
    RangeIndex appropriately.

    Args:
        index: Original pandas Index (DatetimeIndex or RangeIndex). If None,
            creates a RangeIndex starting from 0.
        steps: Number of future steps to generate.

    Returns:
        New pandas Index with `steps` future periods.

    Raises:
        TypeError: If steps is not an integer, or if index is neither DatetimeIndex
            nor RangeIndex.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2.utils.data_transform import expand_index
        >>>
        >>> # DatetimeIndex
        >>> dates = pd.date_range("2023-01-01", periods=5, freq="D")
        >>> new_index = expand_index(dates, 3)
        >>> new_index
        DatetimeIndex(['2023-01-06', '2023-01-07', '2023-01-08'], dtype='datetime64[ns]', freq='D')
        >>>
        >>> # RangeIndex
        >>> range_idx = pd.RangeIndex(start=0, stop=10)
        >>> new_index = expand_index(range_idx, 5)
        >>> new_index
        RangeIndex(start=10, stop=15, step=1)
        >>>
        >>> # None index (creates new RangeIndex)
        >>> new_index = expand_index(None, 3)
        >>> new_index
        RangeIndex(start=0, stop=3, step=1)
        >>>
        >>> # Invalid: steps not an integer
        >>> try:
        ...     expand_index(dates, 3.5)
        ... except TypeError as e:
        ...     print("Error: steps must be an integer")
        Error: steps must be an integer
    """
    if not isinstance(steps, (int, np.integer)):
        raise TypeError(f"`steps` must be an integer. Got {type(steps)}.")

    # Convert numpy integer to Python int if needed
    if isinstance(steps, np.integer):
        steps = int(steps)

    if isinstance(index, pd.Index):
        if isinstance(index, pd.DatetimeIndex):
            new_index = pd.date_range(
                start=index[-1] + index.freq, periods=steps, freq=index.freq
            )
        elif isinstance(index, pd.RangeIndex):
            new_index = pd.RangeIndex(start=index[-1] + 1, stop=index[-1] + 1 + steps)
        else:
            raise TypeError(
                "Argument `index` must be a pandas DatetimeIndex or RangeIndex."
            )
    else:
        new_index = pd.RangeIndex(start=0, stop=steps)

    return new_index


def transform_dataframe(
    df: pd.DataFrame,
    transformer: object,
    fit: bool = False,
    inverse_transform: bool = False,
) -> pd.DataFrame:
    """
    Transform raw values of pandas DataFrame with a scikit-learn alike
    transformer, preprocessor or ColumnTransformer.

    The transformer used must have the following methods: fit, transform,
    fit_transform and inverse_transform. ColumnTransformers are not allowed
    since they do not have inverse_transform method.

    Args:
        df: DataFrame to be transformed.
        transformer: Scikit-learn alike transformer, preprocessor, or ColumnTransformer.
            Must implement fit, transform, fit_transform and inverse_transform.
        fit: Train the transformer before applying it. Defaults to False.
        inverse_transform: Transform back the data to the original representation.
            This is not available when using transformers of class
            scikit-learn ColumnTransformers. Defaults to False.

    Returns:
        Transformed DataFrame.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If inverse_transform is requested for ColumnTransformer.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` argument must be a pandas DataFrame. Got {type(df)}")

    if transformer is None:
        return df

    # Check for ColumnTransformer by class name to avoid importing sklearn
    is_column_transformer = type(
        transformer
    ).__name__ == "ColumnTransformer" or hasattr(transformer, "transformers")

    if inverse_transform and is_column_transformer:
        raise ValueError(
            "`inverse_transform` is not available when using ColumnTransformers."
        )

    if not inverse_transform:
        if fit:
            values_transformed = transformer.fit_transform(df)
        else:
            values_transformed = transformer.transform(df)
    else:
        values_transformed = transformer.inverse_transform(df)

    if hasattr(values_transformed, "toarray"):
        # If the returned values are in sparse matrix format, it is converted to dense
        values_transformed = values_transformed.toarray()

    if isinstance(values_transformed, pd.DataFrame):
        df_transformed = values_transformed
    else:
        df_transformed = pd.DataFrame(
            values_transformed, index=df.index, columns=df.columns
        )

    return df_transformed

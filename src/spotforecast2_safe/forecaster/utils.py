from typing import Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import warnings
import uuid
from importlib.util import find_spec
from sklearn.compose import ColumnTransformer
from spotforecast2_safe.utils import (
    initialize_lags,
    initialize_weights,
    check_select_fit_kwargs,
    check_y,
    check_exog,
    get_exog_dtypes,
    check_exog_dtypes,
    check_predict_input,
    check_interval,
    input_to_frame,
    expand_index,
    transform_dataframe,
)
from spotforecast2_safe.exceptions import set_skforecast_warnings, UnknownLevelWarning

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is not installed
    tqdm = None


def check_preprocess_series(series):
    pass


def check_preprocess_exog_multiseries(exog):
    pass


def exog_to_direct(
    exog: pd.Series | pd.DataFrame, steps: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transforms `exog` to a pandas DataFrame with the shape needed for Direct
    forecasting.

    Args:
        exog: pandas Series, pandas DataFrame
            Exogenous variables.
        steps: int
            Number of steps that will be predicted using exog.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            exog_direct: pandas DataFrame
                Exogenous variables transformed.
            exog_direct_names: list
                Names of the columns of the exogenous variables transformed. Only
                created if `exog` is a pandas Series or DataFrame.
    """

    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"`exog` must be a pandas Series or DataFrame. Got {type(exog)}."
        )

    if isinstance(exog, pd.Series):
        exog = exog.to_frame()

    n_rows = len(exog)
    exog_idx = exog.index
    exog_cols = exog.columns
    exog_direct = []
    for i in range(steps):
        exog_step = exog.iloc[i : n_rows - (steps - 1 - i),]
        exog_step.index = pd.RangeIndex(len(exog_step))
        exog_step.columns = [f"{col}_step_{i + 1}" for col in exog_cols]
        exog_direct.append(exog_step)

    exog_direct = pd.concat(exog_direct, axis=1) if steps > 1 else exog_direct[0]

    exog_direct_names = exog_direct.columns.to_list()
    exog_direct.index = exog_idx[-len(exog_direct) :]

    return exog_direct, exog_direct_names


def exog_to_direct_numpy(
    exog: np.ndarray | pd.Series | pd.DataFrame, steps: int
) -> tuple[np.ndarray, list[str] | None]:
    """
    Transforms `exog` to numpy ndarray with the shape needed for Direct
    forecasting.

    Args:
        exog: numpy ndarray, pandas Series, pandas DataFrame
            Exogenous variables, shape(samples,). If exog is a pandas format, the
            direct exog names are created.
        steps: int
            Number of steps that will be predicted using exog.

    Returns:
        tuple[np.ndarray, list[str] | None]:
            exog_direct: numpy ndarray
                Exogenous variables transformed.
            exog_direct_names: list, None
                Names of the columns of the exogenous variables transformed. Only
                created if `exog` is a pandas Series or DataFrame.
    """

    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog_cols = exog.columns if isinstance(exog, pd.DataFrame) else [exog.name]
        exog_direct_names = [
            f"{col}_step_{i + 1}" for i in range(steps) for col in exog_cols
        ]
        exog = exog.to_numpy()
    else:
        exog_direct_names = None
        if not isinstance(exog, np.ndarray):
            raise TypeError(
                f"`exog` must be a numpy ndarray, pandas Series or DataFrame. "
                f"Got {type(exog)}."
            )

    if exog.ndim == 1:
        exog = np.expand_dims(exog, axis=1)

    n_rows = len(exog)
    exog_direct = [exog[i : n_rows - (steps - 1 - i)] for i in range(steps)]
    exog_direct = np.concatenate(exog_direct, axis=1) if steps > 1 else exog_direct[0]

    return exog_direct, exog_direct_names


def prepare_steps_direct(
    max_step: int | list[int] | np.ndarray, steps: int | list[int] | None = None
) -> list[int]:
    """
    Prepare list of steps to be predicted in Direct Forecasters.

    Args:
        max_step: int, list, numpy ndarray
            Maximum number of future steps the forecaster will predict
            when using predict methods.
        steps: int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the
            value of steps defined when initializing the forecaster. Starts at 1.

            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list
              are predicted.
            - If `None`: As many steps are predicted as were defined at
              initialization.

    Returns:
        list[int]:
            Steps to be predicted.
    """

    if isinstance(steps, int):
        steps_direct = list(range(1, steps + 1))
    elif steps is None:
        if isinstance(max_step, int):
            steps_direct = list(range(1, max_step + 1))
        else:
            steps_direct = [int(s) for s in max_step]
    elif isinstance(steps, list):
        steps_direct = []
        for step in steps:
            if not isinstance(step, (int, np.integer)):
                raise TypeError(
                    f"`steps` argument must be an int, a list of ints or `None`. "
                    f"Got {type(steps)}."
                )
            steps_direct.append(int(step))

    return steps_direct


def transform_numpy(
    array: np.ndarray,
    transformer: object | None,
    fit: bool = False,
    inverse_transform: bool = False,
) -> np.ndarray:
    """
    Transform raw values of a numpy ndarray with a scikit-learn alike
    transformer, preprocessor or ColumnTransformer. The transformer used must
    have the following methods: fit, transform, fit_transform and
    inverse_transform. ColumnTransformers are not allowed since they do not
    have inverse_transform method.

    Args:
        array: numpy ndarray
            Array to be transformed.
        transformer: scikit-learn alike transformer, preprocessor, or ColumnTransformer.
            Scikit-learn alike transformer (preprocessor) with methods: fit, transform,
            fit_transform and inverse_transform.
    fit: bool, default False
        Train the transformer before applying it.
    inverse_transform: bool, default False
        Transform back the data to the original representation. This is not available
        when using transformers of class scikit-learn ColumnTransformers.

    Returns
    -------
    array_transformed : numpy ndarray
        Transformed array.

    """

    if transformer is None:
        return array

    if not isinstance(array, np.ndarray):
        raise TypeError(f"`array` argument must be a numpy ndarray. Got {type(array)}")

    original_ndim = array.ndim
    original_shape = array.shape
    reshaped_for_inverse = False

    if original_ndim == 1:
        array = array.reshape(-1, 1)

    if inverse_transform and isinstance(transformer, ColumnTransformer):
        raise ValueError(
            "`inverse_transform` is not available when using ColumnTransformers."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        if not inverse_transform:
            if fit:
                array_transformed = transformer.fit_transform(array)
            else:
                array_transformed = transformer.transform(array)
        else:
            # Vectorized inverse transformation for 2D arrays with multiple columns.
            # Reshape to single column, transform, and reshape back.
            # This is faster than applying the transformer column by column.
            if array.shape[1] > 1:
                array = array.reshape(-1, 1)
                reshaped_for_inverse = True
            array_transformed = transformer.inverse_transform(array)

    if hasattr(array_transformed, "toarray"):
        # If the returned values are in sparse matrix format, it is converted to dense
        array_transformed = array_transformed.toarray()

    if isinstance(array_transformed, (pd.Series, pd.DataFrame)):
        array_transformed = array_transformed.to_numpy()

    # Reshape back to original shape only if we reshaped for inverse_transform
    if reshaped_for_inverse:
        array_transformed = array_transformed.reshape(original_shape)

    if original_ndim == 1:
        array_transformed = array_transformed.ravel()

    return array_transformed


def select_n_jobs_fit_forecaster(forecaster_name, estimator):
    """
    Select the number of jobs to run in parallel.
    """
    import os

    return os.cpu_count() or 1


__all__ = [
    "initialize_lags",
    "initialize_weights",
    "check_select_fit_kwargs",
    "check_y",
    "check_exog",
    "get_exog_dtypes",
    "check_exog_dtypes",
    "check_predict_input",
    "check_interval",
    "input_to_frame",
    "expand_index",
    "transform_dataframe",
    "check_preprocess_series",
    "check_preprocess_exog_multiseries",
    "set_skforecast_warnings",
    "initialize_window_features",
    "initialize_transformer_series",
    "check_extract_values_and_index",
    "get_style_repr_html",
    "initialize_estimator",
    "check_residuals_input",
    "date_to_index_position",
    "prepare_steps_direct",
    "exog_to_direct",
    "exog_to_direct_numpy",
    "transform_numpy",
    "select_n_jobs_fit_forecaster",
    "predict_multivariate",
]


def initialize_window_features(
    window_features: Any,
) -> Tuple[Optional[List[object]], Optional[List[str]], Optional[int]]:
    """Check window_features argument input and generate the corresponding list.

    This function validates window feature objects and extracts their metadata,
    ensuring they have the required attributes (window_sizes, features_names) and
    methods (transform_batch, transform) for proper forecasting operations.

    Args:
        window_features: Classes used to create window features. Can be a single
            object or a list of objects. Each object must have `window_sizes`,
            `features_names` attributes and `transform_batch`, `transform` methods.

    Returns:
        tuple: A tuple containing:
            - window_features (list or None): List of classes used to create window features.
            - window_features_names (list or None): List with all the features names of the window features.
            - max_size_window_features (int or None): Maximum value of the `window_sizes` attribute of all classes.

    Raises:
        ValueError: If `window_features` is an empty list.
        ValueError: If a window feature is missing required attributes or methods.
        TypeError: If `window_sizes` or `features_names` have incorrect types.

    Examples:
        >>> from spotforecast2_safe.forecaster.preprocessing import RollingFeatures
        >>> wf = RollingFeatures(stats=['mean', 'std'], window_sizes=[7, 14])
        >>> wf_list, names, max_size = initialize_window_features(wf)
        >>> print(f"Max window size: {max_size}")
        Max window size: 14
        >>> print(f"Number of features: {len(names)}")
        Number of features: 4

        Multiple window features:
        >>> wf1 = RollingFeatures(stats=['mean'], window_sizes=7)
        >>> wf2 = RollingFeatures(stats=['max', 'min'], window_sizes=3)
        >>> wf_list, names, max_size = initialize_window_features([wf1, wf2])
        >>> print(f"Max window size: {max_size}")
        Max window size: 7
    """

    needed_atts = ["window_sizes", "features_names"]
    needed_methods = ["transform_batch", "transform"]

    max_window_sizes = None
    window_features_names = None
    max_size_window_features = None
    if window_features is not None:
        if isinstance(window_features, list) and len(window_features) < 1:
            raise ValueError(
                "Argument `window_features` must contain at least one element."
            )
        if not isinstance(window_features, list):
            window_features = [window_features]

        link_to_docs = (
            "\nVisit the documentation for more information about how to create "
            "custom window features:\n"
            "https://skforecast.org/latest/user_guides/window-features-and-custom-features.html#create-your-custom-window-features"
        )

        max_window_sizes = []
        window_features_names = []
        needed_atts_set = set(needed_atts)
        needed_methods_set = set(needed_methods)
        for wf in window_features:
            wf_name = type(wf).__name__
            atts_methods = set(dir(wf))
            if not needed_atts_set.issubset(atts_methods):
                raise ValueError(
                    f"{wf_name} must have the attributes: {needed_atts}." + link_to_docs
                )
            if not needed_methods_set.issubset(atts_methods):
                raise ValueError(
                    f"{wf_name} must have the methods: {needed_methods}." + link_to_docs
                )

            window_sizes = wf.window_sizes
            if not isinstance(window_sizes, (int, list)):
                raise TypeError(
                    f"Attribute `window_sizes` of {wf_name} must be an int or a list "
                    f"of ints. Got {type(window_sizes)}." + link_to_docs
                )

            if isinstance(window_sizes, int):
                if window_sizes < 1:
                    raise ValueError(
                        f"If argument `window_sizes` is an integer, it must be equal to or "
                        f"greater than 1. Got {window_sizes} from {wf_name}."
                        + link_to_docs
                    )
                max_window_sizes.append(window_sizes)
            else:
                if not all(isinstance(ws, int) for ws in window_sizes) or not all(
                    ws >= 1 for ws in window_sizes
                ):
                    raise ValueError(
                        f"If argument `window_sizes` is a list, all elements must be integers "
                        f"equal to or greater than 1. Got {window_sizes} from {wf_name}."
                        + link_to_docs
                    )
                max_window_sizes.append(max(window_sizes))

            features_names = wf.features_names
            if not isinstance(features_names, (str, list)):
                raise TypeError(
                    f"Attribute `features_names` of {wf_name} must be a str or "
                    f"a list of strings. Got {type(features_names)}." + link_to_docs
                )
            if isinstance(features_names, str):
                window_features_names.append(features_names)
            else:
                if not all(isinstance(fn, str) for fn in features_names):
                    raise TypeError(
                        f"If argument `features_names` is a list, all elements "
                        f"must be strings. Got {features_names} from {wf_name}."
                        + link_to_docs
                    )
                window_features_names.extend(features_names)

        max_size_window_features = max(max_window_sizes)
        if len(set(window_features_names)) != len(window_features_names):
            raise ValueError(
                f"All window features names must be unique. Got {window_features_names}."
            )

    return window_features, window_features_names, max_size_window_features


def check_extract_values_and_index(
    data: Union[pd.Series, pd.DataFrame],
    data_label: str = "`y`",
    ignore_freq: bool = False,
    return_values: bool = True,
) -> Tuple[Optional[np.ndarray], pd.Index]:
    """Extract values and index from a pandas Series or DataFrame, ensuring they are valid.

    Validates that the input data has a proper DatetimeIndex or RangeIndex and extracts
    its values and index for use in forecasting operations. Optionally checks for
    index frequency consistency.

    Args:
        data: Input data (pandas Series or DataFrame) to extract values and index from.
        data_label: Label used in exception messages for better error reporting.
            Defaults to "`y`".
        ignore_freq: If True, the frequency of the index is not checked.
            Defaults to False.
        return_values: If True, the values of the data are returned.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - values (numpy.ndarray or None): Values of the data as numpy array,
              or None if return_values is False.
            - index (pandas.Index): Index of the data.

    Raises:
        TypeError: If data is not a pandas Series or DataFrame.
        TypeError: If data index is not a DatetimeIndex or RangeIndex.

    Warnings:
        UserWarning: If DatetimeIndex has no frequency (inferred automatically).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2020-01-01', periods=10, freq='D')
        >>> series = pd.Series(np.arange(10), index=dates)
        >>> values, index = check_extract_values_and_index(series)
        >>> print(values.shape)
        (10,)
        >>> print(type(index))
        <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

        Extract index only:
        >>> _, index = check_extract_values_and_index(series, return_values=False)
        >>> print(index[0])
        2020-01-01 00:00:00
    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{data_label} must be a pandas Series or DataFrame.")

    if not isinstance(data.index, (pd.DatetimeIndex, pd.RangeIndex)):
        raise TypeError(f"{data_label} must have a pandas DatetimeIndex or RangeIndex.")

    if isinstance(data.index, pd.DatetimeIndex) and not ignore_freq:
        if data.index.freq is None:
            warnings.warn(
                f"{data_label} has a DatetimeIndex but no frequency. "
                "The frequency has been inferred from the index.",
                UserWarning,
            )

    values = data.to_numpy() if return_values else None

    return values, data.index


def get_style_repr_html(is_fitted: bool = False) -> Tuple[str, str]:
    """Generate CSS style for HTML representation of the Forecaster.

    Creates a unique CSS style block with a container ID for rendering
    forecaster objects in Jupyter notebooks or HTML documents. The styling
    provides a clean, monospace display with a light gray background.

    Args:
        is_fitted: Parameter to indicate if the Forecaster has been fitted.
            Currently not used in styling but reserved for future extensions.

    Returns:
        tuple: A tuple containing:
            - style (str): CSS style block as a string with unique container class.
            - unique_id (str): Unique 8-character ID for the container element.

    Examples:
        >>> style, uid = get_style_repr_html(is_fitted=True)
        >>> print(f"Container ID: {uid}")
        Container ID: a1b2c3d4
        >>> print(f"Style contains CSS: {'container-' in style}")
        Style contains CSS: True

        Using in HTML rendering:
        >>> style, uid = get_style_repr_html(is_fitted=False)
        >>> html = f"{style}<div class='container-{uid}'>Forecaster Info</div>"
        >>> print("background-color" in html)
        True
    """

    unique_id = str(uuid.uuid4())[:8]
    style = f"""
    <style>
        .container-{unique_id} {{
            font-family: monospace;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
        }}
    </style>
    """
    return style, unique_id


def check_residuals_input(
    forecaster_name: str,
    use_in_sample_residuals: bool,
    in_sample_residuals_: np.ndarray | dict[str, np.ndarray] | None,
    out_sample_residuals_: np.ndarray | dict[str, np.ndarray] | None,
    use_binned_residuals: bool,
    in_sample_residuals_by_bin_: (
        dict[str | int, np.ndarray | dict[int, np.ndarray]] | None
    ),
    out_sample_residuals_by_bin_: (
        dict[str | int, np.ndarray | dict[int, np.ndarray]] | None
    ),
    levels: list[str] | None = None,
    encoding: str | None = None,
) -> None:
    """
    Check residuals input arguments in Forecasters.

    Args:
        forecaster_name: str
            Forecaster name.
        use_in_sample_residuals: bool
            Indicates if in sample or out sample residuals are used.
        in_sample_residuals_: numpy ndarray, dict
            Residuals of the model when predicting training data.
        out_sample_residuals_: numpy ndarray, dict
            Residuals of the model when predicting non training data.
        use_binned_residuals: bool
            Indicates if residuals are binned.
        in_sample_residuals_by_bin_: dict
            In sample residuals binned according to the predicted value each residual
            is associated with.
        out_sample_residuals_by_bin_: dict
            Out of sample residuals binned according to the predicted value each residual
            is associated with.
        levels: list, default None
            Names of the series (levels) to be predicted (Forecasters multiseries).
        encoding: str, default None
            Encoding used to identify the different series (ForecasterRecursiveMultiSeries).

    Returns:
        None

    """

    forecasters_multiseries = (
        "ForecasterRecursiveMultiSeries",
        "ForecasterDirectMultiVariate",
        "ForecasterRnn",
    )

    if use_in_sample_residuals:
        if use_binned_residuals:
            residuals = in_sample_residuals_by_bin_
            literal = "in_sample_residuals_by_bin_"
        else:
            residuals = in_sample_residuals_
            literal = "in_sample_residuals_"

        # Check if residuals are empty or None
        is_empty = (
            residuals is None
            or (isinstance(residuals, dict) and not residuals)
            or (isinstance(residuals, np.ndarray) and residuals.size == 0)
        )
        if is_empty:
            raise ValueError(
                f"`forecaster.{literal}` is either None or empty. Use "
                f"`store_in_sample_residuals = True` when fitting the forecaster "
                f"or use the `set_in_sample_residuals()` method before predicting."
            )

        if forecaster_name in forecasters_multiseries:
            if encoding is not None:
                unknown_levels = set(levels) - set(residuals.keys())
                if unknown_levels:
                    warnings.warn(
                        f"`levels` {unknown_levels} are not present in `forecaster.{literal}`, "
                        f"most likely because they were not present in the training data. "
                        f"A random sample of the residuals from other levels will be used. "
                        f"This can lead to inaccurate intervals for the unknown levels.",
                        UnknownLevelWarning,
                    )
    else:
        if use_binned_residuals:
            residuals = out_sample_residuals_by_bin_
            literal = "out_sample_residuals_by_bin_"
        else:
            residuals = out_sample_residuals_
            literal = "out_sample_residuals_"

        is_empty = (
            residuals is None
            or (isinstance(residuals, dict) and not residuals)
            or (isinstance(residuals, np.ndarray) and residuals.size == 0)
        )
        if is_empty:
            raise ValueError(
                f"`forecaster.{literal}` is either None or empty. Use "
                f"`set_out_sample_residuals()` method before predicting."
            )

        if forecaster_name in forecasters_multiseries:
            if encoding is not None:
                unknown_levels = set(levels) - set(residuals.keys())
                if unknown_levels:
                    warnings.warn(
                        f"`levels` {unknown_levels} are not present in `forecaster.{literal}`, "
                        f"most likely because they were not present in the training data. "
                        f"A random sample of the residuals from other levels will be used. "
                        f"This can lead to inaccurate intervals for the unknown levels.",
                        UnknownLevelWarning,
                    )


def date_to_index_position(
    index: pd.Index,
    date_input: int | str | pd.Timestamp,
    method: str = "prediction",
    date_literal: str = "steps",
    kwargs_pd_to_datetime: dict = {},
) -> int:
    """
    Transform a datetime string or pandas Timestamp to an integer. The integer
    represents the position of the datetime in the index.

    Args:
        index: pandas Index
            Original datetime index (must be a pandas DatetimeIndex if `date_input`
            is not an int).
        date_input: int, str, pandas Timestamp
            Datetime to transform to integer.

            - If int, returns the same integer.
            - If str or pandas Timestamp, it is converted and expanded into the index.
        method: str, default 'prediction'
            Can be 'prediction' or 'validation'.

            - If 'prediction', the date must be later than the last date in the index.
            - If 'validation', the date must be within the index range.
        date_literal: str, default 'steps'
            Variable name used in error messages.
        kwargs_pd_to_datetime: dict, default {}
            Additional keyword arguments to pass to `pd.to_datetime()`.

    Returns:
        int:
            `date_input` transformed to integer position in the `index`.

        + If `date_input` is an integer, it returns the same integer.
        + If method is 'prediction', number of steps to predict from the last
        date in the index.
        + If method is 'validation', position plus one of the date in the index,
        this is done to include the target date in the training set when using
        pandas iloc with slices.

    """

    if method not in ["prediction", "validation"]:
        raise ValueError("`method` must be 'prediction' or 'validation'.")

    if isinstance(date_input, (str, pd.Timestamp)):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(
                f"Index must be a pandas DatetimeIndex when `{date_literal}` is "
                f"not an integer. Check input series or last window."
            )

        target_date = pd.to_datetime(date_input, **kwargs_pd_to_datetime)
        last_date = pd.to_datetime(index[-1])

        if method == "prediction":
            if target_date <= last_date:
                raise ValueError(
                    "If `steps` is a date, it must be greater than the last date "
                    "in the index."
                )
            span_index = pd.date_range(
                start=last_date, end=target_date, freq=index.freq
            )
            output = len(span_index) - 1
        elif method == "validation":
            first_date = pd.to_datetime(index[0])
            if target_date < first_date or target_date > last_date:
                raise ValueError(
                    "If `initial_train_size` is a date, it must be greater than "
                    "the first date in the index and less than the last date."
                )
            span_index = pd.date_range(
                start=first_date, end=target_date, freq=index.freq
            )
            output = len(span_index)

    elif isinstance(date_input, (int, np.integer)):
        output = date_input

    else:
        raise TypeError(
            f"`{date_literal}` must be an integer, string, or pandas Timestamp."
        )

    return output


def initialize_estimator(
    estimator: object | None = None, regressor: object | None = None
) -> None:
    """
    Helper to handle the deprecation of 'regressor' in favor of 'estimator'.
    Returns the valid estimator object.

    Args:
        estimator: estimator or pipeline compatible with the scikit-learn API, default None
            An instance of a estimator or pipeline compatible with the scikit-learn API.
        regressor: estimator or pipeline compatible with the scikit-learn API, default None
            Deprecated. An instance of a estimator or pipeline compatible with the
            scikit-learn API.

    Returns:
        estimator or pipeline compatible with the scikit-learn API
            The valid estimator object.

    """

    if regressor is not None:
        warnings.warn(
            "The `regressor` argument is deprecated and will be removed in a future "
            "version. Please use `estimator` instead.",
            FutureWarning,
            stacklevel=3,  # Important: to point to the user's code
        )
        if estimator is not None:
            raise ValueError(
                "Both `estimator` and `regressor` were provided. Use only `estimator`."
            )
        return regressor

    return estimator


def predict_multivariate(
    forecasters: dict[str, Any],
    steps_ahead: int,
    exog: pd.DataFrame | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Generate multi-output predictions using multiple baseline forecasters.

    Args:
        forecasters (dict): Dictionary of fitted forecaster instances (one per target).
            Keys are target names, values are the fitted forecasters (e.g.,
            ForecasterRecursive, ForecasterEquivalentDate).
        steps_ahead (int): Number of steps to forecast.
        exog (pd.DataFrame, optional): Exogenous variables for prediction.
            If provided, will be passed to each forecaster's predict method.
        show_progress (bool, optional): Show progress bar while predicting
            per target forecaster. Default: False.

    Returns:
        pd.DataFrame: DataFrame with predictions for all targets.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from spotforecast2_safe.forecaster.utils import predict_multivariate
        >>> y1 = pd.Series([1, 2, 3, 4, 5])
        >>> y2 = pd.Series([2, 4, 6, 8, 10])
        >>> f1 = ForecasterRecursive(estimator=LinearRegression(), lags=2)
        >>> f2 = ForecasterRecursive(estimator=LinearRegression(), lags=2)
        >>> f1.fit(y=y1)
        >>> f2.fit(y=y2)
        >>> forecasters = {'target1': f1, 'target2': f2}
        >>> predictions = predict_multivariate(forecasters, steps_ahead=2)
        >>> predictions
           target1  target2
        5      6.0     12.0
        6      7.0     14.0
    """

    if not forecasters:
        return pd.DataFrame()

    predictions = {}

    target_iter = forecasters.items()
    if show_progress and tqdm is not None:
        target_iter = tqdm(
            forecasters.items(),
            desc="Predicting targets",
            unit="model",
        )

    for target, forecaster in target_iter:
        # Generate predictions for this target
        if exog is not None:
            pred = forecaster.predict(steps=steps_ahead, exog=exog)
        else:
            pred = forecaster.predict(steps=steps_ahead)
        predictions[target] = pred

    # Combine into a single DataFrame
    return pd.concat(predictions, axis=1)


def initialize_transformer_series(
    forecaster_name: str,
    series_names_in_: list[str],
    encoding: str | None = None,
    transformer_series: object | dict[str, object | None] | None = None,
) -> dict[str, object | None]:
    """Initialize transformer_series_ attribute for multivariate/multiseries forecasters.

    Creates a dictionary of transformers for each time series in multivariate or
    multiseries forecasting. Handles three cases: no transformation (None), same
    transformer for all series (single object), or different transformers per series
    (dictionary). Clones transformer objects to avoid overwriting.

    Args:
        forecaster_name: Name of the forecaster using this function. Special handling
            is applied for 'ForecasterRecursiveMultiSeries'.
        series_names_in_: Names of the time series (levels) used during training.
            These will be the keys in the returned transformer dictionary.
        encoding: Encoding used to identify different series. Only used for
            ForecasterRecursiveMultiSeries. If None, creates a single '_unknown_level'
            entry. Defaults to None.
        transformer_series: Transformer(s) to apply to series. Can be:
            - None: No transformation applied
            - Single transformer object: Same transformer cloned for all series
            - Dict mapping series names to transformers: Different transformer per series
            Defaults to None.

    Returns:
        dict: Dictionary with series names as keys and transformer objects (or None)
            as values. Transformers are cloned to prevent overwriting.

    Warnings:
        IgnoredArgumentWarning: If transformer_series is a dict and some series_names_in_
            are not present in the dict keys (those series get no transformation).

    Examples:
        No transformation:
        >>> from spotforecast2_safe.forecaster.utils import initialize_transformer_series
        >>> series = ['series1', 'series2', 'series3']
        >>> result = initialize_transformer_series(
        ...     forecaster_name='ForecasterDirectMultiVariate',
        ...     series_names_in_=series,
        ...     transformer_series=None
        ... )
        >>> print(result)
        {'series1': None, 'series2': None, 'series3': None}

        Same transformer for all series:
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> result = initialize_transformer_series(
        ...     forecaster_name='ForecasterDirectMultiVariate',
        ...     series_names_in_=['series1', 'series2'],
        ...     transformer_series=scaler
        ... )
        >>> len(result)
        2
        >>> all(isinstance(v, StandardScaler) for v in result.values())
        True
        >>> result['series1'] is result['series2']  # Different clones
        False

        Different transformer per series:
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> transformers = {
        ...     'series1': StandardScaler(),
        ...     'series2': MinMaxScaler()
        ... }
        >>> result = initialize_transformer_series(
        ...     forecaster_name='ForecasterDirectMultiVariate',
        ...     series_names_in_=['series1', 'series2'],
        ...     transformer_series=transformers
        ... )
        >>> isinstance(result['series1'], StandardScaler)
        True
        >>> isinstance(result['series2'], MinMaxScaler)
        True
    """
    from copy import deepcopy
    from sklearn.base import clone
    from spotforecast2_safe.exceptions import IgnoredArgumentWarning

    if forecaster_name == "ForecasterRecursiveMultiSeries":
        if encoding is None:
            series_names_in_ = ["_unknown_level"]
        else:
            series_names_in_ = series_names_in_ + ["_unknown_level"]

    if transformer_series is None:
        transformer_series_ = {serie: None for serie in series_names_in_}
    elif not isinstance(transformer_series, dict):
        transformer_series_ = {
            serie: clone(transformer_series) for serie in series_names_in_
        }
    else:
        transformer_series_ = {serie: None for serie in series_names_in_}
        # Only elements already present in transformer_series_ are updated
        transformer_series_.update(
            {
                k: deepcopy(v)
                for k, v in transformer_series.items()
                if k in transformer_series_
            }
        )

        series_not_in_transformer_series = (
            set(series_names_in_) - set(transformer_series.keys())
        ) - {"_unknown_level"}
        if series_not_in_transformer_series:
            warnings.warn(
                f"{series_not_in_transformer_series} not present in `transformer_series`."
                f" No transformation is applied to these series.",
                IgnoredArgumentWarning,
            )

    return transformer_series_

from typing import Optional, Dict, Any

from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def mark_outliers(
    data: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 1234,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Marks outliers as NaN in the dataset using Isolation Forest.

    Args:
        data (pd.DataFrame):
            The input dataset.
        contamination (float):
            The (estimated) proportion of outliers in the dataset.
        random_state (int):
            Random seed for reproducibility. Default is 1234.
        verbose (bool):
            Whether to print additional information.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the modified dataset with outliers marked as NaN and the outlier labels.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_data
        >>> from spotforecast2.preprocessing.outlier import mark_outliers
        >>> data = fetch_data()
        >>> cleaned_data, outlier_labels = mark_outliers(data, contamination=0.1, random_state=42, verbose=True)
    """
    for col in data.columns:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        # Fit and predict (-1 for outliers, 1 for inliers)
        outliers = iso.fit_predict(data[[col]])

        # Mark outliers as NaN
        data.loc[outliers == -1, col] = np.nan

        pct_outliers = (outliers == -1).mean() * 100
        if verbose:
            print(
                f"Column '{col}': Marked {pct_outliers:.4f}% of data points as outliers."
            )
    return data, outliers


def manual_outlier_removal(
    data: pd.DataFrame,
    column: str,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Manual outlier removal function.
    Args:
        data (pd.DataFrame):
            The input dataset.
        column (str):
            The column name in which to perform manual outlier removal.
        lower_threshold (float | None):
            The lower threshold below which values are considered outliers.
            If None, no lower threshold is applied.
        upper_threshold (float | None):
            The upper threshold above which values are considered outliers.
            If None, no upper threshold is applied.
        verbose (bool):
            Whether to print additional information.

    Returns:
        tuple[pd.DataFrame, int]: A tuple containing the modified dataset with outliers marked as NaN and the number of outliers marked.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_data
        >>> from spotforecast2.preprocessing.outlier import manual_outlier_removal
        >>> data = fetch_data()
        >>> data, n_manual_outliers = manual_outlier_removal(
        ...     data,
        ...     column='ABC',
        ...     lower_threshold=50,
        ...     upper_threshold=700,
        ...     verbose=True
    """
    if lower_threshold is None and upper_threshold is None:
        if verbose:
            print(f"No thresholds provided for {column}; no outliers marked.")
        return data, 0

    if lower_threshold is not None and upper_threshold is not None:
        mask = (data[column] > upper_threshold) | (data[column] < lower_threshold)
    elif lower_threshold is not None:
        mask = data[column] < lower_threshold
    else:
        mask = data[column] > upper_threshold

    n_manual_outliers = mask.sum()

    data.loc[mask, column] = np.nan

    if verbose:
        if lower_threshold is not None and upper_threshold is not None:
            print(
                f"Manually marked {n_manual_outliers} values > {upper_threshold} or < {lower_threshold} as outliers in {column}."
            )
        elif lower_threshold is not None:
            print(
                f"Manually marked {n_manual_outliers} values < {lower_threshold} as outliers in {column}."
            )
        else:
            print(
                f"Manually marked {n_manual_outliers} values > {upper_threshold} as outliers in {column}."
            )
    return data, n_manual_outliers


def get_outliers(
    data: pd.DataFrame,
    data_original: Optional[pd.DataFrame] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
) -> Dict[str, pd.Series]:
    """Detect outliers in each column using Isolation Forest.

    This function uses scikit-learn's IsolationForest algorithm to detect outliers
    in each column of the input DataFrame. The original data (before any NaN values
    were introduced) can be provided to identify which values were marked as NaN due
    to outlier detection.

    Args:
        data: The input DataFrame to check for outliers.
        data_original: Optional original DataFrame before outlier marking. If provided,
            helps identify which values became NaN due to outlier detection.
            Default: None.
        contamination: The estimated proportion of outliers in the dataset.
            Default: 0.01.
        random_state: Random seed for reproducibility. Default: 1234.

    Returns:
        A dictionary mapping column names to Series of outlier values.
        For columns without outliers, an empty Series is returned.

    Raises:
        ValueError: If data is empty or contains no columns.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2.preprocessing.outlier import get_outliers
        >>>
        >>> # Create sample data with outliers
        >>> np.random.seed(42)
        >>> data = pd.DataFrame({
        ...     'A': np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]]),
        ...     'B': np.concatenate([np.random.normal(5, 2, 100), [100, 110, 120]])
        ... })
        >>> data_original = data.copy()
        >>>
        >>> # Detect outliers
        >>> outliers = get_outliers(data_original, contamination=0.03)
        >>> for col, outlier_vals in outliers.items():
        ...     print(f"{col}: {len(outlier_vals)} outliers detected")
    """
    if data.empty:
        raise ValueError("Input data is empty")
    if len(data.columns) == 0:
        raise ValueError("Input data contains no columns")

    outliers_dict = {}

    for col in data.columns:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        # Fit and predict (-1 for outliers, 1 for inliers)
        predictions = iso.fit_predict(data[[col]])

        # Get outlier values
        if data_original is not None:
            # Use original data to identify outlier values
            outlier_mask = predictions == -1
            outliers_dict[col] = data_original.loc[outlier_mask, col]
        else:
            # Use current data
            outlier_mask = predictions == -1
            outliers_dict[col] = data.loc[outlier_mask, col]

    return outliers_dict


def visualize_outliers_hist(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    columns: Optional[list[str]] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
    figsize: tuple[int, int] = (10, 5),
    bins: int = 50,
    **kwargs: Any,
) -> None:
    """Visualize outliers in DataFrame using stacked histograms.

    Creates a histogram for each specified column, displaying both regular data
    and detected outliers in different colors. Uses IsolationForest for outlier
    detection.

    Args:
        data: The DataFrame with cleaned data (outliers may be NaN).
        data_original: The original DataFrame before outlier detection.
        columns: List of column names to visualize. If None, all columns are used.
            Default: None.
        contamination: The estimated proportion of outliers in the dataset.
            Default: 0.01.
        random_state: Random seed for reproducibility. Default: 1234.
        figsize: Figure size as (width, height). Default: (10, 5).
        bins: Number of histogram bins. Default: 50.
        **kwargs: Additional keyword arguments passed to plt.hist() (e.g., color,
            alpha, edgecolor, etc.).

    Returns:
        None. Displays matplotlib figures.

    Raises:
        ValueError: If data or data_original is empty, or if specified columns
            don't exist.
        ImportError: If matplotlib is not installed.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2.preprocessing.outlier import visualize_outliers_hist
        >>>
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> data_original = pd.DataFrame({
        ...     'temperature': np.concatenate([
        ...         np.random.normal(20, 5, 100),
        ...         [50, 60, 70]  # outliers
        ...     ]),
        ...     'humidity': np.concatenate([
        ...         np.random.normal(60, 10, 100),
        ...         [95, 98, 99]  # outliers
        ...     ])
        ... })
        >>> data_cleaned = data_original.copy()
        >>>
        >>> # Visualize outliers
        >>> visualize_outliers_hist(
        ...     data_cleaned,
        ...     data_original,
        ...     contamination=0.03,
        ...     figsize=(12, 5),
        ...     alpha=0.7
        ... )
    """
    if data.empty or data_original.empty:
        raise ValueError("Input data is empty")

    columns_to_plot = columns if columns is not None else data.columns

    # Validate columns exist
    missing_cols = set(columns_to_plot) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")

    # Detect outliers
    outliers = get_outliers(
        data_original,
        data_original=data_original,
        contamination=contamination,
        random_state=random_state,
    )

    for col in columns_to_plot:
        # Get inliers (non-NaN values in cleaned data)
        inliers = data[col].dropna()

        # Get outlier values
        outlier_vals = outliers[col]

        # Calculate percentage
        pct_outliers = (len(outlier_vals) / len(data_original)) * 100

        # Create figure
        plt.figure(figsize=figsize)
        plt.hist(
            [inliers, outlier_vals],
            bins=bins,
            stacked=True,
            color=["lightgrey", "red"],
            label=["Regular Data", "Outliers"],
            **kwargs,
        )
        plt.grid(True, alpha=0.3)
        plt.title(f"{col} Distribution with Outliers ({pct_outliers:.2f}%)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()


def visualize_outliers_plotly_scatter(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    columns: Optional[list[str]] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
    **kwargs: Any,
) -> None:
    """Visualize outliers in time series using Plotly scatter plots.

    Creates an interactive time series plot for each specified column, showing
    regular data as a line and detected outliers as scatter points. Uses
    IsolationForest for outlier detection.

    Args:
        data: The DataFrame with cleaned data (outliers may be NaN).
        data_original: The original DataFrame before outlier detection.
        columns: List of column names to visualize. If None, all columns are used.
            Default: None.
        contamination: The estimated proportion of outliers in the dataset.
            Default: 0.01.
        random_state: Random seed for reproducibility. Default: 1234.
        **kwargs: Additional keyword arguments passed to go.Figure.update_layout()
            (e.g., template, height, etc.).

    Returns:
        None. Displays Plotly figures.

    Raises:
        ValueError: If data or data_original is empty, or if specified columns
            don't exist.
        ImportError: If plotly is not installed.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2.preprocessing.outlier import visualize_outliers_plotly_scatter
        >>>
        >>> # Create sample time series data
        >>> np.random.seed(42)
        >>> dates = pd.date_range('2024-01-01', periods=103, freq='h')
        >>> data_original = pd.DataFrame({
        ...     'temperature': np.concatenate([
        ...         np.random.normal(20, 5, 100),
        ...         [50, 60, 70]  # outliers
        ...     ]),
        ...     'humidity': np.concatenate([
        ...         np.random.normal(60, 10, 100),
        ...         [95, 98, 99]  # outliers
        ...     ])
        ... }, index=dates)
        >>> data_cleaned = data_original.copy()
        >>>
        >>> # Visualize outliers
        >>> visualize_outliers_plotly_scatter(
        ...     data_cleaned,
        ...     data_original,
        ...     contamination=0.03,
        ...     template='plotly_white'
        ... )
    """
    if go is None:
        raise ImportError(
            "plotly is required for this function. " "Install with: pip install plotly"
        )

    if data.empty or data_original.empty:
        raise ValueError("Input data is empty")

    columns_to_plot = columns if columns is not None else data.columns

    # Validate columns exist
    missing_cols = set(columns_to_plot) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")

    # Detect outliers
    outliers = get_outliers(
        data_original,
        data_original=data_original,
        contamination=contamination,
        random_state=random_state,
    )

    for col in columns_to_plot:
        fig = go.Figure()

        # Add regular data as line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                mode="lines",
                name="Regular Data",
                line=dict(color="lightgrey"),
            )
        )

        # Add outliers as scatter points
        outlier_vals = outliers[col]
        if not outlier_vals.empty:
            fig.add_trace(
                go.Scatter(
                    x=outlier_vals.index,
                    y=outlier_vals,
                    mode="markers",
                    name="Outliers",
                    marker=dict(color="red", size=8, symbol="x"),
                )
            )

        # Calculate percentage
        pct_outliers = (len(outlier_vals) / len(data_original)) * 100

        # Update layout with custom kwargs
        layout_kwargs = {
            "title": f"{col} Time Series with Outliers ({pct_outliers:.2f}%)",
            "xaxis_title": "Time",
            "yaxis_title": "Value",
            "template": "plotly_white",
            "legend": dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        }
        layout_kwargs.update(kwargs)
        fig.update_layout(**layout_kwargs)
        fig.show()

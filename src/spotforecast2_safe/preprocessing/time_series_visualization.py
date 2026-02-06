"""Time series visualization."""

from typing import Dict, List, Optional, Any, Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def visualize_ts_plotly(
    dataframes: Dict[str, pd.DataFrame],
    columns: Optional[List[str]] = None,
    title_suffix: str = "",
    figsize: tuple[int, int] = (1000, 500),
    template: str = "plotly_white",
    colors: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """Visualize multiple time series datasets interactively with Plotly.

    Creates interactive Plotly scatter plots for specified columns across multiple
    datasets (e.g., train, validation, test splits). Each dataset is displayed as
    a separate line with a unique color and name in the legend.

    Args:
        dataframes: Dictionary mapping dataset names to pandas DataFrames with datetime
            index. Example: {'Train': df_train, 'Validation': df_val, 'Test': df_test}
        columns: List of column names to visualize. If None, all columns are used.
            Default: None.
        title_suffix: Suffix to append to the column name in the title. Useful for
            adding units or descriptions. Default: "".
        figsize: Figure size as (width, height) in pixels. Default: (1000, 500).
        template: Plotly template name for styling. Options include 'plotly_white',
            'plotly_dark', 'plotly', 'ggplot2', etc. Default: 'plotly_white'.
        colors: Dictionary mapping dataset names to colors. If None, uses Plotly
            default colors. Example: {'Train': 'blue', 'Validation': 'orange'}.
            Default: None.
        **kwargs: Additional keyword arguments passed to go.Scatter() (e.g.,
            mode='lines+markers', line=dict(dash='dash')).

    Returns:
        None. Displays Plotly figures.

    Raises:
        ValueError: If dataframes dict is empty, contains no columns, or if
            specified columns don't exist in all dataframes.
        ImportError: If plotly is not installed.
        TypeError: If dataframes parameter is not a dictionary.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2.preprocessing.time_series_visualization import visualize_ts_plotly
        >>>
        >>> # Create sample time series data
        >>> np.random.seed(42)
        >>> dates_train = pd.date_range('2024-01-01', periods=100, freq='h')
        >>> dates_val = pd.date_range('2024-05-11', periods=50, freq='h')
        >>> dates_test = pd.date_range('2024-07-01', periods=30, freq='h')
        >>>
        >>> data_train = pd.DataFrame({
        ...     'temperature': np.random.normal(20, 5, 100),
        ...     'humidity': np.random.normal(60, 10, 100)
        ... }, index=dates_train)
        >>>
        >>> data_val = pd.DataFrame({
        ...     'temperature': np.random.normal(22, 5, 50),
        ...     'humidity': np.random.normal(55, 10, 50)
        ... }, index=dates_val)
        >>>
        >>> data_test = pd.DataFrame({
        ...     'temperature': np.random.normal(25, 5, 30),
        ...     'humidity': np.random.normal(50, 10, 30)
        ... }, index=dates_test)
        >>>
        >>> # Visualize all datasets
        >>> dataframes = {
        ...     'Train': data_train,
        ...     'Validation': data_val,
        ...     'Test': data_test
        ... }
        >>> visualize_ts_plotly(dataframes)

        Single dataset example:

        >>> # Visualize single dataset
        >>> dataframes = {'Data': data_train}
        >>> visualize_ts_plotly(dataframes, columns=['temperature'])

        Custom styling:

        >>> visualize_ts_plotly(
        ...     dataframes,
        ...     columns=['temperature'],
        ...     template='plotly_dark',
        ...     colors={'Train': 'blue', 'Validation': 'green', 'Test': 'red'},
        ...     mode='lines+markers'
        ... )
    """
    if go is None:
        raise ImportError(
            "plotly is required for this function. " "Install with: pip install plotly"
        )

    if not isinstance(dataframes, dict):
        raise TypeError("dataframes parameter must be a dictionary")

    if not dataframes:
        raise ValueError("dataframes dictionary is empty")

    # Validate all dataframes have data
    for name, df in dataframes.items():
        if df.empty:
            raise ValueError(f"DataFrame '{name}' is empty")
        if len(df.columns) == 0:
            raise ValueError(f"DataFrame '{name}' contains no columns")

    # Determine columns to plot
    all_columns = set()
    for df in dataframes.values():
        all_columns.update(df.columns)

    if not all_columns:
        raise ValueError("No columns found in any dataframe")

    columns_to_plot = columns if columns is not None else sorted(list(all_columns))

    # Validate columns exist in all dataframes
    for col in columns_to_plot:
        for name, df in dataframes.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe '{name}'")

    # Default colors if not provided
    if colors is None:
        # Use a set of distinct colors
        default_colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#17becf",  # cyan
        ]
        colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(dataframes.keys())
        }

    # Create figures for each column
    for col in columns_to_plot:
        fig = go.Figure()

        # Add trace for each dataset
        for dataset_name, df in dataframes.items():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=dataset_name,
                    line=dict(color=colors[dataset_name]),
                    **kwargs,
                )
            )

        # Create title
        title = col
        if title_suffix:
            title = f"{col} {title_suffix}"

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=col,
            width=figsize[0],
            height=figsize[1],
            template=template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
        )

        fig.show()


def visualize_ts_comparison(
    dataframes: Dict[str, pd.DataFrame],
    columns: Optional[List[str]] = None,
    title_suffix: str = "",
    figsize: tuple[int, int] = (1000, 500),
    template: str = "plotly_white",
    colors: Optional[Dict[str, str]] = None,
    show_mean: bool = False,
    **kwargs: Any,
) -> None:
    """Visualize time series with optional statistical overlays.

    Similar to visualize_ts_plotly but adds options for statistical overlays
    like mean values across all datasets.

    Args:
        dataframes: Dictionary mapping dataset names to pandas DataFrames.
        columns: List of column names to visualize. If None, all columns are used.
            Default: None.
        title_suffix: Suffix to append to column names. Default: "".
        figsize: Figure size as (width, height) in pixels. Default: (1000, 500).
        template: Plotly template. Default: 'plotly_white'.
        colors: Dictionary mapping dataset names to colors. Default: None.
        show_mean: If True, overlay the mean of all datasets. Default: False.
        **kwargs: Additional keyword arguments for go.Scatter().

    Returns:
        None. Displays Plotly figures.

    Raises:
        ValueError: If dataframes is empty.
        ImportError: If plotly is not installed.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2.preprocessing.time_series_visualization import visualize_ts_comparison
        >>>
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> dates1 = pd.date_range('2024-01-01', periods=100, freq='h')
        >>> dates2 = pd.date_range('2024-05-11', periods=100, freq='h')
        >>>
        >>> df1 = pd.DataFrame({
        ...     'temperature': np.random.normal(20, 5, 100)
        ... }, index=dates1)
        >>>
        >>> df2 = pd.DataFrame({
        ...     'temperature': np.random.normal(22, 5, 100)
        ... }, index=dates2)
        >>>
        >>> # Compare with mean overlay
        >>> visualize_ts_comparison(
        ...     {'Dataset1': df1, 'Dataset2': df2},
        ...     show_mean=True
        ... )
    """
    if go is None:
        raise ImportError(
            "plotly is required for this function. " "Install with: pip install plotly"
        )

    if not dataframes:
        raise ValueError("dataframes dictionary is empty")

    # First visualize normally
    visualize_ts_plotly(
        dataframes,
        columns=columns,
        title_suffix=title_suffix,
        figsize=figsize,
        template=template,
        colors=colors,
        **kwargs,
    )

    # If show_mean, create additional mean plot
    if show_mean:
        # Determine columns to plot
        all_columns = set()
        for df in dataframes.values():
            all_columns.update(df.columns)

        columns_to_plot = columns if columns is not None else sorted(list(all_columns))

        for col in columns_to_plot:
            fig = go.Figure()

            # Add individual traces
            if colors is None:
                default_colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                ]
                colors_dict = {
                    name: default_colors[i % len(default_colors)]
                    for i, name in enumerate(dataframes.keys())
                }
            else:
                colors_dict = colors

            for dataset_name, df in dataframes.items():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode="lines",
                        name=dataset_name,
                        line=dict(color=colors_dict[dataset_name], width=1),
                        opacity=0.5,
                        **kwargs,
                    )
                )

            # Calculate and add mean
            # Align all dataframes by index and compute mean
            aligned_dfs = [
                dataframes[name][[col]].rename(columns={col: name})
                for name in dataframes.keys()
            ]
            combined = pd.concat(aligned_dfs, axis=1)
            mean_values = combined.mean(axis=1)

            fig.add_trace(
                go.Scatter(
                    x=mean_values.index,
                    y=mean_values,
                    mode="lines",
                    name="Mean",
                    line=dict(color="black", width=3, dash="dash"),
                )
            )

            title = f"{col} (with mean){title_suffix}"

            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=col,
                width=figsize[0],
                height=figsize[1],
                template=template,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                hovermode="x unified",
            )

            fig.show()


def plot_zoomed_timeseries(
    data: pd.DataFrame,
    target: str,
    zoom: tuple[str, str],
    title: Optional[str] = None,
    figsize: tuple[int, int] = (8, 4),
    show: bool = True,
) -> plt.Figure:
    """Plot a time series with a zoomed-in focus area.

    Creates a two-panel plot:
    1. Top panel: Full time series with the zoom area highlighted.
    2. Bottom panel: Zoomed-in view of the specified time range.

    Args:
        data: DataFrame containing the time series data. Must have a DatetimeIndex
            or an index convertible to datetime.
        target: Name of the column to plot.
        zoom: Tuple of (start_date, end_date) strings defining the zoom range.
        title: Optional title for the plot. If None, defaults to target name.
        figsize: Figure dimensions (width, height). Defaults to (8, 4).
        show: Whether to display the plot immediately. Defaults to True.

    Returns:
        plt.Figure: The matplotlib Figure object.

    Examples:
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from spotforecast2.preprocessing.time_series_visualization import plot_zoomed_timeseries
        >>> # Create sample data
        >>> dates = pd.date_range("2023-01-01", periods=100, freq="h")
        >>> df = pd.DataFrame({"value": range(100)}, index=dates)
        >>> # Plot with zoom
        >>> fig = plot_zoomed_timeseries(
        ...     data=df,
        ...     target="value",
        ...     zoom=("2023-01-02 00:00", "2023-01-03 00:00"),
        ...     show=False
        ... )
        >>> plt.close(fig)
    """
    if title is None:
        title = target

    fig, axs = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 2]}
    )

    # Top plot: Full series with highlighted zoom area
    data[target].plot(ax=axs[0], color="black", alpha=0.5)
    axs[0].axvspan(zoom[0], zoom[1], color="blue", alpha=0.7)
    axs[0].set_title(f"{title}")
    axs[0].set_xlabel("")
    axs[0].grid(True)

    # Bottom plot: Zoomed view
    data.loc[zoom[0] : zoom[1], target].plot(ax=axs[1], color="blue")
    axs[1].set_title(f"Zoom: {zoom[0]} to {zoom[1]}", fontsize=10)
    axs[1].grid(True)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_seasonality(
    data: pd.DataFrame,
    target: str,
    figsize: tuple[int, int] = (8, 5),
    show: bool = True,
    logscale: Union[bool, list[bool]] = False,
) -> plt.Figure:
    """Plot seasonal patterns (annual, weekly, daily) for a given target.

    Creates a 2x2 grid of plots:
    1. Distribution by month (boxplot + median).
    2. Distribution by week day (boxplot + median).
    3. Distribution by hour of day (boxplot + median).
    4. Mean target value by day of week and hour.

    Args:
        data: DataFrame containing the time series data. Must have a DatetimeIndex
            or an index convertible to datetime.
        target: Name of the column to plot.
        figsize: Figure dimensions (width, height). Defaults to (8, 5).
        show: Whether to display the plot immediately. Defaults to True.
        logscale: Whether to use a log scale for the y-axis.
            Can be a single boolean (applies to all 4 plots) or a list of 4
            booleans (applies to each plot individually). Defaults to False.

    Returns:
        plt.Figure: The matplotlib Figure object.

    Examples:
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from spotforecast2.preprocessing.time_series_visualization import plot_seasonality
        >>> # Create sample data
        >>> dates = pd.date_range("2023-01-01", periods=1000, freq="h")
        >>> df = pd.DataFrame({"value": range(1, 1001)}, index=dates)
        >>> # Plot seasonality with log scale for all plots
        >>> fig = plot_seasonality(data=df, target="value", logscale=True, show=False)
        >>> plt.close(fig)
        >>> # Plot seasonality with log scale for the first plot only
        >>> fig = plot_seasonality(
        ...     data=df,
        ...     target="value",
        ...     logscale=[True, False, False, False],
        ...     show=False
        ... )
        >>> plt.close(fig)
    """
    # Work on a copy to avoid modifying the original dataframe with localized features
    df = data.copy()

    # Create temporal features
    df["month"] = df.index.month
    df["week_day"] = df.index.day_of_week + 1
    df["hour_day"] = df.index.hour + 1

    # Handle logscale
    if isinstance(logscale, bool):
        logscales = [logscale] * 4
        sharey = True
    else:
        if len(logscale) != 4:
            raise ValueError("logscale list must have length 4.")
        logscales = logscale
        # If different scales are used, we should not share y-axis
        sharey = len(set(logscales)) == 1

    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=sharey)
    axs = axs.ravel()

    # 1. Distribution by month
    df.boxplot(
        column=target, by="month", ax=axs[0], flierprops={"markersize": 3, "alpha": 0.3}
    )
    df.groupby("month")[target].median().plot(style="o-", linewidth=0.8, ax=axs[0])
    axs[0].set_ylabel(target)
    axs[0].set_title(f"{target} distribution by month", fontsize=9)

    # 2. Distribution by week day
    df.boxplot(
        column=target,
        by="week_day",
        ax=axs[1],
        flierprops={"markersize": 3, "alpha": 0.3},
    )
    df.groupby("week_day")[target].median().plot(style="o-", linewidth=0.8, ax=axs[1])
    axs[1].set_ylabel(target)
    axs[1].set_title(f"{target} distribution by week day", fontsize=9)

    # 3. Distribution by the hour of the day
    df.boxplot(
        column=target,
        by="hour_day",
        ax=axs[2],
        flierprops={"markersize": 3, "alpha": 0.3},
    )
    df.groupby("hour_day")[target].median().plot(style="o-", linewidth=0.8, ax=axs[2])
    axs[2].set_ylabel(target)
    axs[2].set_title(f"{target} distribution by the hour of the day", fontsize=9)

    # 4. Distribution by week day and hour of the day
    mean_day_hour = df.groupby(["week_day", "hour_day"])[target].mean()
    mean_day_hour.plot(ax=axs[3])
    axs[3].set(
        title=f"Mean {target} during week",
        xticks=[i * 24 for i in range(7)],
        xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        xlabel="Day and hour",
        ylabel=f"Number of {target}",
    )
    axs[3].grid(True)
    axs[3].title.set_size(10)

    # Apply logscale
    for i, ax in enumerate(axs):
        if logscales[i]:
            ax.set_yscale("log")

    fig.suptitle(f"Seasonality plots: {target}", fontsize=12)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_predictions(
    y_true: Union[pd.Series, pd.DataFrame],
    predictions: Dict[str, Union[pd.Series, pd.DataFrame, np.ndarray]],
    slice_seq: Optional[slice] = None,
    title: str = "Predictions vs Actuals",
    figsize: Optional[tuple] = None,
    show: bool = True,
    nrows: Optional[int] = None,
    ncols: int = 1,
    sharex: bool = True,
) -> plt.Figure:
    """Plot actual values against one or more prediction series.

    Allows visualizing model performance by overlaying predictions on top of
    actual data. Supports slicing to focus on a specific time range (e.g.,
    the recent test set). Handles both univariate and multivariate targets
    by creating subplots for multiple targets.

    Args:
        y_true: Series or DataFrame containing the actual target values.
        predictions: Dictionary where keys are labels (e.g., model names) and
            values are the corresponding predictions.
            If arrays are provided, they must have the same length as the
            sliced `y_true`.
        slice_seq: Optional slice object to select a subset of the data.
            If None, the entire series is plotted.
            Example: `slice(-96, None)` to select the last 96 points.
        title: Title of the plot. Defaults to "Predictions vs Actuals".
        figsize: Tuple defining figure width and height. If None, automatically
            calculated based on number of subplots.
        show: Whether to display the plot. Defaults to True.
        nrows: Number of rows for subplots (multivariate). Defaults to n_targets.
        ncols: Number of columns for subplots (multivariate). Defaults to 1.
        sharex: Whether to share x-axis for subplots. Defaults to True.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2.preprocessing.time_series_visualization import plot_predictions
        >>> # Create sample data
        >>> dates = pd.date_range("2023-01-01", periods=10, freq="D")
        >>> y_true = pd.Series(np.arange(10), index=dates, name="Target")
        >>> predictions = {"Model A": y_true + 0.5}
        >>> # Plot predictions
        >>> fig = plot_predictions(y_true, predictions, show=False)
        >>> plt.close(fig)
    """
    if slice_seq is None:
        slice_seq = slice(None)

    # Handle y_true slicing
    y_plot = y_true.iloc[slice_seq]

    # Determine dimensions
    if isinstance(y_plot, pd.Series):
        targets = [y_plot.name] if y_plot.name else ["Target"]
        # Convert to DataFrame for consistent interface
        y_plot = y_plot.to_frame(name=targets[0])
    else:
        targets = y_plot.columns.tolist()

    n_targets = len(targets)

    # Setup layout
    if nrows is None:
        nrows = n_targets

    # Check if nrows * ncols covers all targets
    if nrows * ncols < n_targets:
        # Auto-adjust if invalid
        nrows = (n_targets + ncols - 1) // ncols

    if figsize is None:
        figsize = (12, 4 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        squeeze=False,  # Ensure axes is always 2D array
    )
    fig.suptitle(title)

    # Flatten axes for iteration
    axes_flat = axes.flatten()

    for i, target in enumerate(targets):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]

        # Plot Actuals
        target_actuals = y_plot[target]
        ax.plot(
            target_actuals.index,
            target_actuals.values,
            "x-",
            alpha=0.5,
            label="Actual",
            color="black",
            linewidth=2,
        )

        # Plot Predictions
        for label, y_pred in predictions.items():
            if isinstance(y_pred, pd.DataFrame):
                # Try specific column logic
                if target in y_pred.columns:
                    pred_part = y_pred[target]
                elif len(y_pred.columns) == n_targets:
                    # Assume aligned order? Risky but fallback
                    pred_part = y_pred.iloc[:, i]
                else:
                    continue  # Warning?

            elif isinstance(y_pred, np.ndarray):
                # If array, check dimensions
                if y_pred.ndim > 1 and y_pred.shape[1] == n_targets:
                    pred_part = y_pred[:, i]
                elif y_pred.ndim == 1 and n_targets == 1:
                    pred_part = y_pred
                else:
                    continue  # Mismatch

            elif isinstance(y_pred, pd.Series):
                if n_targets == 1:
                    pred_part = y_pred
                else:
                    continue  # Mismatch?

            else:
                continue

            # Process slice/alignment for pred_part
            # Logic borrowed from previous:
            # If length matches full y_true, slice it.
            # If length matches y_plot (sliced), use as is.

            full_len = len(y_true)
            sliced_len = len(y_plot)

            vals_to_plot = None

            # Simple heuristic
            if isinstance(pred_part, (pd.Series, pd.DataFrame)):
                vals_to_plot = pred_part.values
            else:
                vals_to_plot = pred_part

            if len(vals_to_plot) == full_len:
                vals_to_plot = vals_to_plot[slice_seq]
            elif len(vals_to_plot) != sliced_len:
                # Length mismatch warning?
                pass

            ax.plot(target_actuals.index, vals_to_plot, "x-", label=label, alpha=0.8)

        ax.set_title(target)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if not sharex:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    if sharex:
        # Rotate labels for bottom row axes
        for ax in axes[-1, :]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_forecast(
    model: Any,
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    cv_results: Optional[Dict[str, Any]] = None,
    title: str = "Forecast",
    figsize: Optional[tuple] = None,
    show: bool = True,
    nrows: Optional[int] = None,
    ncols: int = 1,
    sharex: bool = True,
) -> plt.Figure:
    """Plot model forecast against actuals and display CV metrics.

    Args:
        model: Fitted scikit-learn model.
        X: Feature matrix (e.g., test set).
        y: Target series or DataFrame (e.g., test set).
        cv_results: Optional dictionary of cross-validation results from
            `evaluate()` or `sklearn.model_selection.cross_validate()`.
        title: Title of the plot. Defaults to "Forecast".
        figsize: Figure dimensions.
        show: Whether to display the plot. Defaults to True.
        nrows: Number of rows for subplots (multivariate).
        ncols: Number of columns for subplots (multivariate).
        sharex: Whether to share x-axis for subplots. Defaults to True.

    Returns:
        plt.Figure: The matplotlib Figure object.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> from spotforecast2.preprocessing.time_series_visualization import plot_forecast
        >>> # Create sample data
        >>> dates = pd.date_range("2023-01-01", periods=10, freq="D")
        >>> X = pd.DataFrame({"feat": np.arange(10)}, index=dates)
        >>> y = pd.Series(np.arange(10), index=dates)
        >>> model = LinearRegression().fit(X, y)
        >>> # Plot forecast
        >>> fig = plot_forecast(model, X, y, show=False)
        >>> plt.close(fig)
    """
    # 1. Generate predictions/forecast
    # Assume model is already fitted
    y_pred = model.predict(X)

    # 2. Format title with metrics if available
    if cv_results:
        metrics_str = []
        if "test_neg_mean_absolute_error" in cv_results:
            mae = -cv_results["test_neg_mean_absolute_error"]
            metrics_str.append(f"MAE: {np.mean(mae):.3f} (±{np.std(mae):.3f})")
        if "test_neg_root_mean_squared_error" in cv_results:
            rmse = -cv_results["test_neg_root_mean_squared_error"]
            metrics_str.append(f"RMSE: {np.mean(rmse):.3f} (±{np.std(rmse):.3f})")

        if metrics_str:
            title += "\n" + " | ".join(metrics_str)

    # 3. Plot
    predictions = {"Forecast": y_pred}
    return plot_predictions(
        y,
        predictions,
        slice_seq=None,
        title=title,
        figsize=figsize,
        show=show,
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
    )

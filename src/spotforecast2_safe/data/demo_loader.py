# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Demo data loader for safety-critical forecasting tasks.

This module provides flexible data loading functions for ground truth
validation in forecasting demonstrations and production workflows.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from spotforecast2_safe.data.demo_data import DemoConfig
from spotforecast2_safe.processing.agg_predict import agg_predict


def load_actual_combined(
    config: DemoConfig,
    columns: List[str],
    forecast_horizon: Optional[int] = None,
    weights: Optional[List[float]] = None,
    data_path: Optional[Path] = None,
) -> pd.Series:
    """
    Load ground truth and compute combined actual series with validation.

    This function loads a CSV file containing ground truth data, validates
    the presence of required columns, extracts a subset based on forecast
    horizon, and aggregates multiple columns using weighted averaging.

    Args:
        config: Configuration object containing default paths and parameters.
        columns: List of column names to extract from the ground truth data.
        forecast_horizon: Number of time steps to extract. If None, uses
            config.forecast_horizon. Must be positive.
        weights: Weights for aggregating columns. If None, uses config.weights.
            Length must match number of columns.
        data_path: Path to the ground truth CSV file. If None, uses
            config.data_path. File must exist.

    Returns:
        Aggregated time series combining all columns with specified weights.

    Raises:
        FileNotFoundError: If the ground truth file does not exist.
        ValueError: If required columns are missing from the data or if
            weights length doesn't match columns.

    Examples:
        Basic usage with explicit forecast horizon and weights:

        ```{python}
        import tempfile
        from pathlib import Path

        import pandas as pd

        from spotforecast2_safe.data.demo_data import DemoConfig
        from spotforecast2_safe.data.demo_loader import load_actual_combined

        idx = pd.date_range("2020-01-01", periods=3, freq="h", name="timestamp")
        sample_df = pd.DataFrame({"col1": [1.0, 3.0, 5.0], "col2": [2.0, 4.0, 6.0]}, index=idx)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_df.to_csv(f.name)
            temp_path = Path(f.name)

        config = DemoConfig(data_path=temp_path)
        result = load_actual_combined(
            config, columns=["col1", "col2"], forecast_horizon=2, weights=[1.0, 1.0]
        )
        print(f"Result length: {len(result)}")
        print(f"First value: {result.iloc[0]:.1f}")
        temp_path.unlink()
        ```

        Override forecast horizon and weights:

        ```{python}
        idx = pd.date_range("2020-01-01", periods=10, freq="h", name="timestamp")
        sample_df = pd.DataFrame(
            {"col1": [float(i) for i in range(10)], "col2": [float(i * 2) for i in range(10)]},
            index=idx,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_df.to_csv(f.name)
            temp_path = Path(f.name)

        config = DemoConfig(data_path=temp_path, forecast_horizon=24)
        result = load_actual_combined(
            config, columns=["col1", "col2"], forecast_horizon=5, weights=[1.0, 0.5]
        )
        print(f"Custom horizon length: {len(result)}")
        temp_path.unlink()
        ```

        Override `data_path` while keeping the rest of the config:

        ```{python}
        idx = pd.date_range("2020-01-01", periods=1, freq="h", name="timestamp")
        sample_df = pd.DataFrame({"X": [100.0], "Y": [200.0]}, index=idx)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_df.to_csv(f.name)
            custom_path = Path(f.name)

        config = DemoConfig()  # default data_path
        result = load_actual_combined(
            config,
            columns=["X", "Y"],
            data_path=custom_path,
            forecast_horizon=1,
            weights=[0.5, 0.5],
        )
        print(f"Custom path result: {result.iloc[0]:.1f}")
        custom_path.unlink()
        ```

        Error handling — missing file:

        ```{python}
        config = DemoConfig(data_path=Path("/nonexistent/file.csv"))
        try:
            load_actual_combined(
                config, columns=["A"], forecast_horizon=1, weights=[1.0]
            )
        except FileNotFoundError:
            print("File not found as expected")
        ```

        Error handling — missing columns:

        ```{python}
        idx = pd.date_range("2020-01-01", periods=1, freq="h", name="timestamp")
        sample_df = pd.DataFrame({"col1": [1.0]}, index=idx)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_df.to_csv(f.name)
            temp_path = Path(f.name)

        config = DemoConfig(data_path=temp_path)
        try:
            load_actual_combined(
                config,
                columns=["col1", "col2"],
                forecast_horizon=1,
                weights=[1.0, 1.0],
            )
        except ValueError:
            print("Missing columns detected")
        temp_path.unlink()
        ```

        Production usage with horizon and weights drawn from the config:

        ```{python}
        idx = pd.date_range("2020-01-01", periods=30, freq="h", name="timestamp")
        sample_df = pd.DataFrame(
            {
                "load": [100 + i for i in range(30)],
                "solar": [50 + i for i in range(30)],
                "wind": [25 + i for i in range(30)],
            },
            index=idx,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_df.to_csv(f.name)
            temp_path = Path(f.name)

        config = DemoConfig(
            data_path=temp_path, forecast_horizon=24, weights=[1.0, -0.5, -0.5]
        )
        result = load_actual_combined(config, columns=["load", "solar", "wind"])
        print(f"Production forecast length: {len(result)}")
        print(f"Result is pandas Series: {isinstance(result, pd.Series)}")
        temp_path.unlink()
        ```
    """
    # Use config defaults if parameters not provided
    actual_forecast_horizon = (
        forecast_horizon if forecast_horizon is not None else config.forecast_horizon
    )
    actual_weights = weights if weights is not None else config.weights
    actual_data_path = data_path if data_path is not None else config.data_path

    # Validate file existence
    if not actual_data_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {actual_data_path}")

    # Load data
    data_test = pd.read_csv(actual_data_path, index_col=0, parse_dates=True)

    # Safety Check: Ensure all requested columns exist in the test data
    missing_cols = [col for col in columns if col not in data_test.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in test data: {missing_cols}")

    # Extract forecast horizon subset
    actual_df = data_test[columns].iloc[:actual_forecast_horizon]

    # Weight validation happens inside agg_predict
    return agg_predict(actual_df, weights=actual_weights)

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

from spotforecast2_safe.manager.datasets.demo_data import DemoConfig
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
        >>> import tempfile
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.datasets.demo_data import DemoConfig
        >>> from spotforecast2_safe.manager.datasets.demo_loader import load_actual_combined
        >>>
        >>> # Example 1: Basic usage with default config parameters
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        ...     _ = f.write('timestamp,col1,col2\\n')
        ...     _ = f.write('2020-01-01 00:00:00,1.0,2.0\\n')
        ...     _ = f.write('2020-01-01 01:00:00,3.0,4.0\\n')
        ...     _ = f.write('2020-01-01 02:00:00,5.0,6.0\\n')
        ...     temp_path = Path(f.name)
        >>> config = DemoConfig(data_path=temp_path)
        >>> result = load_actual_combined(config, columns=['col1', 'col2'],
        ...                               forecast_horizon=2, weights=[1.0, 1.0])
        >>> print(f"Result length: {len(result)}")
        Result length: 2
        >>> print(f"First value: {result.iloc[0]:.1f}")
        First value: 3.0
        >>> temp_path.unlink()  # Clean up
        >>>
        >>> # Example 2: Override forecast_horizon
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        ...     _ = f.write('timestamp,col1,col2\\n')
        ...     for i in range(10):
        ...         _ = f.write(f'2020-01-01 {i:02d}:00:00,{i}.0,{i*2}.0\\n')
        ...     temp_path = Path(f.name)
        >>> config = DemoConfig(data_path=temp_path, forecast_horizon=24)
        >>> result = load_actual_combined(config, columns=['col1', 'col2'],
        ...                               forecast_horizon=5, weights=[1.0, 0.5])
        >>> print(f"Custom horizon length: {len(result)}")
        Custom horizon length: 5
        >>> temp_path.unlink()
        >>>
        >>> # Example 3: Override weights for custom aggregation
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        ...     _ = f.write('timestamp,A,B,C\\n')
        ...     _ = f.write('2020-01-01 00:00:00,10.0,5.0,2.0\\n')
        ...     _ = f.write('2020-01-01 01:00:00,20.0,10.0,4.0\\n')
        ...     temp_path = Path(f.name)
        >>> config = DemoConfig(data_path=temp_path)
        >>> result = load_actual_combined(config, columns=['A', 'B', 'C'],
        ...                               forecast_horizon=2,
        ...                               weights=[1.0, -1.0, 1.0])
        >>> print(f"Weighted result shape: {result.shape}")
        Weighted result shape: (2,)
        >>> temp_path.unlink()
        >>>
        >>> # Example 4: Override data_path
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        ...     _ = f.write('timestamp,X,Y\\n')
        ...     _ = f.write('2020-01-01 00:00:00,100.0,200.0\\n')
        ...     custom_path = Path(f.name)
        >>> config = DemoConfig()  # Uses default path
        >>> result = load_actual_combined(config, columns=['X', 'Y'],
        ...                               data_path=custom_path,
        ...                               forecast_horizon=1,
        ...                               weights=[0.5, 0.5])
        >>> print(f"Custom path result: {result.iloc[0]:.1f}")
        Custom path result: 150.0
        >>> custom_path.unlink()
        >>>
        >>> # Example 5: Error handling - missing file
        >>> config = DemoConfig(data_path=Path('/nonexistent/file.csv'))
        >>> try:
        ...     result = load_actual_combined(config, columns=['A'],
        ...                                   forecast_horizon=1, weights=[1.0])
        ... except FileNotFoundError as e:
        ...     print("File not found as expected")
        File not found as expected
        >>>
        >>> # Example 6: Error handling - missing columns
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        ...     _ = f.write('timestamp,col1\\n')
        ...     _ = f.write('2020-01-01 00:00:00,1.0\\n')
        ...     temp_path = Path(f.name)
        >>> config = DemoConfig(data_path=temp_path)
        >>> try:
        ...     result = load_actual_combined(config, columns=['col1', 'col2'],
        ...                                   forecast_horizon=1, weights=[1.0, 1.0])
        ... except ValueError as e:
        ...     print("Missing columns detected")
        Missing columns detected
        >>> temp_path.unlink()
        >>>
        >>> # Example 7: Production usage with all defaults from config
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        ...     _ = f.write('timestamp,load,solar,wind\\n')
        ...     for i in range(30):
        ...         _ = f.write(f'2020-01-01 {i:02d}:00:00,{100+i},{50+i},{25+i}\\n')
        ...     temp_path = Path(f.name)
        >>> config = DemoConfig(
        ...     data_path=temp_path,
        ...     forecast_horizon=24,
        ...     weights=[1.0, -0.5, -0.5]
        ... )
        >>> result = load_actual_combined(config, columns=['load', 'solar', 'wind'])
        >>> print(f"Production forecast length: {len(result)}")
        Production forecast length: 24
        >>> print(f"Result is pandas Series: {isinstance(result, pd.Series)}")
        Result is pandas Series: True
        >>> temp_path.unlink()
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

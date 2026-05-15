# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Demo dataset configuration for safety-critical forecasting tasks.

This module provides a flexible configuration dataclass for managing
parameters in demonstration and production forecasting workflows.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from spotforecast2_safe.data.fetch_data import get_package_data_home


@dataclass(frozen=True)
class DemoConfig:
    """
    Configuration for the safety-critical demo task.

    This immutable dataclass encapsulates all parameters needed for
    forecasting demonstrations, including data paths, model directories,
    and hyperparameters. All fields have sensible defaults but can be
    overridden during initialization.

    Attributes:
        data_path: Path to the ground truth CSV file for testing.
        model_root: Root directory for storing trained models.
        log_root: Directory for storing log files.
        forecast_horizon: Number of time steps to forecast ahead.
        contamination: Contamination factor for outlier detection (0.0-1.0).
        window_size: Size of the sliding window for feature extraction.
        lags: Number of lagged features to include.
        train_ratio: Ratio of data used for training (0.0-1.0).
        random_seed: Random seed for reproducibility.
        weights: Weights for aggregating multi-variable predictions.

    Examples:
        Default configuration:

        ```{python}
        from spotforecast2_safe.data.demo_data import DemoConfig

        config = DemoConfig()
        print(f"Forecast horizon: {config.forecast_horizon}")
        print(f"Window size: {config.window_size}")
        print(f"Random seed: {config.random_seed}")
        ```

        Override specific parameters:

        ```{python}
        config = DemoConfig(forecast_horizon=48, contamination=0.05, random_seed=123)
        print(f"Forecast horizon: {config.forecast_horizon}")
        print(f"Contamination: {config.contamination}")
        ```

        Customize paths for a specific environment:

        ```{python}
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_config = DemoConfig(
                data_path=Path(tmpdir) / "my_data.csv",
                model_root=Path(tmpdir) / "models",
                log_root=Path(tmpdir) / "logs",
            )
            print(f"Data path name: {custom_config.data_path.name}")
            print(f"Model root name: {custom_config.model_root.name}")
        ```

        Custom weights for multi-variable aggregation:

        ```{python}
        config = DemoConfig(weights=[1.0, 1.0, -1.0, 1.0, 1.0])
        print(f"Number of weights: {len(config.weights)}")
        ```

        Immutability check (frozen dataclass):

        ```{python}
        config = DemoConfig()
        try:
            config.forecast_horizon = 100
        except AttributeError:
            print("Config is immutable as expected")
        ```

        Production configuration with a long horizon:

        ```{python}
        prod_config = DemoConfig(
            forecast_horizon=168,  # 1 week
            window_size=336,       # 2 weeks
            lags=48,
            train_ratio=0.9,
            contamination=0.005,
        )
        print(f"Production horizon: {prod_config.forecast_horizon} hours")
        print(f"Training ratio: {prod_config.train_ratio}")
        ```
    """

    data_path: Path = field(
        default_factory=lambda: get_package_data_home() / "demo11.csv"
    )
    model_root: Path = field(
        default_factory=lambda: Path.home() / "spotforecast2_safe_models"
    )
    log_root: Path = field(
        default_factory=lambda: Path.home() / "spotforecast2_safe_models" / "logs"
    )
    forecast_horizon: int = 24
    contamination: float = 0.01
    window_size: int = 72
    lags: int = 24
    train_ratio: float = 0.8
    random_seed: int = 42

    weights: List[float] = field(
        default_factory=lambda: [
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
        ]
    )

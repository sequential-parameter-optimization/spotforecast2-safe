# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration for ENTSO-E task pipeline."""

from typing import List, Optional

import pandas as pd
from spotforecast2_safe.data import Period


class ConfigEntsoe:
    """Configuration for the ENTSO-E forecasting pipeline.

    This class manages all configuration parameters for the ENTSO-E task,
    including API settings, training/prediction intervals, and feature
    engineering specifications. All parameters can be customized during
    initialization or used with sensible defaults.

    Args:
        api_country_code: ISO country code for ENTSO-E API queries.
        periods: List of Period objects defining cyclical feature encodings.
        lags_consider: List of lag values to consider for feature selection.
        train_size: Time window for training data.
        end_train_default: Default end date for training period (ISO format with timezone).
        delta_val: Validation window size.
        predict_size: Number of hours to predict ahead.
        refit_size: Number of days between model refits.
        random_state: Random seed for reproducibility.
        n_hyperparameters_trials: Number of trials for hyperparameter optimization.
        data_filename: Path to the interim merged data file.

    Attributes:
        API_COUNTRY_CODE: ISO country code for API queries.
        periods: Cyclical feature encoding specifications.
        lags_consider: Lag values for autoregressive features.
        train_size: Training data window.
        end_train_default: Default training end date.
        delta_val: Validation window.
        predict_size: Prediction horizon in hours.
        refit_size: Refit interval in days.
        random_state: Random seed.
        n_hyperparameters_trials: Hyperparameter tuning trials.

    Examples:
        >>> from spotforecast2_safe import Config
        >>> import pandas as pd
        >>>
        >>> # Use default configuration
        >>> config = Config()
        >>> config.API_COUNTRY_CODE
        'DE'
        >>> config.predict_size
        24
        >>> config.random_state
        314159
        >>>
        >>> # Create custom configuration
        >>> custom_config = Config(
        ...     api_country_code='FR',
        ...     predict_size=48,
        ...     random_state=42
        ... )
        >>> custom_config.API_COUNTRY_CODE
        'FR'
        >>> custom_config.predict_size
        48
        >>>
        >>> # Verify training window
        >>> config.train_size == pd.Timedelta(days=3 * 365)
        True
        >>>
        >>> # Check default periods
        >>> len(config.periods)
        5
        >>> config.periods[0].name
        'daily'
    """

    def __init__(
        self,
        api_country_code: str = "DE",
        periods: Optional[List[Period]] = None,
        lags_consider: Optional[List[int]] = None,
        train_size: Optional[pd.Timedelta] = None,
        end_train_default: str = "2025-12-31 00:00+00:00",
        delta_val: Optional[pd.Timedelta] = None,
        predict_size: int = 24,
        refit_size: int = 7,
        random_state: int = 314159,
        n_hyperparameters_trials: int = 20,
        data_filename: str = "interim/energy_load.csv",
    ):
        """Initialize ConfigEntsoe with specified or default parameters."""
        self.API_COUNTRY_CODE = api_country_code

        # Default periods use deliberate n_periods choices:
        # - daily: n_periods=12 for 24 hours (2:1 ratio) provides 2-hour resolution,
        #   balancing detail vs overfitting while reducing dimensionality by 50%
        # - weekly/monthly/quarterly: n_periods matches range_size (1:1 ratio)
        # - yearly: n_periods=12 for 365 days (30:1 ratio) provides strong smoothing
        # See docs/PERIOD_CONFIGURATION_RATIONALE.md for detailed analysis
        self.periods = (
            periods
            if periods is not None
            else [
                Period(name="daily", n_periods=12, column="hour", input_range=(1, 24)),
                Period(
                    name="weekly", n_periods=7, column="dayofweek", input_range=(0, 6)
                ),
                Period(
                    name="monthly", n_periods=12, column="month", input_range=(1, 12)
                ),
                Period(
                    name="quarterly", n_periods=4, column="quarter", input_range=(1, 4)
                ),
                Period(
                    name="yearly",
                    n_periods=12,
                    column="dayofyear",
                    input_range=(1, 365),
                ),
            ]
        )
        self.lags_consider = (
            lags_consider if lags_consider is not None else list(range(1, 24))
        )
        self.train_size = (
            train_size if train_size is not None else pd.Timedelta(days=3 * 365)
        )
        self.end_train_default = end_train_default
        self.delta_val = (
            delta_val if delta_val is not None else pd.Timedelta(hours=24 * 7 * 10)
        )
        self.predict_size = predict_size
        self.refit_size = refit_size
        self.random_state = random_state
        self.n_hyperparameters_trials = n_hyperparameters_trials
        self.data_filename = data_filename

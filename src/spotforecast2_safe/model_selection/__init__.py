"""
Model selection module for spotforecast2_safe.

This module provides tools for model selection, validation, and cross-validation
for time series forecasting.
"""

from .validation import backtesting_forecaster, _backtesting_forecaster
from .split_ts_cv import TimeSeriesFold

__all__ = ["backtesting_forecaster", "_backtesting_forecaster", "TimeSeriesFold"]


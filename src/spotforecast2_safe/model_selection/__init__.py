# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model selection module for spotforecast2_safe.

This module provides tools for model selection, validation, and cross-validation
for time series forecasting.
"""

from .validation import backtesting_forecaster, _backtesting_forecaster
from .split_ts_cv import TimeSeriesFold
from .split_one_step import OneStepAheadFold

__all__ = [
    "backtesting_forecaster",
    "_backtesting_forecaster",
    "TimeSeriesFold",
    "OneStepAheadFold",
]

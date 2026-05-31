# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Backtesting module for spotforecast2_safe.

This module provides the orchestration layer that runs a fitted
forecaster over a sequence of folds produced by
`spotforecast2_safe.splitter` and reports out-of-sample metrics.
"""

from .per_fold import metrics_per_fold
from .validation import _backtesting_forecaster, backtesting_forecaster

__all__ = [
    "_backtesting_forecaster",
    "backtesting_forecaster",
    "metrics_per_fold",
]

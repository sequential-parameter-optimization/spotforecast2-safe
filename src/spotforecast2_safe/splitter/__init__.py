# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Splitter module for spotforecast2_safe.

This module groups time-series-aware splitters:

- Cross-validation folds (`TimeSeriesFold`, `OneStepAheadFold`).
- Train / validation / test holdout helpers
  (`split_abs_train_val_test`, `split_rel_train_val_test`).

The orchestration code that consumes these splitters (backtesting) lives
in `spotforecast2_safe.backtesting`.
"""

from .split import split_abs_train_val_test, split_rel_train_val_test
from .split_one_step import OneStepAheadFold
from .split_ts_cv import TimeSeriesFold

__all__ = [
    "OneStepAheadFold",
    "TimeSeriesFold",
    "split_abs_train_val_test",
    "split_rel_train_val_test",
]

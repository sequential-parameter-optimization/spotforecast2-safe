# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application-level wrappers around ForecasterRecursive."""

from spotforecast2_safe.forecaster.wrappers.lgbm import ForecasterRecursiveLGBM
from spotforecast2_safe.forecaster.wrappers.model import ForecasterRecursiveModel
from spotforecast2_safe.forecaster.wrappers.xgb import ForecasterRecursiveXGB

__all__ = [
    "ForecasterRecursiveModel",
    "ForecasterRecursiveLGBM",
    "ForecasterRecursiveXGB",
]

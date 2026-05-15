# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exogenous feature engineering sub-package.

Provides the public weather feature helper that can be passed as an
exogenous variable to
:class:`~spotforecast2_safe.forecaster.recursive.ForecasterRecursive`
and related forecasters.

Calendar / day-night / holiday helpers moved to
:mod:`spotforecast2_safe.calendar` — import them from there.
"""

from spotforecast2_safe.manager.exo.weather import get_weather_features

__all__ = ["get_weather_features"]

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Calendar, holiday, and day/night feature engineering.

Consolidates what was previously split across
``spotforecast2_safe.holiday`` (holiday DataFrame generation) and
``spotforecast2_safe.manager.exo.calendar`` (calendar / day-night /
holiday feature builders).
"""

from spotforecast2_safe.calendar.features import (
    get_calendar_features,
    get_day_night_features,
)
from spotforecast2_safe.calendar.holiday import (
    create_holiday_df,
    get_holiday_features,
)

__all__ = [
    "create_holiday_df",
    "get_calendar_features",
    "get_day_night_features",
    "get_holiday_features",
]

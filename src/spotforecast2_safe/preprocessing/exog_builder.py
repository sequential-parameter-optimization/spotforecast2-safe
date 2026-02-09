# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exogenous feature builder for time series forecasting."""

from typing import List, Optional

import holidays
import pandas as pd

from spotforecast2_safe.data.data import Period
from spotforecast2_safe.preprocessing.repeating_basis_function import (
    RepeatingBasisFunction,
)


class ExogBuilder:
    """
    Builds a set of exogenous features for a given date range.

    This builder combines temporal features (day of year, day of week, hour, etc.)
    with cyclical features encoded via RepeatingBasisFunctions and optional
    holiday indicators.

    Attributes:
        periods (List[Period]): List of periodic features to encode.
        country_code (Optional[str]): Country code for holiday lookups.
        holidays_list (Optional[holidays.HolidayBase]): List of holidays for the specified country.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.data.data import Period
        >>> from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder
        >>> periods = [Period(name="hour", n_periods=24, column="hour", input_range=(0, 23))]
        >>> builder = ExogBuilder(periods=periods, country_code="DE")
        >>> start = pd.Timestamp("2025-01-01", tz="UTC")
        >>> end = pd.Timestamp("2025-01-02", tz="UTC")
        >>> exog = builder.build(start, end)
        >>> exog.shape[1] > 0
        True
    """

    def __init__(
        self, periods: Optional[List[Period]] = None, country_code: Optional[str] = None
    ):
        """
        Initialize the ExogBuilder.

        Args:
            periods: List of Period objects defining cyclical features.
            country_code: country code (ISO) for holiday detection.
        """
        self.periods = periods or []
        self.country_code = country_code
        self.holidays_list = (
            holidays.country_holidays(country_code) if country_code else None
        )

    def _get_time_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic time-based columns from the DataFrame index.

        Args:
            X: DataFrame with DatetimeIndex.

        Returns:
            pd.DataFrame: Copy of X with extra time columns.
        """
        X = X.copy()
        X["dayofyear"] = X.index.dayofyear
        X["dayofweek"] = X.index.dayofweek
        X["quarter"] = X.index.quarter
        X["month"] = X.index.month
        X["hour"] = X.index.hour
        return X

    def build(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Build the exogenous feature DataFrame for a date range.

        The generated DataFrame has an hourly frequency.

        Args:
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive).

        Returns:
            pd.DataFrame: DataFrame containing exogenous features.
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq="h")
        X = pd.DataFrame(index=date_range)
        X = self._get_time_columns(X)

        seasons_encoded = []
        for period in self.periods:
            rbf = RepeatingBasisFunction(
                n_periods=period.n_periods,
                column=period.column,
                input_range=period.input_range,
            )
            season_encoded = rbf.transform(X)
            cols = [f"{period.name}_{i}" for i in range(season_encoded.shape[1])]
            seasons_encoded.append(
                pd.DataFrame(season_encoded, index=X.index, columns=cols)
            )

        X_ = pd.concat(seasons_encoded, axis=1) if seasons_encoded else X

        if self.holidays_list is not None:
            # List comprehension is robust for holiday detection across different
            # pandas/holidays versions and handling of DatetimeIndex
            X_["holidays"] = [int(d in self.holidays_list) for d in X_.index]

        X_["is_weekend"] = X_.index.dayofweek.isin([5, 6]).astype(int)
        return X_

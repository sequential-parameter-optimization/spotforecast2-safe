# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exogenous feature builder for time series forecasting."""

import logging
from typing import List, Optional

import holidays
import pandas as pd

from spotforecast2_safe.data.data_classes import Period
from spotforecast2_safe.preprocessing.exog_providers import (
    ExogFeatureProvider,
    ExogProviderError,
)
from spotforecast2_safe.preprocessing.repeating_basis_function import (
    RepeatingBasisFunction,
)

logger = logging.getLogger(__name__)


class ExogBuilder:
    """
    Builds a set of exogenous features for a given date range.

    This builder combines temporal features (day of year, day of week, hour, etc.)
    with cyclical features encoded via RepeatingBasisFunctions and optional
    holiday indicators.

    Optional `ExogFeatureProvider` objects extend the built frame with additional
    drivers (e.g. ENTSO-E day-ahead forecasts, COVID infection rate). Each
    provider returns numeric, NaN-free columns aligned to the same hourly index;
    a provider that cannot cover the range is either re-raised or skipped
    according to *on_provider_failure*.

    Attributes:
        periods (List[Period]): List of periodic features to encode.
        country_code (Optional[str]): Country code for holiday lookups.
        holidays_list (Optional[holidays.HolidayBase]): List of holidays for the specified country.
        providers (List[ExogFeatureProvider]): Extra exogenous-feature providers
            appended to every built frame.
        on_provider_failure (str): ``"raise"`` (default) to propagate an
            ``ExogProviderError`` from a provider, or ``"skip"`` to log and omit
            that provider's columns.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.data.data_classes import Period
        from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder

        periods = [Period(name="hour", n_periods=24, column="hour", input_range=(0, 23))]
        builder = ExogBuilder(periods=periods, country_code="DE")
        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2025-01-02", tz="UTC")
        exog = builder.build(start, end)
        print(f"shape: {exog.shape}")
        assert exog.shape[1] > 0
        ```
    """

    def __init__(
        self,
        periods: Optional[List[Period]] = None,
        country_code: Optional[str] = None,
        providers: Optional[List[ExogFeatureProvider]] = None,
        on_provider_failure: str = "raise",
    ):
        """
        Initialize the ExogBuilder.

        Args:
            periods: List of Period objects defining cyclical features.
            country_code: country code (ISO) for holiday detection.
            providers: Optional extra exogenous-feature providers whose columns
                are appended to every built frame. Defaults to ``None`` (no
                providers; behaviour is unchanged).
            on_provider_failure: ``"raise"`` (default) to propagate an
                ``ExogProviderError`` from a provider, or ``"skip"`` to log a
                warning and omit that provider's columns.
        """
        if on_provider_failure not in ("raise", "skip"):
            raise ValueError(
                "on_provider_failure must be 'raise' or 'skip'; got "
                f"{on_provider_failure!r}."
            )
        self.periods = periods or []
        self.country_code = country_code
        self.providers = providers or []
        self.on_provider_failure = on_provider_failure
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

        Raises:
            ValueError: If the date range is invalid.

        Examples:
            ```{python}
            import pandas as pd
            from spotforecast2_safe.data.data_classes import Period
            from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder

            periods = [Period(name="hour", n_periods=24, column="hour", input_range=(0, 23))]
            builder = ExogBuilder(periods=periods, country_code="DE")
            start = pd.Timestamp("2025-01-01", tz="UTC")
            end = pd.Timestamp("2025-01-02", tz="UTC")
            exog = builder.build(start, end)
            print(f"shape: {exog.shape}, columns: {list(exog.columns[:4])}")
            assert exog.shape == (25, 26)
            assert "holidays" in exog.columns
            assert "is_weekend" in exog.columns
            ```
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

        for provider in self.providers:
            provider_name = getattr(provider, "name", provider.__class__.__name__)
            try:
                extra = provider.build(X_.index)
            except ExogProviderError as exc:
                if self.on_provider_failure == "raise":
                    raise
                logger.warning("Skipping exog provider %r: %s", provider_name, exc)
                continue
            if not extra.index.equals(X_.index):
                extra = extra.reindex(X_.index)
            X_ = pd.concat([X_, extra], axis=1)

        return X_

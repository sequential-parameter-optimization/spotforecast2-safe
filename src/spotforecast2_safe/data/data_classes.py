# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data structures for input and processed data."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from spotforecast2_safe.utils.convert_to_utc import convert_to_utc

logger = logging.getLogger(__name__)


@dataclass
class Data:
    """Container for input time series data.

    Attributes:
        data: pandas DataFrame containing the input time series data.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.data.data_classes import Data

        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"load_mw": range(48)}, index=idx)
        container = Data(data=df)
        print(type(container.data))
        print(container.data.shape)
        assert container.data.shape == (48, 1)
        assert str(container.data.index.tz) == "UTC"
        ```
    """

    data: pd.DataFrame

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        timezone: Optional[str],
        columns: Optional[List[str]] = None,
        parse_dates=True,
        index_col=0,
        **kwargs,
    ) -> "Data":
        """Load data from a CSV file.

        The CSV must contain a datetime column that becomes the DataFrame index.
        The index is localized to the provided timezone if it is naive, and then
        converted to UTC.

        Args:
            csv_path (Path): Path to the CSV file.
            timezone (Optional[str]): Timezone to assign if the index has no
                timezone. Must be provided if the index is naive.
            columns (Optional[List[str]]): List of column names to include. If
                provided, only these columns will be loaded from the CSV
                (optimizes reading speed). If None, all columns are loaded.
            parse_dates (bool or list, optional): Passed to ``pd.read_csv``.
                Defaults to True.
            index_col (int or str, optional): Column to use as index. Defaults to 0.
            **kwargs (Any): Additional keyword arguments forwarded to ``pd.read_csv``.

        Returns:
            Data: Instance containing the loaded DataFrame.

        Raises:
            ValueError: If the CSV does not yield a DatetimeIndex.
            ValueError: If the index is timezone-naive and no timezone is provided.

        Examples:
            ```{python}
            import tempfile
            from pathlib import Path

            import pandas as pd

            from spotforecast2_safe.data.data_classes import Data

            # Write a tiny CSV with a naive datetime index (UTC will be inferred)
            idx = pd.date_range("2024-01-01", periods=24, freq="h")
            df_src = pd.DataFrame({"load_mw": range(24)}, index=idx)
            with tempfile.NamedTemporaryFile(
                suffix=".csv", mode="w", delete=False
            ) as fh:
                df_src.to_csv(fh.name)
                csv_path = Path(fh.name)

            data = Data.from_csv(csv_path, timezone="UTC", columns=["load_mw"])
            print(data.data.shape)
            print(data.data.index.tz)
            assert data.data.shape == (24, 1)
            assert str(data.data.index.tz) == "UTC"
            ```
        """
        # If columns specified, add index column to usecols for efficient reading
        usecols = None
        if columns is not None:
            # Get the index column name/number
            if isinstance(index_col, int):
                # Read header first to get column names
                header_df = pd.read_csv(csv_path, nrows=0)
                index_col_name = header_df.columns[index_col]
            else:
                index_col_name = index_col
            usecols = [index_col_name] + columns

        df = pd.read_csv(
            csv_path,
            parse_dates=parse_dates,
            index_col=index_col,
            usecols=usecols,
            **kwargs,
        )
        df = convert_to_utc(df, timezone)
        if df.index.freq is None:
            try:
                df.index.freq = pd.infer_freq(df.index)
            except (ValueError, TypeError) as e:
                logger.debug(
                    "Could not infer frequency for index in from_csv: %s. "
                    "Frequency will remain None.",
                    e,
                )
        return cls(data=df)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        timezone: Optional[str],
        columns: Optional[List[str]] = None,
    ) -> "Data":
        """Create a new Data instance from an existing DataFrame.

        The DataFrame must have a datetime index. The index is localized to the
        provided timezone if it is naive, and then converted to UTC.

        Args:
            df (pd.DataFrame): Input DataFrame containing data.
            timezone (Optional[str]): Timezone to assign if the index is naive.
                Must be provided if the index has no timezone.
            columns (Optional[List[str]]): List of column names to include.
                If provided, only these columns will be selected from the
                DataFrame. If None, all columns are used.

        Returns:
            Data: Instance containing the provided DataFrame.

        Raises:
            ValueError: If the DataFrame index is not a DatetimeIndex.
            ValueError: If the index is timezone-naive and no timezone is provided.

        Examples:
            ```{python}
            import pandas as pd
            from spotforecast2_safe.data.data_classes import Data

            # Build a tz-naive hourly DataFrame and wrap it in Data
            idx = pd.date_range("2024-03-01", periods=48, freq="h")
            df = pd.DataFrame(
                {"temperature": [15.0 + i * 0.1 for i in range(48)]},
                index=idx,
            )
            container = Data.from_dataframe(df, timezone="Europe/Berlin")
            print(container.data.shape)
            print(container.data.index.tz)
            assert container.data.shape == (48, 1)
            assert str(container.data.index.tz) == "UTC"

            # Column subsetting: only keep 'temperature'
            df2 = df.copy()
            df2["extra"] = 0
            container2 = Data.from_dataframe(
                df2, timezone="Europe/Berlin", columns=["temperature"]
            )
            assert list(container2.data.columns) == ["temperature"]
            print(container2.data.columns.tolist())
            ```
        """
        df = convert_to_utc(df, timezone)

        # Select columns if specified
        if columns is not None:
            df = df[columns].copy()

        if df.index.freq is None:
            try:
                df.index.freq = pd.infer_freq(df.index)
            except (ValueError, TypeError) as e:
                logger.debug(
                    "Could not infer frequency for index in from_dataframe: %s. "
                    "Frequency will remain None.",
                    e,
                )

        return cls(data=df)


@dataclass(frozen=True)
class Period:
    """Class abstraction for the information required to encode a period.

    Attributes:
        name: Name of the period (e.g., 'hour', 'day').
        n_periods: Number of periods to encode (e.g., 24 for hours).
        column: Name of the column in the DataFrame containing the period information.
        input_range: Tuple of (min, max) values for the period (e.g., (0, 23) for hours).

    Examples:
        ```{python}
        from spotforecast2_safe.data import Period
        period = Period(name="hour", n_periods=24, column="hour", input_range=(0, 23))
        period.name
        ```
    """

    name: str
    n_periods: int
    column: str
    input_range: Tuple[int, int]

    def __post_init__(self):
        """Validate the period configuration."""
        if self.n_periods <= 0:
            raise ValueError(f"n_periods must be positive, got {self.n_periods}")
        if self.input_range[0] >= self.input_range[1]:
            raise ValueError(
                f"input_range[0] must be less than input_range[1], got {self.input_range}"
            )

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home


def test_fetch_demo01_data():
    """Test that demo01.csv is loaded correctly using get_package_data_home()."""
    demo_path = get_package_data_home() / "demo01.csv"
    df = fetch_data(filename=demo_path)

    # Check if it's a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Check for expected columns
    assert "Forecast" in df.columns
    assert "Actual" in df.columns

    # Check index is datetime with timezone
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None

    # Check that it's not empty
    assert not df.empty

    print("\nVerified demo01.csv loading:")
    print(df.head())

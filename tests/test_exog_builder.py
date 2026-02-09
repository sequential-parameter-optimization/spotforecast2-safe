import pandas as pd
import pytest
from spotforecast2_safe.data.data import Period
from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder


def test_exog_builder_basic_build():
    """Test building exogenous features with basic time columns."""
    builder = ExogBuilder(periods=[], country_code=None)
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-01-01 23:00:00", tz="UTC")
    
    exog = builder.build(start, end)
    
    assert len(exog) == 24
    assert "dayofyear" in exog.columns
    assert "hour" in exog.columns
    assert "is_weekend" in exog.columns
    assert "holidays" not in exog.columns


def test_exog_builder_with_periods():
    """Test building exogenous features with RBF periods."""
    periods = [
        Period(name="hour", n_periods=4, column="hour", input_range=(0, 23))
    ]
    builder = ExogBuilder(periods=periods, country_code=None)
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-01-01 05:00:00", tz="UTC")
    
    exog = builder.build(start, end)
    
    assert len(exog) == 6
    for i in range(4):
        assert f"hour_{i}" in exog.columns


def test_exog_builder_with_holidays():
    """Test building exogenous features with country holidays."""
    # 2025-01-01 is Neujahr in DE
    builder = ExogBuilder(periods=[], country_code="DE")
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-01-01 23:00:00", tz="UTC")
    
    exog = builder.build(start, end)
    
    assert "holidays" in exog.columns
    assert exog["holidays"].iloc[0] == 1


def test_exog_builder_docstring_example():
    """Test the example provided in the docstring."""
    # The docstring example:
    # >>> periods = [Period(name="hour", n_periods=24, column="hour", input_range=(0, 23))]
    # >>> builder = ExogBuilder(periods=periods, country_code="DE")
    # >>> start = pd.Timestamp("2025-01-01", tz="UTC")
    # >>> end = pd.Timestamp("2025-01-02", tz="UTC")
    # >>> exog = builder.build(start, end)
    
    periods = [Period(name="hour", n_periods=24, column="hour", input_range=(0, 23))]
    builder = ExogBuilder(periods=periods, country_code="DE")
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-02", tz="UTC")
    exog = builder.build(start, end)
    
    assert exog.shape[1] > 0
    assert len(exog) == 25  # Hourly from 01-01 to 01-02 inclusive (00:00 to 00:00 next day)

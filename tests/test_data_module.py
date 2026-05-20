# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest coverage for previously-untested branches in spotforecast2_safe.data.

Targets coverage gaps left after migrating the docstring examples in
`configurator/config_demo.py`, `demo_loader.py`, and `fetch_data.py` from
doctest-style blocks to Quarto `{python}` blocks (so `doctest.testmod` no
longer exercises them).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spotforecast2_safe.configurator import ConfigDemo
from spotforecast2_safe.data import (
    Data,
    Period,
    fetch_data,
    load_actual_combined,
)
from spotforecast2_safe.data.fetch_data import (
    _apply_on_missing,
    load_timeseries,
    load_timeseries_forecast,
)


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame({"y": range(24), "x": range(24, 48)}, index=idx)


@pytest.fixture
def hourly_csv(tmp_path, hourly_df) -> Path:
    csv_path = tmp_path / "data.csv"
    hourly_df.to_csv(csv_path)
    return csv_path


# --------------------------------------------------------------------------- #
# Data.from_csv / Data.from_dataframe
# --------------------------------------------------------------------------- #


class TestDataFromCsv:
    def test_round_trip_preserves_columns_and_utc(self, hourly_csv, hourly_df):
        data = Data.from_csv(hourly_csv, timezone="UTC")
        assert isinstance(data, Data)
        assert list(data.data.columns) == list(hourly_df.columns)
        assert data.data.index.tz is not None
        assert str(data.data.index.tz) == "UTC"
        assert data.data.index.freq is not None

    def test_columns_filter_uses_usecols_with_int_index(self, hourly_csv):
        data = Data.from_csv(hourly_csv, timezone="UTC", columns=["y"])
        assert list(data.data.columns) == ["y"]

    def test_columns_filter_with_named_index_col(self, hourly_df, tmp_path):
        csv_path = tmp_path / "named.csv"
        hourly_df.to_csv(csv_path, index_label="ts")
        data = Data.from_csv(csv_path, timezone="UTC", columns=["x"], index_col="ts")
        assert list(data.data.columns) == ["x"]

    def test_unparseable_freq_logs_and_keeps_none(self, tmp_path):
        irregular = pd.DataFrame(
            {"y": [1, 2, 3]},
            index=pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-05"]
            ),
        )
        csv_path = tmp_path / "irregular.csv"
        irregular.to_csv(csv_path)
        data = Data.from_csv(csv_path, timezone="UTC")
        assert data.data.index.freq is None

    def test_too_few_rows_for_freq_inference_keeps_none(self, tmp_path):
        # pd.infer_freq raises ValueError on indexes with fewer than 3 points.
        tiny = pd.DataFrame(
            {"y": [1, 2]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        csv_path = tmp_path / "tiny.csv"
        tiny.to_csv(csv_path)
        data = Data.from_csv(csv_path, timezone="UTC")
        assert data.data.index.freq is None


class TestDataFromDataframe:
    def test_from_dataframe_localizes_naive_index(self, hourly_df):
        naive = hourly_df.copy()
        naive.index = naive.index.tz_localize(None)
        data = Data.from_dataframe(naive, timezone="UTC")
        assert str(data.data.index.tz) == "UTC"

    def test_from_dataframe_column_subset(self, hourly_df):
        data = Data.from_dataframe(hourly_df, timezone=None, columns=["x"])
        assert list(data.data.columns) == ["x"]

    def test_from_dataframe_unparseable_freq_keeps_none(self):
        df = pd.DataFrame(
            {"y": [1, 2, 3]},
            index=pd.DatetimeIndex(
                ["2024-01-01", "2024-01-02", "2024-01-05"], tz="UTC"
            ),
        )
        data = Data.from_dataframe(df, timezone=None)
        assert data.data.index.freq is None

    def test_from_dataframe_too_few_rows_keeps_none(self):
        df = pd.DataFrame(
            {"y": [1, 2]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
        )
        data = Data.from_dataframe(df, timezone=None)
        assert data.data.index.freq is None


# --------------------------------------------------------------------------- #
# Period validation
# --------------------------------------------------------------------------- #


class TestPeriod:
    def test_rejects_nonpositive_n_periods(self):
        with pytest.raises(ValueError, match="n_periods must be positive"):
            Period(name="hour", n_periods=0, column="hour", input_range=(0, 23))

    def test_rejects_inverted_input_range(self):
        with pytest.raises(ValueError, match="input_range\\[0\\] must be less"):
            Period(name="hour", n_periods=24, column="hour", input_range=(23, 0))


# --------------------------------------------------------------------------- #
# fetch_data: previously-uncovered branches
# --------------------------------------------------------------------------- #


class TestFetchData:
    def test_named_index_col_with_columns_filter(self, hourly_df, tmp_path):
        csv_path = tmp_path / "named.csv"
        hourly_df.to_csv(csv_path, index_label="ts")
        df = fetch_data(filename=csv_path, columns=["x"], index_col="ts")
        assert list(df.columns) == ["x"]

    def test_unparseable_freq_is_silently_swallowed(self, tmp_path):
        irregular = pd.DataFrame(
            {"y": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]),
        )
        csv_path = tmp_path / "irregular.csv"
        irregular.to_csv(csv_path)
        df = fetch_data(filename=csv_path)
        assert df.index.freq is None

    def test_too_few_rows_for_freq_inference(self, tmp_path):
        tiny = pd.DataFrame(
            {"y": [1, 2]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        csv_path = tmp_path / "tiny.csv"
        tiny.to_csv(csv_path)
        df = fetch_data(filename=csv_path)
        assert df.index.freq is None


# --------------------------------------------------------------------------- #
# _apply_on_missing: invalid selector
# --------------------------------------------------------------------------- #


class TestApplyOnMissing:
    def test_rejects_unknown_selector(self):
        y = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="on_missing must be"):
            _apply_on_missing(y, "bogus", "col", Path("/tmp/x.csv"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# load_timeseries / load_timeseries_forecast share the new private helper
# --------------------------------------------------------------------------- #


@pytest.fixture
def energy_load_dir(tmp_path, monkeypatch):
    interim = tmp_path / "interim"
    interim.mkdir()
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"Actual Load": [10.0, 11.0, 12.0, 13.0], "Forecasted Load": [9.5, 11.2, 12.4, 12.9]},
        index=idx,
    )
    df.index.name = "Time (UTC)"
    df.to_csv(interim / "energy_load.csv")
    monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
    return tmp_path


class TestLoadTimeseries:
    def test_actual_and_forecast_share_index_and_length(self, energy_load_dir):
        y = load_timeseries()
        y_f = load_timeseries_forecast()
        assert isinstance(y, pd.Series) and isinstance(y_f, pd.Series)
        assert (y.index == y_f.index).all()
        assert y.name == "Actual Load"
        assert y_f.name == "Forecasted Load"

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="energy_load.csv"):
            load_timeseries()

    def test_explicit_data_home_argument(self, energy_load_dir):
        y = load_timeseries(data_home=energy_load_dir)
        assert len(y) == 4


# --------------------------------------------------------------------------- #
# load_actual_combined: error paths and defaulting from config
# --------------------------------------------------------------------------- #


def _write_demo_csv(path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h")
    pd.DataFrame(
        {"load": [10, 11, 12, 13, 14, 15], "solar": [1, 1, 2, 2, 3, 3]}, index=idx
    ).to_csv(path, index_label="timestamp")


class TestLoadActualCombined:
    def test_raises_when_file_missing(self, tmp_path):
        config = ConfigDemo(data_path=tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError, match="Ground truth file not found"):
            load_actual_combined(
                config, columns=["load"], forecast_horizon=1, weights=[1.0]
            )

    def test_raises_when_columns_missing(self, tmp_path):
        csv_path = tmp_path / "demo.csv"
        _write_demo_csv(csv_path)
        config = ConfigDemo(data_path=csv_path)
        with pytest.raises(ValueError, match="Missing columns"):
            load_actual_combined(
                config,
                columns=["load", "ghost"],
                forecast_horizon=2,
                weights=[1.0, 1.0],
            )

    def test_defaults_pulled_from_config_when_args_none(self, tmp_path):
        csv_path = tmp_path / "demo.csv"
        _write_demo_csv(csv_path)
        config = ConfigDemo(
            data_path=csv_path, forecast_horizon=3, weights=[1.0, -1.0]
        )
        result = load_actual_combined(config, columns=["load", "solar"])
        assert len(result) == 3

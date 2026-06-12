# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ENTSO-E interim-CSV loaders in ``data.entsoe_loader``."""

import pandas as pd
import pytest

import spotforecast2_safe.data.entsoe_loader as entsoe_loader
from spotforecast2_safe.configurator import ConfigEntsoe
from spotforecast2_safe.data.entsoe_loader import (
    entsoe_data_loader,
    entsoe_test_data_loader,
)


def _write_interim_csv(path, start: str, periods: int, tz: str | None = "UTC"):
    """Write a synthetic merged ENTSO-E CSV and return its DataFrame."""
    idx = pd.date_range(start, periods=periods, freq="h", tz=tz, name="Time (UTC)")
    df = pd.DataFrame({"Actual Load": range(periods)}, index=idx)
    df.to_csv(path)
    return df


class TestEntsoeDataLoader:
    def test_absolute_path_loads_full_frame(self, tmp_path):
        csv_path = tmp_path / "energy_load.csv"
        _write_interim_csv(csv_path, "2025-01-01", 48)

        config = ConfigEntsoe()
        config.data_filename = str(csv_path)
        df = entsoe_data_loader(config)

        assert df.shape == (48, 1)
        assert df.index.name == "Time (UTC)"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_relative_path_resolves_against_data_home(self, tmp_path, monkeypatch):
        _write_interim_csv(tmp_path / "energy_load.csv", "2025-01-01", 24)
        monkeypatch.setattr(entsoe_loader, "get_data_home", lambda: tmp_path)

        config = ConfigEntsoe()
        config.data_filename = "energy_load.csv"
        df = entsoe_data_loader(config)

        assert df.shape == (24, 1)

    def test_missing_file_raises_with_cli_hint(self, tmp_path):
        config = ConfigEntsoe()
        config.data_filename = str(tmp_path / "does_not_exist.csv")

        with pytest.raises(FileNotFoundError, match="spotforecast2-entsoe"):
            entsoe_data_loader(config)


class TestEntsoeTestDataLoader:
    def _config(self, csv_path, end_train: str, predict_size: int = 24):
        config = ConfigEntsoe()
        config.data_filename = str(csv_path)
        config.end_train_default = end_train
        config.predict_size = predict_size
        return config

    def test_slices_predict_size_steps_after_end_train(self, tmp_path):
        csv_path = tmp_path / "energy_load.csv"
        _write_interim_csv(csv_path, "2025-12-29 00:00", 120)
        config = self._config(csv_path, "2025-12-31 00:00+00:00")

        test_df = entsoe_test_data_loader(config)

        assert test_df.shape == (24, 1)
        assert test_df.index[0] == pd.Timestamp("2025-12-31 01:00", tz="UTC")
        assert test_df.index[-1] == pd.Timestamp("2026-01-01 00:00", tz="UTC")

    def test_naive_end_train_is_localized_to_utc(self, tmp_path):
        csv_path = tmp_path / "energy_load.csv"
        _write_interim_csv(csv_path, "2025-12-29 00:00", 120)
        config = self._config(csv_path, "2025-12-31 00:00")  # no tz marker

        test_df = entsoe_test_data_loader(config)

        assert test_df.shape == (24, 1)
        assert test_df.index[0] == pd.Timestamp("2025-12-31 01:00", tz="UTC")

    def test_naive_csv_index_is_supported(self, tmp_path):
        csv_path = tmp_path / "energy_load.csv"
        _write_interim_csv(csv_path, "2025-12-29 00:00", 120, tz=None)
        config = self._config(csv_path, "2025-12-31 00:00+00:00")

        test_df = entsoe_test_data_loader(config)

        assert test_df.shape == (24, 1)
        assert test_df.index[0] == pd.Timestamp("2025-12-31 01:00")
        assert test_df.index.tz is None

    def test_window_shorter_when_data_runs_out(self, tmp_path):
        csv_path = tmp_path / "energy_load.csv"
        _write_interim_csv(csv_path, "2025-12-29 00:00", 60)  # ends 12-31 11:00
        config = self._config(csv_path, "2025-12-31 00:00+00:00")

        test_df = entsoe_test_data_loader(config)

        assert len(test_df) == 11  # only the rows that exist after the cutoff

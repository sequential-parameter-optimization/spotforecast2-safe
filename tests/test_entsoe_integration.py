# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import MagicMock, patch

import pandas as pd

from spotforecast2_safe.downloader.entsoe import download_new_data, merge_build_manual


def test_merge_build_manual_creates_merged_file(tmp_path):
    # Setup: create raw directory and CSVs
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    df1 = pd.DataFrame(
        {"Time (UTC)": ["2026-01-01 00:00", "2026-01-01 01:00"], "Actual Load": [1, 2]}
    )
    df2 = pd.DataFrame(
        {"Time (UTC)": ["2026-01-01 01:00", "2026-01-01 02:00"], "Actual Load": [2, 3]}
    )
    df1.to_csv(raw_dir / "a.csv", index=False)
    df2.to_csv(raw_dir / "b.csv", index=False)
    with patch(
        "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
    ):
        merge_build_manual(output_file="merged.csv")
    merged = pd.read_csv(interim_dir / "merged.csv", index_col=0, parse_dates=True)
    assert len(merged) == 3
    assert "Actual Load" in merged.columns


def test_merge_build_manual_no_raw_dir(tmp_path, caplog):
    # No raw dir should log a warning and do nothing
    with patch(
        "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
    ):
        merge_build_manual(output_file="should_not_exist.csv")
    assert not (tmp_path / "interim" / "should_not_exist.csv").exists()


def test_download_new_data_success(tmp_path):
    # Patch get_data_home, fetch_data, and sys.modules["entsoe"]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    with (
        patch(
            "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
        ),
        patch("spotforecast2_safe.downloader.entsoe.fetch_data") as mock_fetch,
    ):
        # Setup fetch_data to return a DataFrame with a recent index
        dates = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=dates)
        # Patch sys.modules to inject a mock entsoe module
        import sys

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        mock_client = mock_client_class.return_value
        mock_df = pd.DataFrame(
            {"Actual Load": [123]}, index=[dates[-1] + pd.Timedelta(hours=1)]
        )
        mock_client.query_load_and_forecast.return_value = mock_df
        sys.modules["entsoe"] = mock_entsoe_mod
        download_new_data(api_key="fake_key", force=True)
        # Check that a file was created in raw
        files = list(raw_dir.glob("entsoe_load_*.csv"))
        assert len(files) == 1
        # Check merged file exists
        merged = pd.read_csv(interim_dir / "energy_load.csv")
        assert "Actual Load" in merged.columns


def _write_past_future_raw(raw_dir):
    """Write a raw CSV with one past and one future row.

    The future row mimics ENTSO-E's day-ahead publication: `Forecasted Load`
    is populated while `Actual Load` is still blank. Stamps are placed two
    days either side of today's UTC midnight so the past/future split is
    unambiguous regardless of the exact moment the test runs.
    """
    day = pd.Timestamp.now(tz="UTC").floor("D")
    past = day - pd.Timedelta(days=2)
    future = day + pd.Timedelta(days=2)
    df = pd.DataFrame(
        {
            "Time (UTC)": [past.isoformat(), future.isoformat()],
            "Actual Load": [42000.0, "-"],          # future actual not yet published
            "Forecasted Load": [41000.0, 39000.0],  # day-ahead forecast covers future
        }
    )
    df.to_csv(raw_dir / "load.csv", index=False)
    return past, future


def test_merge_build_manual_drops_future_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    past, future = _write_past_future_raw(raw_dir)
    with patch(
        "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
    ):
        merge_build_manual(output_file="merged.csv")
    merged = pd.read_csv(
        tmp_path / "interim" / "merged.csv", index_col=0, parse_dates=True
    )
    merged.index = pd.to_datetime(merged.index, utc=True)
    assert past in merged.index
    assert future not in merged.index  # default: future row dropped


def test_merge_build_manual_keeps_future_when_flag(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    past, future = _write_past_future_raw(raw_dir)
    with patch(
        "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
    ):
        merge_build_manual(output_file="merged.csv", keep_forecast_future=True)
    merged = pd.read_csv(
        tmp_path / "interim" / "merged.csv", index_col=0, parse_dates=True
    )
    merged.index = pd.to_datetime(merged.index, utc=True)
    # Future row retained, with the day-ahead forecast but no actual.
    assert future in merged.index
    assert merged.loc[future, "Forecasted Load"] == 39000.0
    assert pd.isna(merged.loc[future, "Actual Load"])


def test_merge_build_manual_index_is_utc(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_past_future_raw(raw_dir)
    with patch(
        "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
    ):
        merge_build_manual(output_file="merged.csv", keep_forecast_future=True)
    merged = pd.read_csv(
        tmp_path / "interim" / "merged.csv", index_col=0, parse_dates=True
    )
    merged.index = pd.to_datetime(merged.index)
    assert merged.index.tz is not None
    assert str(merged.index.tz) == "UTC"


def test_download_new_data_keep_forecast_future_retains_forecast(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (tmp_path / "interim").mkdir()
    day = pd.Timestamp.now(tz="UTC").floor("D")
    future = day + pd.Timedelta(days=1)
    with (
        patch(
            "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
        ),
        patch("spotforecast2_safe.downloader.entsoe.fetch_data") as mock_fetch,
    ):
        mock_fetch.side_effect = FileNotFoundError("no prior data")
        import sys

        mock_entsoe_mod = MagicMock()
        mock_client = mock_entsoe_mod.EntsoePandasClient.return_value
        # Client returns a frame whose forecast extends into the future day.
        mock_df = pd.DataFrame(
            {"Actual Load": [42000.0, None], "Forecasted Load": [41000.0, 39000.0]},
            index=[day - pd.Timedelta(hours=1), future],
        )
        mock_client.query_load_and_forecast.return_value = mock_df
        sys.modules["entsoe"] = mock_entsoe_mod
        download_new_data(
            api_key="fake_key",
            start="202601010000",
            end="202601050000",
            force=True,
            keep_forecast_future=True,
        )
    merged = pd.read_csv(
        tmp_path / "interim" / "energy_load.csv", index_col=0, parse_dates=True
    )
    merged.index = pd.to_datetime(merged.index, utc=True)
    assert future in merged.index
    assert merged.loc[future, "Forecasted Load"] == 39000.0


def test_download_new_data_resume_skips_future_rows(tmp_path):
    """start=None must resume from the last observed (<= now) row, not a future
    forecast row sitting in a keep_forecast_future interim."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (tmp_path / "interim").mkdir()
    now = pd.Timestamp.now(tz="UTC")
    last_observed = now.floor("h") - pd.Timedelta(hours=3)
    future = now.floor("h") + pd.Timedelta(days=1)
    with (
        patch(
            "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
        ),
        patch("spotforecast2_safe.downloader.entsoe.fetch_data") as mock_fetch,
    ):
        # Interim already carries a future forecast row past `now`.
        mock_fetch.return_value = pd.DataFrame(
            {"Actual Load": [1.0, None]}, index=[last_observed, future]
        )
        import sys

        mock_entsoe_mod = MagicMock()
        mock_client = mock_entsoe_mod.EntsoePandasClient.return_value
        mock_client.query_load_and_forecast.return_value = pd.DataFrame(
            {"Actual Load": [2.0]}, index=[last_observed + pd.Timedelta(hours=1)]
        )
        sys.modules["entsoe"] = mock_entsoe_mod
        download_new_data(api_key="fake_key", force=True, keep_forecast_future=True)
        # Resumed from last_observed + 1h, NOT from the future row.
        _, kwargs = mock_client.query_load_and_forecast.call_args
        assert kwargs["start"] == last_observed + pd.Timedelta(hours=1)


def test_download_new_data_cooldown_skips(tmp_path):
    # Patch get_data_home, fetch_data, and sys.modules["entsoe"]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    with (
        patch(
            "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
        ),
        patch("spotforecast2_safe.downloader.entsoe.fetch_data") as mock_fetch,
    ):
        now = pd.Timestamp.now(tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=[now - pd.Timedelta(hours=2)])
        import sys

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        sys.modules["entsoe"] = mock_entsoe_mod
        download_new_data(api_key="fake_key", force=False)
        # Should not call the client
        mock_client_class.assert_not_called()

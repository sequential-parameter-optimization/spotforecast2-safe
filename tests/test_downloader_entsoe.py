# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from spotforecast2_safe.downloader.entsoe import download_new_data, merge_build_manual

# Mock entsoe before importing our module
mock_entsoe = MagicMock()
sys.modules["entsoe"] = mock_entsoe


class TestEntsoeDownloader(unittest.TestCase):
    """Tests for the ENTSO-E downloader."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw"
        self.raw_dir.mkdir()
        self.interim_dir = self.test_dir / "interim"
        self.interim_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    def test_merge_build_manual(self, mock_get_home):
        """Test merging raw CSV files."""
        mock_get_home.return_value = self.test_dir

        # Create some dummy raw files
        df1 = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01 00:00", "2026-01-01 01:00"],
                "Actual": [100, 110],
            }
        )
        df2 = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01 01:00", "2026-01-01 02:00"],
                "Actual": [110, 120],
            }
        )

        df1.to_csv(self.raw_dir / "file1.csv", index=False)
        df2.to_csv(self.raw_dir / "file2.csv", index=False)

        merge_build_manual(output_file="test_merged.csv")

        # Verify output
        output_path = self.interim_dir / "test_merged.csv"
        self.assertTrue(output_path.exists())

        merged_df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        self.assertEqual(len(merged_df), 3)  # Overlap handled
        self.assertEqual(merged_df.index[0], pd.Timestamp("2026-01-01 00:00:00+0000"))

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_download_new_data_success(self, mock_fetch, mock_get_home):
        """Test successful data download."""
        mock_get_home.return_value = self.test_dir

        # Setup mock fetch_data for start date calculation
        dates = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=dates)

        # Patch sys.modules to inject a mock entsoe module
        import sys
        from unittest.mock import MagicMock

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        mock_client = mock_client_class.return_value
        mock_df = pd.DataFrame(
            {"Actual": [123]}, index=[dates[-1] + pd.Timedelta(hours=1)]
        )
        mock_client.query_load_and_forecast.return_value = mock_df
        sys.modules["entsoe"] = mock_entsoe_mod

        download_new_data(api_key="fake_key", force=True)

        # Verify client call
        mock_client_class.assert_called_once_with(api_key="fake_key")

        # Verify file creation in raw
        raw_files = list(self.raw_dir.glob("entsoe_load_*.csv"))
        self.assertEqual(len(raw_files), 1)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_download_new_data_cooldown(self, mock_fetch, mock_get_home):
        """Test that download is skipped if too recent."""
        mock_get_home.return_value = self.test_dir

        # Last index is very recent
        now = pd.Timestamp.now(tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=[now - pd.Timedelta(hours=2)])

        import sys
        from unittest.mock import MagicMock

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        sys.modules["entsoe"] = mock_entsoe_mod

        download_new_data(api_key="fake_key", force=False)

        mock_client_class.assert_not_called()


class TestDownloadNewDataResumeFallback(unittest.TestCase):
    """B4 regression: narrow the "no prior data" fallback.

    ``download_new_data`` used to catch every ``Exception`` from
    ``fetch_data()`` and silently default to "seven days ago". That
    masks real bugs (import failures, schema drift, permission errors)
    behind the same log line as a first-run bootstrap. We now only
    absorb ``FileNotFoundError``, ``ValueError`` (the current "no
    filename" signal from ``fetch_data``), and ``IndexError`` (empty
    frame). Anything else must propagate.
    """

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        (self.test_dir / "raw").mkdir()
        (self.test_dir / "interim").mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_filenotfound_triggers_7day_fallback(self, mock_fetch, mock_get_home):
        """FileNotFoundError is the canonical "no prior data" signal."""
        mock_get_home.return_value = self.test_dir
        mock_fetch.side_effect = FileNotFoundError("no interim file yet")

        mock_entsoe_mod = MagicMock()
        mock_client = mock_entsoe_mod.EntsoePandasClient.return_value
        mock_client.query_load_and_forecast.return_value = pd.DataFrame(
            {"Actual": [1.0]},
            index=[pd.Timestamp.now(tz="UTC")],
        )
        sys.modules["entsoe"] = mock_entsoe_mod

        download_new_data(api_key="fake_key", force=True)

        # Client was called; fallback resolved.
        mock_entsoe_mod.EntsoePandasClient.assert_called_once()

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_indexerror_on_empty_frame_triggers_fallback(
        self, mock_fetch, mock_get_home
    ):
        """An empty DataFrame raises IndexError on ``index[-1]``; absorb it."""
        mock_get_home.return_value = self.test_dir
        mock_fetch.return_value = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

        mock_entsoe_mod = MagicMock()
        mock_client = mock_entsoe_mod.EntsoePandasClient.return_value
        mock_client.query_load_and_forecast.return_value = pd.DataFrame(
            {"Actual": [1.0]},
            index=[pd.Timestamp.now(tz="UTC")],
        )
        sys.modules["entsoe"] = mock_entsoe_mod

        download_new_data(api_key="fake_key", force=True)

        mock_entsoe_mod.EntsoePandasClient.assert_called_once()

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_unexpected_exception_propagates(self, mock_fetch, mock_get_home):
        """Anything outside the narrowed set must NOT be silently absorbed.

        A PermissionError here used to be swallowed into the 7-day
        default, masking a real filesystem bug. It should surface.
        """
        mock_get_home.return_value = self.test_dir
        mock_fetch.side_effect = PermissionError("cannot read interim dir")

        mock_entsoe_mod = MagicMock()
        sys.modules["entsoe"] = mock_entsoe_mod

        with self.assertRaises(PermissionError):
            download_new_data(api_key="fake_key", force=True)

        mock_entsoe_mod.EntsoePandasClient.assert_not_called()


class TestMergeBuildManualMalformedCSV(unittest.TestCase):
    """A malformed CSV in the raw dir is logged and skipped, not fatal."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw"
        self.raw_dir.mkdir()
        self.interim_dir = self.test_dir / "interim"
        self.interim_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    def test_merge_build_manual_skips_malformed_csv(self, mock_get_home):
        """Good CSVs merge; bad ones get an ERROR log and are skipped."""
        mock_get_home.return_value = self.test_dir

        good_df = pd.DataFrame(
            {
                "Time (UTC)": [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                ],
                "Actual": [100, 110, 120],
            }
        )
        good_df.to_csv(self.raw_dir / "good.csv", index=False)
        # bad.csv parses as CSV but its time column is not parseable as a
        # datetime, which trips pd.to_datetime inside the merge loop.
        (self.raw_dir / "bad.csv").write_text(
            "Time (UTC),Actual\nnope,1\nstill-not-a-date,2\n"
        )

        with self.assertLogs(
            "spotforecast2_safe.downloader.entsoe", level="ERROR"
        ) as cm:
            merge_build_manual(output_file="merged.csv")

        output_path = self.interim_dir / "merged.csv"
        self.assertTrue(output_path.exists())
        merged_df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        self.assertEqual(len(merged_df), 3)
        self.assertTrue(
            any("Failed to process raw file" in msg for msg in cm.output),
            f"Expected ERROR log mentioning the bad file; got {cm.output!r}",
        )


class TestDownloadNewDataRetrySuccess(unittest.TestCase):
    """The retry loop tolerates transient failures before eventual success."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        (self.test_dir / "raw").mkdir()
        (self.test_dir / "interim").mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.downloader.entsoe.time.sleep", lambda _s: None)
    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_download_succeeds_after_two_transient_failures(
        self, mock_fetch, mock_get_home
    ):
        """Two RuntimeErrors then a DataFrame: function should complete."""
        mock_get_home.return_value = self.test_dir
        mock_fetch.side_effect = FileNotFoundError("no prior data")

        mock_entsoe_mod = MagicMock()
        mock_client = mock_entsoe_mod.EntsoePandasClient.return_value
        valid_df = pd.DataFrame(
            {"Actual": [1.0]}, index=[pd.Timestamp("2023-01-01", tz="UTC")]
        )
        mock_client.query_load_and_forecast.side_effect = [
            RuntimeError("transient 1"),
            RuntimeError("transient 2"),
            valid_df,
        ]
        sys.modules["entsoe"] = mock_entsoe_mod

        with self.assertLogs(
            "spotforecast2_safe.downloader.entsoe", level="WARNING"
        ) as cm:
            download_new_data(
                api_key="fake_key",
                country_code="DE",
                start="202301010000",
                end="202301020000",
                force=True,
            )

        self.assertEqual(mock_client.query_load_and_forecast.call_count, 3)
        warning_count = sum(
            1 for msg in cm.output if "Download failed" in msg
        )
        self.assertEqual(warning_count, 2)
        raw_files = list((self.test_dir / "raw").glob("entsoe_load_*.csv"))
        self.assertEqual(len(raw_files), 1)


class TestDownloadNewDataFailSafe(unittest.TestCase):
    """Fail-safe contracts: NaT input rejection and raise-on-persistent-failure."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        (self.test_dir / "raw").mkdir()
        (self.test_dir / "interim").mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.downloader.entsoe.time.sleep", lambda _s: None)
    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_raises_runtimeerror_after_persistent_failure(
        self, mock_fetch, mock_get_home
    ):
        """After exhausting the retry budget, raise RuntimeError instead of returning silently."""
        mock_get_home.return_value = self.test_dir
        mock_fetch.side_effect = FileNotFoundError("no prior data")

        mock_entsoe_mod = MagicMock()
        mock_client = mock_entsoe_mod.EntsoePandasClient.return_value
        mock_client.query_load_and_forecast.side_effect = RuntimeError("api down")
        sys.modules["entsoe"] = mock_entsoe_mod

        with self.assertRaises(RuntimeError) as ctx:
            download_new_data(api_key="fake_key", force=True)

        self.assertIn("5 attempts", str(ctx.exception))
        # All 5 attempts were made
        self.assertEqual(mock_client.query_load_and_forecast.call_count, 5)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    def test_invalid_start_raises_value_error(self, mock_get_home):
        """A start string that does not parse as a timestamp raises ValueError."""
        mock_get_home.return_value = self.test_dir

        with self.assertRaises(ValueError) as ctx:
            download_new_data(
                api_key="fake_key",
                start="not a date",
                end="202301020000",
                force=True,
            )
        self.assertIn("start=", str(ctx.exception))

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    def test_invalid_end_raises_value_error(self, mock_get_home):
        """An end string that does not parse as a timestamp raises ValueError."""
        mock_get_home.return_value = self.test_dir

        with self.assertRaises(ValueError) as ctx:
            download_new_data(
                api_key="fake_key",
                start="202301010000",
                end="not a date",
                force=True,
            )
        self.assertIn("end=", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
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

        # Verify client call includes the default timeout
        mock_client_class.assert_called_once_with(api_key="fake_key", timeout=60.0)

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
        warning_count = sum(1 for msg in cm.output if "Download failed" in msg)
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


class TestMergeRawFileOrdering(unittest.TestCase):
    """Regression: a stale partial pull must not clobber newer complete data.

    ENTSO-E publishes "Actual Load" with a lag, so a raw pull made before a
    day's actuals were published stores NaN rows for that day. Previously
    ``merge_build_manual`` concatenated raw files in arbitrary filesystem
    glob order and deduplicated whole rows with ``keep="last"``; a stale
    partial pull globbed after a newer complete one masked already
    downloaded values with NaN in the interim file (observed for DE on
    2026-06-02). The merge now orders files oldest-mtime-first and keeps,
    per cell, the newest non-missing value.
    """

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw"
        self.raw_dir.mkdir()
        self.interim_dir = self.test_dir / "interim"
        self.interim_dir.mkdir()
        self.times = [
            "2026-06-02 00:00",
            "2026-06-02 01:00",
            "2026-06-02 02:00",
        ]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_raw(self, name, values, mtime):
        df = pd.DataFrame(
            {"Time (UTC)": self.times[: len(values)], "Actual Load": values}
        )
        path = self.raw_dir / name
        df.to_csv(path, index=False)
        os.utime(path, (mtime, mtime))
        return path

    def _merged(self):
        merge_build_manual(output_file="merged.csv")
        merged = pd.read_csv(
            self.interim_dir / "merged.csv", index_col=0, parse_dates=True
        )
        merged.index = pd.to_datetime(merged.index, utc=True)
        return merged

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    def test_stale_nan_pull_does_not_clobber_complete_pull(self, mock_get_home):
        """The 2026-06-02 incident shape: stale all-NaN file, newer complete one.

        The stale file's name sorts after the complete one, so under the old
        glob-order ``keep="last"`` rule its NaN rows used to win.
        """
        mock_get_home.return_value = self.test_dir
        self._write_raw("a_complete.csv", [100.0, 110.0, 120.0], mtime=2_000_000)
        self._write_raw("z_stale.csv", [None, None, None], mtime=1_000_000)

        merged = self._merged()

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged["Actual Load"].isna().sum(), 0)
        self.assertListEqual(merged["Actual Load"].tolist(), [100.0, 110.0, 120.0])

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    def test_newer_values_revise_but_newer_nan_does_not_erase(self, mock_get_home):
        """Per-cell rule: a newer pull revises values, its NaN cells do not erase."""
        mock_get_home.return_value = self.test_dir
        self._write_raw("older.csv", [100.0, 110.0], mtime=1_000_000)
        self._write_raw("newer.csv", [None, 115.0, 120.0], mtime=2_000_000)

        merged = self._merged()

        self.assertListEqual(merged["Actual Load"].tolist(), [100.0, 115.0, 120.0])


class TestDownloadNewDataActualLoadBackfill(unittest.TestCase):
    """Incremental downloads re-fetch hours whose "Actual Load" is missing.

    The interim file carries rows for hours whose day-ahead forecast is
    already published but whose actuals are not (late publication /
    transparency-platform outages). Resuming strictly from the last row
    would never re-fetch those hours; the fetch window now restarts at the
    first missing actual, bounded by ``_MAX_BACKFILL_DAYS``.
    """

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        (self.test_dir / "raw").mkdir()
        (self.test_dir / "interim").mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _run_download(self, mock_fetch, mock_get_home, frame):
        mock_get_home.return_value = self.test_dir
        mock_fetch.return_value = frame

        mock_entsoe_mod = MagicMock()
        mock_client = mock_entsoe_mod.EntsoePandasClient.return_value
        mock_client.query_load_and_forecast.return_value = pd.DataFrame(
            {"Actual Load": [1.0]}, index=[frame.index[-1] + pd.Timedelta(hours=1)]
        )
        sys.modules["entsoe"] = mock_entsoe_mod

        end = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y%m%d%H00")
        download_new_data(api_key="fake_key", end=end, force=True)
        return mock_client.query_load_and_forecast.call_args.kwargs["start"]

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_resume_restarts_at_first_missing_actual(self, mock_fetch, mock_get_home):
        """NaN tail in "Actual Load" pulls the fetch start back to its first hour."""
        idx = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1),
            periods=72,
            freq="h",
        )
        actual = [float(i) for i in range(42)] + [None] * 30
        frame = pd.DataFrame(
            {"Actual Load": actual, "Forecasted Load": [1.0] * 72}, index=idx
        )

        start = self._run_download(mock_fetch, mock_get_home, frame)

        # First hour whose actuals are missing, not last_row + 1h.
        self.assertEqual(start, idx[42])

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_backfill_is_bounded_by_max_backfill_days(self, mock_fetch, mock_get_home):
        """An all-NaN actuals column must not trigger an unbounded re-pull."""
        idx = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1),
            periods=24 * 30,
            freq="h",
        )
        frame = pd.DataFrame(
            {"Actual Load": [None] * len(idx), "Forecasted Load": [1.0] * len(idx)},
            index=idx,
        )

        start = self._run_download(mock_fetch, mock_get_home, frame)

        expected_floor = idx[-1] + pd.Timedelta(hours=1) - pd.Timedelta(days=7)
        self.assertEqual(start, expected_floor)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_complete_actuals_resume_unchanged(self, mock_fetch, mock_get_home):
        """With no missing actuals the resume point stays last_row + 1h."""
        idx = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1),
            periods=72,
            freq="h",
        )
        frame = pd.DataFrame(
            {"Actual Load": [1.0] * 72, "Forecasted Load": [1.0] * 72}, index=idx
        )

        start = self._run_download(mock_fetch, mock_get_home, frame)

        self.assertEqual(start, idx[-1] + pd.Timedelta(hours=1))


class TestDownloadTimeoutForwarding(unittest.TestCase):
    """Feature B: timeout is forwarded to the client and triggers retry on Timeout."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        (self.test_dir / "raw").mkdir()
        (self.test_dir / "interim").mkdir()
        import spotforecast2_safe.downloader.entsoe as _entsoe_mod

        self._entsoe_mod = _entsoe_mod
        self._orig_flag = _entsoe_mod._TIMEOUT_WARNING_ISSUED
        _entsoe_mod._TIMEOUT_WARNING_ISSUED = False

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self._entsoe_mod._TIMEOUT_WARNING_ISSUED = self._orig_flag

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_explicit_timeout_none_forwarded(self, mock_fetch, mock_get_home):
        """timeout=None is passed to the client constructor."""
        mock_get_home.return_value = self.test_dir

        dates = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=dates)

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        mock_client = mock_client_class.return_value
        mock_df = pd.DataFrame(
            {"Actual": [1.0]}, index=[dates[-1] + pd.Timedelta(hours=1)]
        )
        mock_client.query_load_and_forecast.return_value = mock_df
        sys.modules["entsoe"] = mock_entsoe_mod

        download_new_data(api_key="fake_key", force=True, timeout=None)

        # With timeout=None _make_client falls through to api_key-only construction
        mock_client_class.assert_called_once_with(api_key="fake_key")

    @patch("spotforecast2_safe.downloader.entsoe.time.sleep", lambda _s: None)
    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_requests_timeout_triggers_retry_then_runtimeerror(
        self, mock_fetch, mock_get_home
    ):
        """A requests.exceptions.Timeout on the query method triggers the retry
        loop and raises RuntimeError after exhausting the budget."""
        import requests

        mock_get_home.return_value = self.test_dir
        mock_fetch.side_effect = FileNotFoundError("no prior data")

        mock_entsoe_mod = MagicMock()
        mock_client = mock_entsoe_mod.EntsoePandasClient.return_value
        mock_client.query_load_and_forecast.side_effect = requests.exceptions.Timeout(
            "read timeout"
        )
        sys.modules["entsoe"] = mock_entsoe_mod

        with self.assertRaises(RuntimeError) as ctx:
            download_new_data(api_key="fake_key", force=True, timeout=60.0)

        self.assertIn("5 attempts", str(ctx.exception))
        self.assertEqual(mock_client.query_load_and_forecast.call_count, 5)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_typeerror_fallback_logs_warning_and_constructs_without_timeout(
        self, mock_fetch, mock_get_home
    ):
        """When EntsoePandasClient rejects 'timeout', _make_client falls back
        to api_key-only construction and logs a warning."""
        mock_get_home.return_value = self.test_dir
        dates = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=dates)

        # _TIMEOUT_WARNING_ISSUED is already reset to False by setUp.

        class StrictClient:
            """Rejects the 'timeout' keyword to simulate old entsoe-py."""

            def __init__(self, api_key):
                self.api_key = api_key

            def query_load_and_forecast(self, country_code, start, end):
                return pd.DataFrame(
                    {"Actual": [1.0]},
                    index=[dates[-1] + pd.Timedelta(hours=1)],
                )

        mock_entsoe_mod = MagicMock()
        mock_entsoe_mod.EntsoePandasClient = StrictClient
        sys.modules["entsoe"] = mock_entsoe_mod

        with self.assertLogs(
            "spotforecast2_safe.downloader.entsoe", level="WARNING"
        ) as cm:
            download_new_data(api_key="fake_key", force=True, timeout=30.0)

        self.assertTrue(
            any("entsoe-py does not support 'timeout'" in msg for msg in cm.output),
            f"Expected timeout warning; got: {cm.output!r}",
        )


if __name__ == "__main__":
    unittest.main()

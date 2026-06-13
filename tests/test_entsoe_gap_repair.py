# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ENTSO-E gap detection, gap repair, and unavailability handling.

Covers the failure mode shown by the June 1st-2nd 2026 incident: ENTSO-E
publishes nothing for an interval, the interim file ends up with an
interior hole, and the pipeline must be able to repair it from already
downloaded raw files or a targeted re-download -- without ever inventing
values. Dates are shifted to January 2026 (safely in the past) so the
default future-row filter in ``merge_build_manual`` never interferes.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spotforecast2_safe.downloader.entsoe import (
    download_new_data,
    find_missing_intervals,
    merge_build_manual,
    repair_data_gaps,
)


def _write_raw(raw_dir, name, start, periods, value=50000.0):
    """Write an hourly raw CSV covering ``periods`` hours from ``start``."""
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    pd.DataFrame({"Time (UTC)": idx.astype(str), "Actual Load": value}).to_csv(
        raw_dir / name, index=False
    )
    return idx


def _mock_entsoe(response=None, side_effect=None):
    """Inject a mocked ``entsoe`` module; return (module, client) mocks."""
    mod = MagicMock()
    client = mod.EntsoePandasClient.return_value
    if side_effect is not None:
        client.query_load_and_forecast.side_effect = side_effect
    else:
        client.query_load_and_forecast.return_value = response
    sys.modules["entsoe"] = mod
    return mod, client


@pytest.fixture
def gap_env(tmp_path):
    """Data home with raw files for Jan 1 and Jan 3 but nothing for Jan 2.

    Merging therefore leaves an interior 24h hole on 2026-01-02 -- the
    June-1st/2nd pattern from the field incident.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (tmp_path / "interim").mkdir()
    _write_raw(raw_dir, "jan01.csv", "2026-01-01 00:00", 24)
    _write_raw(raw_dir, "jan03.csv", "2026-01-03 00:00", 24)
    with patch(
        "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
    ):
        yield tmp_path


GAP = (
    pd.Timestamp("2026-01-02 00:00", tz="UTC"),
    pd.Timestamp("2026-01-02 23:00", tz="UTC"),
)


class TestFindMissingIntervals:
    """Unit tests for the index-completeness gap finder."""

    def test_complete_index_has_no_gaps(self):
        idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
        assert find_missing_intervals(idx) == []

    def test_detects_single_interior_gap(self):
        full = pd.date_range("2026-01-01", periods=72, freq="h", tz="UTC")
        gappy = full.delete(list(range(24, 48)))  # Jan 2nd missing
        assert find_missing_intervals(gappy) == [GAP]

    def test_detects_multiple_gaps_and_15min_resolution(self):
        full = pd.date_range("2026-01-01", periods=96 * 3, freq="15min", tz="UTC")
        gappy = full.delete(list(range(8, 12)) + list(range(200, 204)))
        gaps = find_missing_intervals(gappy)
        assert gaps == [(full[8], full[11]), (full[200], full[203])]

    def test_short_index_yields_no_gaps(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
            ]
        )
        assert find_missing_intervals(idx) == []

    def test_unsorted_index_raises(self):
        idx = pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-03"], utc=True)
        with pytest.raises(ValueError, match="sorted"):
            find_missing_intervals(idx)


class TestEmptyEntsoeResponse:
    """ENTSO-E is reachable but has no data for the requested interval."""

    def test_empty_dataframe_writes_nothing_and_warns(self, gap_env, caplog):
        _, client = _mock_entsoe(response=pd.DataFrame())
        with caplog.at_level(
            "WARNING", logger="spotforecast2_safe.downloader.entsoe"
        ):
            download_new_data(
                api_key="k", start="202601020000", end="202601030000", force=True
            )
        assert client.query_load_and_forecast.call_count == 1
        assert not list((gap_env / "raw").glob("entsoe_load_*.csv"))
        assert any("returned no data" in m for m in caplog.messages)

    def test_all_nan_rows_write_nothing_and_warn(self, gap_env, caplog):
        idx = pd.date_range("2026-01-02", periods=24, freq="h", tz="UTC")
        all_nan = pd.DataFrame(
            {
                "Actual Load": [float("nan")] * 24,
                "Forecasted Load": [float("nan")] * 24,
            },
            index=idx,
        )
        _mock_entsoe(response=all_nan)
        with caplog.at_level(
            "WARNING", logger="spotforecast2_safe.downloader.entsoe"
        ):
            download_new_data(
                api_key="k", start="202601020000", end="202601030000", force=True
            )
        assert not list((gap_env / "raw").glob("entsoe_load_*.csv"))
        assert any("returned no data" in m for m in caplog.messages)


class TestRepairDataGaps:
    """repair_data_gaps: disk first, targeted download second, never invent."""

    def test_repair_from_raw_files_already_on_disk(self, gap_env):
        """A stale interim file is healed by re-merging raw coverage; no network."""
        # Raw coverage of the hole arrived later (e.g. manual export).
        _write_raw(gap_env / "raw", "jan02.csv", "2026-01-02 00:00", 24)
        # Interim still carries the hole: built only from Jan 1 + Jan 3.
        idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC").append(
            pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
        )
        pd.DataFrame({"Actual Load": 50000.0}, index=idx).rename_axis(
            "Time (UTC)"
        ).to_csv(gap_env / "interim" / "energy_load.csv")

        _, client = _mock_entsoe(response=pd.DataFrame())

        remaining = repair_data_gaps(api_key="k")

        assert remaining == []
        client.query_load_and_forecast.assert_not_called()
        merged = pd.read_csv(gap_env / "interim" / "energy_load.csv", index_col=0)
        assert pd.Timestamp("2026-01-02 12:00", tz="UTC") in pd.to_datetime(
            merged.index, utc=True
        )

    def test_targeted_download_fetches_exactly_the_missing_range(self, gap_env):
        """A gap with no raw coverage triggers one bounded ENTSO-E query."""
        gap_data = pd.DataFrame(
            {"Actual Load": 50000.0},
            index=pd.date_range("2026-01-02 00:00", periods=24, freq="h", tz="UTC"),
        )
        _, client = _mock_entsoe(response=gap_data)

        remaining = repair_data_gaps(api_key="k", country_code="DE")

        assert remaining == []
        client.query_load_and_forecast.assert_called_once()
        _, kwargs = client.query_load_and_forecast.call_args
        assert kwargs["country_code"] == "DE"
        # Window = gap padded by 1h on each side.
        assert kwargs["start"] == pd.Timestamp("2026-01-01 23:00", tz="UTC")
        assert kwargs["end"] == pd.Timestamp("2026-01-03 00:00", tz="UTC")
        merged = pd.read_csv(gap_env / "interim" / "energy_load.csv", index_col=0)
        assert pd.Timestamp("2026-01-02 12:00", tz="UTC") in pd.to_datetime(
            merged.index, utc=True
        )

    def test_unfillable_gap_raises_by_default(self, gap_env):
        """ENTSO-E has nothing for the interval: fail loudly, never invent."""
        _mock_entsoe(response=pd.DataFrame())
        with pytest.raises(ValueError, match="could not be repaired"):
            repair_data_gaps(api_key="k")

    def test_use_existing_returns_remaining_gaps_and_warns(self, gap_env, caplog):
        _mock_entsoe(response=pd.DataFrame())
        with caplog.at_level(
            "WARNING", logger="spotforecast2_safe.downloader.entsoe"
        ):
            remaining = repair_data_gaps(api_key="k", on_unavailable="use_existing")
        assert remaining == [GAP]
        assert any("unrepaired gap" in m for m in caplog.messages)

    def test_no_gaps_is_a_clean_no_op(self, gap_env):
        _write_raw(gap_env / "raw", "jan02.csv", "2026-01-02 00:00", 24)
        _, client = _mock_entsoe(response=pd.DataFrame())
        assert repair_data_gaps(api_key="k") == []
        client.query_load_and_forecast.assert_not_called()

    def test_invalid_on_unavailable_rejected(self, gap_env):
        with pytest.raises(ValueError, match="on_unavailable"):
            repair_data_gaps(api_key="k", on_unavailable="banana")

    def test_entsoe_down_keeps_gap_and_raises_by_default(self, gap_env):
        """Persistent network failure during targeted download: gap survives."""
        _mock_entsoe(side_effect=RuntimeError("api down"))
        with patch(
            "spotforecast2_safe.downloader.entsoe.time.sleep", lambda _s: None
        ):
            with pytest.raises(ValueError, match="could not be repaired"):
                repair_data_gaps(api_key="k")


class TestCooldownSemantics:
    """force=False means 'recent successful download', never 'small window'."""

    def test_force_false_backfill_of_gap_proceeds_despite_recent_download(
        self, gap_env
    ):
        """The June-1st regression: force=False must not skip a gap backfill."""
        merge_build_manual()  # interim now carries the Jan 2 hole
        # A freshly written raw file marks a recent successful download.
        (gap_env / "raw" / "entsoe_load_202601030000_202601040000.csv").write_text(
            "Time (UTC),Actual Load\n2026-01-03 00:00,1.0\n"
        )

        gap_data = pd.DataFrame(
            {"Actual Load": 50000.0},
            index=pd.date_range("2026-01-02 00:00", periods=24, freq="h", tz="UTC"),
        )
        _, client = _mock_entsoe(response=gap_data)

        # Sub-24h window inside the hole, force=False: the old window-width
        # cooldown silently skipped exactly this call.
        download_new_data(
            api_key="k", start="202601020000", end="202601021200", force=False
        )

        client.query_load_and_forecast.assert_called_once()

    def test_force_false_skips_when_recent_and_no_gap(self, gap_env):
        # Heal the hole first so no gap bypass applies.
        _write_raw(gap_env / "raw", "jan02.csv", "2026-01-02 00:00", 24)
        merge_build_manual()
        (gap_env / "raw" / "entsoe_load_202601030000_202601040000.csv").write_text(
            "Time (UTC),Actual Load\n2026-01-03 00:00,1.0\n"
        )

        _, client = _mock_entsoe(response=pd.DataFrame())
        download_new_data(
            api_key="k", start="202601040000", end="202601050000", force=False
        )
        client.query_load_and_forecast.assert_not_called()

    def test_force_false_proceeds_when_last_download_is_stale(self, gap_env):
        _write_raw(gap_env / "raw", "jan02.csv", "2026-01-02 00:00", 24)
        merge_build_manual()
        stale = gap_env / "raw" / "entsoe_load_202601030000_202601040000.csv"
        stale.write_text("Time (UTC),Actual Load\n2026-01-03 00:00,1.0\n")
        two_days_ago = pd.Timestamp.now(tz="UTC").timestamp() - 48 * 3600
        os.utime(stale, (two_days_ago, two_days_ago))

        new_data = pd.DataFrame(
            {"Actual Load": [1.0]},
            index=[pd.Timestamp("2026-01-04 00:00", tz="UTC")],
        )
        _, client = _mock_entsoe(response=new_data)
        download_new_data(
            api_key="k", start="202601040000", end="202601050000", force=False
        )
        client.query_load_and_forecast.assert_called_once()


class TestDownloadOnUnavailable:
    """Opt-in degradation when ENTSO-E is unreachable after all retries."""

    def test_use_existing_logs_and_returns_when_interim_exists(self, gap_env, caplog):
        merge_build_manual()  # interim now exists
        _mock_entsoe(side_effect=RuntimeError("api down"))
        with (
            patch(
                "spotforecast2_safe.downloader.entsoe.time.sleep", lambda _s: None
            ),
            caplog.at_level("WARNING", logger="spotforecast2_safe.downloader.entsoe"),
        ):
            download_new_data(
                api_key="k",
                start="202601040000",
                end="202601050000",
                force=True,
                on_unavailable="use_existing",
            )
        assert any(
            "Proceeding with the existing interim data" in m for m in caplog.messages
        )

    def test_use_existing_without_interim_still_raises(self, tmp_path):
        """Degradation needs data to degrade onto; with none, fail loudly."""
        (tmp_path / "raw").mkdir()
        (tmp_path / "interim").mkdir()
        _mock_entsoe(side_effect=RuntimeError("api down"))
        with (
            patch(
                "spotforecast2_safe.downloader.entsoe.get_data_home",
                return_value=tmp_path,
            ),
            patch(
                "spotforecast2_safe.downloader.entsoe.time.sleep", lambda _s: None
            ),
        ):
            with pytest.raises(RuntimeError, match="5 attempts"):
                download_new_data(
                    api_key="k",
                    start="202601040000",
                    end="202601050000",
                    force=True,
                    on_unavailable="use_existing",
                )

    def test_default_raise_preserved(self, gap_env):
        _mock_entsoe(side_effect=RuntimeError("api down"))
        with patch(
            "spotforecast2_safe.downloader.entsoe.time.sleep", lambda _s: None
        ):
            with pytest.raises(RuntimeError, match="5 attempts"):
                download_new_data(
                    api_key="k", start="202601040000", end="202601050000", force=True
                )

    def test_invalid_on_unavailable_rejected(self, gap_env):
        with pytest.raises(ValueError, match="on_unavailable"):
            download_new_data(api_key="k", on_unavailable="nope")


class TestEmptyWindowGuard:
    """end <= start: invalid input when explicit, clean no-op when derived."""

    def test_explicit_inverted_window_raises(self, gap_env):
        _mock_entsoe(response=pd.DataFrame())
        with pytest.raises(ValueError, match="not after"):
            download_new_data(
                api_key="k", start="202601050000", end="202601040000", force=True
            )

    def test_incremental_up_to_date_returns_quietly(self, tmp_path, caplog):
        """A resume landing past ``end`` is 'already up to date', not an error."""
        (tmp_path / "raw").mkdir()
        (tmp_path / "interim").mkdir()
        now = pd.Timestamp.now(tz="UTC").floor("h")
        idx = pd.date_range(now - pd.Timedelta(hours=5), now, freq="h")
        pd.DataFrame({"Actual Load": 1.0}, index=idx).rename_axis(
            "Time (UTC)"
        ).to_csv(tmp_path / "interim" / "energy_load.csv")
        _, client = _mock_entsoe(response=pd.DataFrame())
        with (
            patch(
                "spotforecast2_safe.downloader.entsoe.get_data_home",
                return_value=tmp_path,
            ),
            caplog.at_level("INFO", logger="spotforecast2_safe.downloader.entsoe"),
        ):
            download_new_data(api_key="k", force=True)
        client.query_load_and_forecast.assert_not_called()
        assert any("Nothing to download" in m for m in caplog.messages)

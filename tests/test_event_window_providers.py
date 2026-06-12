# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for EventWindowProvider, FootballMatchWindowProvider, and
EnergyCrisisWindowProvider."""

import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.exog_providers import (
    EXOG_PROVIDER_REGISTRY,
    EnergyCrisisWindowProvider,
    ExogProviderError,
    FootballMatchWindowProvider,
)


def _make_two_row_csv(tmp_path, intensity_a=0.5, intensity_b=1.0):
    """Write a two-row event CSV with overlapping windows to *tmp_path*."""
    csv = tmp_path / "test_events.csv"
    csv.write_text(
        "event,start_utc,end_utc,intensity\n"
        f"EVT_A,2024-01-01T00:00:00+00:00,2024-01-01T03:00:00+00:00,{intensity_a}\n"
        f"EVT_B,2024-01-01T02:00:00+00:00,2024-01-01T05:00:00+00:00,{intensity_b}\n"
    )
    return csv


# ---------------------------------------------------------------------------
# FootballMatchWindowProvider
# ---------------------------------------------------------------------------


class TestFootballMatchWindowProvider:
    def test_in_window_hour_is_one(self):
        """Hour inside a match window is 1.0."""
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        # EURO 2024 GER vs SCO window: 2024-06-14 18:00–22:00 UTC
        assert float(out.loc["2024-06-14T19:00:00Z", "football_match_window"]) == 1.0

    def test_outside_window_hour_is_zero(self):
        """Hour outside all match windows is 0.0."""
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        # Two hours before the window opens
        assert float(out.loc["2024-06-14T17:00:00Z", "football_match_window"]) == 0.0

    def test_nan_free(self):
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        assert not out.isna().any().any()

    def test_float32(self):
        idx = pd.date_range("2024-06-14", periods=24, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        assert str(out.dtypes.iloc[0]) == "float32"

    def test_single_named_column(self):
        idx = pd.date_range("2024-06-14", periods=24, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        assert out.columns.tolist() == ["football_match_window"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ExogProviderError):
            FootballMatchWindowProvider(csv_path=tmp_path / "nonexistent.csv").build(
                pd.date_range("2024-06-14", periods=2, freq="h", tz="UTC")
            )

    def test_custom_csv_path(self, tmp_path):
        """A custom CSV with a single event is respected."""
        csv = tmp_path / "custom.csv"
        csv.write_text(
            "event,start_utc,end_utc\n"
            "TEST_MATCH,2024-01-01T10:00:00+00:00,2024-01-01T13:00:00+00:00\n"
        )
        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        out = FootballMatchWindowProvider(csv_path=csv).build(idx)
        assert float(out.loc["2024-01-01T10:00:00Z", "football_match_window"]) == 1.0
        assert float(out.loc["2024-01-01T09:00:00Z", "football_match_window"]) == 0.0

    def test_inclusive_boundary_start(self):
        """Value at exactly start_utc is 1.0."""
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        # EURO2024_GER_SCO_group start_utc = 2024-06-14T18:00:00+00:00
        assert float(out.loc["2024-06-14T18:00:00Z", "football_match_window"]) == 1.0

    def test_inclusive_boundary_end(self):
        """Value at exactly end_utc is 1.0."""
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        # EURO2024_GER_SCO_group end_utc = 2024-06-14T22:00:00+00:00
        assert float(out.loc["2024-06-14T22:00:00Z", "football_match_window"]) == 1.0

    def test_one_hour_before_start_is_zero(self):
        """One hour before start_utc is 0.0."""
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        # start = 18:00 UTC, so 17:00 is outside
        assert float(out.loc["2024-06-14T17:00:00Z", "football_match_window"]) == 0.0

    def test_one_hour_after_end_is_zero(self):
        """One hour after end_utc is 0.0."""
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        # end = 22:00 UTC, so 23:00 is outside
        assert float(out.loc["2024-06-14T23:00:00Z", "football_match_window"]) == 0.0

    def test_timezone_correctness_euro2024_ger_sco(self):
        """EURO 2024 GER-SCO (21:00 CEST = 19:00 UTC) in-window and out-of-window."""
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        # 19:00 UTC is inside the window
        assert float(out.loc["2024-06-14T19:00:00Z", "football_match_window"]) == 1.0
        # 17:00 UTC is outside (before window opens at 18:00 UTC)
        assert float(out.loc["2024-06-14T17:00:00Z", "football_match_window"]) == 0.0

    def test_overlapping_windows_max_intensity(self, tmp_path):
        """Two overlapping rows with intensities 0.5 and 1.0 yield 1.0 in overlap."""
        csv = _make_two_row_csv(tmp_path, intensity_a=0.5, intensity_b=1.0)
        idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
        out = FootballMatchWindowProvider(csv_path=csv).build(idx)
        # hours 00–01: only EVT_A (0.5); hours 02–03: both (max=1.0); hours 04–05: only EVT_B (1.0)
        assert float(
            out.loc["2024-01-01T00:00:00Z", "football_match_window"]
        ) == pytest.approx(0.5)
        assert float(
            out.loc["2024-01-01T02:00:00Z", "football_match_window"]
        ) == pytest.approx(1.0)
        assert float(out.loc["2024-01-01T06:00:00Z", "football_match_window"]) == 0.0

    def test_determinism(self):
        """Two successive build() calls return identical results."""
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        p = FootballMatchWindowProvider()
        out1 = p.build(idx)
        out2 = p.build(idx)
        pd.testing.assert_frame_equal(out1, out2)

    def test_fifteen_minute_cadence(self):
        """15-minute cadence index is handled correctly."""
        # EURO2024_GER_SCO_group: window 18:00–22:00 UTC (inclusive).
        # Span: 18:00, 18:15, ..., 22:00 = 17 quarter-hours inside; 22:15 is outside.
        idx = pd.date_range("2024-06-14 17:00", periods=24, freq="15min", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        assert out.shape == (24, 1)
        assert not out.isna().any().any()
        # 17:00–17:45 UTC: outside (before window opens at 18:00 UTC)
        assert (
            out.loc[
                "2024-06-14 17:00:00+00:00":"2024-06-14 17:45:00+00:00",
                "football_match_window",
            ]
            == 0.0
        ).all()
        # 18:00 UTC: window starts — inside
        assert (
            float(out.loc["2024-06-14 18:00:00+00:00", "football_match_window"]) == 1.0
        )
        # 22:00 UTC: window end — inside (inclusive)
        assert (
            float(out.loc["2024-06-14 22:00:00+00:00", "football_match_window"]) == 1.0
        )


# ---------------------------------------------------------------------------
# EnergyCrisisWindowProvider
# ---------------------------------------------------------------------------


class TestEnergyCrisisWindowProvider:
    def test_in_window_value_is_intensity(self):
        """Hour inside the ENSIKUMAV window is 1.0."""
        idx = pd.date_range("2022-10-01", periods=24, freq="h", tz="UTC")
        out = EnergyCrisisWindowProvider().build(idx)
        assert (out["energy_saving_window"] == 1.0).all()

    def test_outside_window_is_zero(self):
        """Hour before both windows is 0.0."""
        idx = pd.date_range("2022-07-01", periods=24, freq="h", tz="UTC")
        out = EnergyCrisisWindowProvider().build(idx)
        assert (out["energy_saving_window"] == 0.0).all()

    def test_nan_free(self):
        idx = pd.date_range("2022-09-01", periods=100, freq="h", tz="UTC")
        out = EnergyCrisisWindowProvider().build(idx)
        assert not out.isna().any().any()

    def test_float32(self):
        idx = pd.date_range("2022-10-01", periods=24, freq="h", tz="UTC")
        out = EnergyCrisisWindowProvider().build(idx)
        assert str(out.dtypes.iloc[0]) == "float32"

    def test_single_named_column(self):
        idx = pd.date_range("2022-10-01", periods=24, freq="h", tz="UTC")
        out = EnergyCrisisWindowProvider().build(idx)
        assert out.columns.tolist() == ["energy_saving_window"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ExogProviderError):
            EnergyCrisisWindowProvider(csv_path=tmp_path / "nonexistent.csv").build(
                pd.date_range("2022-10-01", periods=2, freq="h", tz="UTC")
            )

    def test_custom_csv_path(self, tmp_path):
        csv = tmp_path / "custom_energy.csv"
        csv.write_text(
            "event,start_utc,end_utc,intensity\n"
            "CUSTOM_WINDOW,2022-10-01T00:00:00+00:00,2022-10-01T05:00:00+00:00,1.0\n"
        )
        idx = pd.date_range("2022-10-01", periods=10, freq="h", tz="UTC")
        out = EnergyCrisisWindowProvider(csv_path=csv).build(idx)
        assert float(out.loc["2022-10-01T03:00:00Z", "energy_saving_window"]) == 1.0
        assert float(out.loc["2022-10-01T06:00:00Z", "energy_saving_window"]) == 0.0

    def test_inclusive_boundary_start(self):
        """Value at exactly start_utc of ENSIKUMAV is 1.0."""
        # ENSIKUMAV start_utc = 2022-08-31T22:00:00+00:00
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2022-08-31T22:00:00", tz="UTC")], freq=None
        )
        out = EnergyCrisisWindowProvider().build(idx)
        assert float(out.iloc[0, 0]) == 1.0

    def test_inclusive_boundary_end(self):
        """Value at exactly end_utc of ENSIKUMAV is 1.0."""
        # ENSIKUMAV end_utc = 2023-04-15T21:00:00+00:00
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2023-04-15T21:00:00", tz="UTC")], freq=None
        )
        out = EnergyCrisisWindowProvider().build(idx)
        assert float(out.iloc[0, 0]) == 1.0

    def test_one_hour_before_start_is_zero(self):
        """One hour before ENSIKUMAV start is 0.0."""
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2022-08-31T21:00:00", tz="UTC")], freq=None
        )
        out = EnergyCrisisWindowProvider().build(idx)
        assert float(out.iloc[0, 0]) == 0.0

    def test_one_hour_after_end_is_zero(self):
        """One hour after ENSIKUMAV end is 0.0."""
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2023-04-15T22:00:00", tz="UTC")], freq=None
        )
        out = EnergyCrisisWindowProvider().build(idx)
        assert float(out.iloc[0, 0]) == 0.0

    def test_overlapping_windows_max_intensity(self, tmp_path):
        """Custom two-row CSV with intensities 0.5/1.0: overlap yields 1.0."""
        csv = _make_two_row_csv(tmp_path, intensity_a=0.5, intensity_b=1.0)
        idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
        out = EnergyCrisisWindowProvider(csv_path=csv).build(idx)
        # hour 00: only EVT_A = 0.5
        assert float(
            out.loc["2024-01-01T00:00:00Z", "energy_saving_window"]
        ) == pytest.approx(0.5)
        # hour 02: both = max(0.5, 1.0) = 1.0
        assert float(
            out.loc["2024-01-01T02:00:00Z", "energy_saving_window"]
        ) == pytest.approx(1.0)
        # hour 06: outside both = 0.0
        assert float(out.loc["2024-01-01T06:00:00Z", "energy_saving_window"]) == 0.0

    def test_determinism(self):
        """Two successive build() calls return identical results."""
        idx = pd.date_range("2022-10-01", periods=48, freq="h", tz="UTC")
        p = EnergyCrisisWindowProvider()
        out1 = p.build(idx)
        out2 = p.build(idx)
        pd.testing.assert_frame_equal(out1, out2)

    def test_fifteen_minute_cadence(self):
        """15-minute cadence index works correctly."""
        # 2022-10-01 is inside the ENSIKUMAV window
        idx = pd.date_range("2022-10-01", periods=96, freq="15min", tz="UTC")
        out = EnergyCrisisWindowProvider().build(idx)
        assert out.shape == (96, 1)
        assert not out.isna().any().any()
        assert (out["energy_saving_window"] == 1.0).all()


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


class TestRegistryNewKeys:
    def test_registry_contains_new_keys(self):
        assert "include_football_match_window" in EXOG_PROVIDER_REGISTRY
        assert "include_energy_saving_window" in EXOG_PROVIDER_REGISTRY

    def test_registry_order_new_keys_after_price(self):
        """New keys appear after include_entsoe_day_ahead_price in declaration order."""
        keys = list(EXOG_PROVIDER_REGISTRY.keys())
        price_idx = keys.index("include_entsoe_day_ahead_price")
        football_idx = keys.index("include_football_match_window")
        energy_idx = keys.index("include_energy_saving_window")
        assert football_idx > price_idx
        assert energy_idx > football_idx

    def test_registry_exact_set(self):
        assert set(EXOG_PROVIDER_REGISTRY) == {
            "include_covid_infection_rate",
            "include_entsoe_forecast_load",
            "include_entsoe_renewable_forecast",
            "include_entsoe_net_load",
            "include_entsoe_day_ahead_price",
            "include_football_match_window",
            "include_energy_saving_window",
        }


# ---------------------------------------------------------------------------
# Timezone robustness (non-UTC tz-aware index)
# ---------------------------------------------------------------------------


class TestTimezoneRobustness:
    def test_non_utc_tz_aware_index_football(self):
        """A Europe/Berlin index is handled correctly: membership is evaluated in
        UTC, but the returned frame retains the original index (same tz, not
        silently converted).

        EURO 2024 GER vs SCO kicked off at 21:00 CEST = 19:00 UTC on
        2024-06-14.  The bundled window is 18:00–22:00 UTC (= 20:00–00:00
        CEST).  With a Europe/Berlin index the in-window hour is therefore
        21:00 local time (= 19:00 UTC).
        """
        tz_berlin = "Europe/Berlin"
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz=tz_berlin)
        out = FootballMatchWindowProvider().build(idx)

        # Index identity: same timezone, not converted to UTC.
        assert out.index.tz is not None
        assert str(out.index.tz) == tz_berlin
        pd.testing.assert_index_equal(out.index, idx)

        # 2024-06-14T19:00 UTC = 2024-06-14T21:00 Europe/Berlin is in-window.
        in_window_local = pd.Timestamp("2024-06-14T21:00:00", tz=tz_berlin)
        assert float(out.loc[in_window_local, "football_match_window"]) == 1.0

        # Well outside the window: 06:00 local (= 04:00 UTC).
        outside_local = pd.Timestamp("2024-06-14T06:00:00", tz=tz_berlin)
        assert float(out.loc[outside_local, "football_match_window"]) == 0.0

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Soundness tests for weather-data generation.

These tests document and guard against the *flat temperature* bug seen in the
ddmo-sose-26 chapter 14 figure ``fig-team4-weather`` (the temperature line goes
constant for ~26--30 May). The root cause is two interacting defects in
``spotforecast2_safe``:

A. Fetch coverage hole. ``WeatherService._fetch_hybrid`` (weather/client.py)
   fetches the *archive* endpoint only up to ``now - 5 days`` and the *forecast*
   endpoint (``fetch_forecast``, which sends ``forecast_days`` but no
   ``past_days``) only from *today 00:00* onward. The window
   ``[now - 5d, today_00:00)`` -- the most recent ~5 days -- is fetched by
   neither branch, so those rows are simply absent from the merged frame.

B. Silent last-observation-carried-forward. ``get_weather_features``
   (weather/features.py) reindexes the fetched frame onto a regular hourly grid
   with ``reindex(extended_index, method="ffill")``, turning the absent recent
   rows into a repeat of the last real value -> a multi-day constant temperature.
   The fail-safe ``fill_missing=False`` path in ``_finalize_df`` does not catch
   this, because at ``freq="h"`` it skips the resample/reindex, so the *absent*
   rows are never materialised as ``NaN`` and the gap-rejection ``isna()`` check
   sees nothing.

"Sound weather data" here means: a recent fetch gap must be surfaced (raised or
otherwise refused), never silently turned into a long run of identical values.

All tests run offline -- the Open-Meteo HTTP boundary is mocked.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.weather import WeatherFetchError, get_weather_features
from spotforecast2_safe.weather.client import WeatherService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HOURLY_PARAMS = WeatherService.HOURLY_PARAMS

# A real national temperature series is never bit-identical for a full day; a
# constant run longer than this is implausible and signals carried-forward fill.
MAX_PLAUSIBLE_FLAT_HOURS = 24


def _hourly_frame(start, end, vary_temperature=True) -> pd.DataFrame:
    """Build a synthetic Open-Meteo-shaped hourly frame on ``[start, end]``.

    ``temperature_2m`` carries a diurnal sine so that *any* long constant run in
    a result is attributable to downstream fill, not to the input.
    """
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(idx)
    cols = {}
    for param in HOURLY_PARAMS:
        if param == "temperature_2m" and vary_temperature:
            cols[param] = 10.0 + 10.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
        else:
            cols[param] = np.linspace(1.0, 2.0, n)
    return pd.DataFrame(cols, index=idx)


def _max_constant_run(series: pd.Series) -> int:
    """Length of the longest run of consecutive (near-)identical values."""
    vals = series.to_numpy()
    if len(vals) == 0:
        return 0
    best = run = 1
    for i in range(1, len(vals)):
        if np.isclose(vals[i], vals[i - 1], rtol=0.0, atol=1e-9):
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


# ---------------------------------------------------------------------------
# Helper self-tests (these must pass on current code)
# ---------------------------------------------------------------------------


class TestConstantRunHelper:
    """`_max_constant_run` measures carried-forward flatness correctly."""

    def test_varying_series_has_short_runs(self):
        s = pd.Series(10.0 + 10.0 * np.sin(np.arange(240) * 2 * np.pi / 24))
        assert _max_constant_run(s) < MAX_PLAUSIBLE_FLAT_HOURS

    def test_flat_tail_is_detected(self):
        varying = list(10.0 + 10.0 * np.sin(np.arange(48) * 2 * np.pi / 24))
        flat = [varying[-1]] * 120  # 5 days carried forward
        s = pd.Series(varying + flat)
        assert _max_constant_run(s) >= 120


# ---------------------------------------------------------------------------
# Defect A — hybrid fetch must cover the recent [now-5d, now] window
# ---------------------------------------------------------------------------


class TestHybridFetchCoverage:
    """`_fetch_hybrid` jointly covers the recent window via the forecast
    endpoint's ``past_days``, leaving no hole for downstream LOCF (Defect A)."""

    def test_hybrid_fetch_covers_recent_window(self):
        """Every hour in [now-5d, now] is fetched (archive + forecast past_days)."""
        now = pd.Timestamp.now(tz="UTC").floor("h")
        start = now - pd.Timedelta(days=10)
        end = now + pd.Timedelta(days=1)
        archive_cutoff = now - pd.Timedelta(days=5)

        service = WeatherService(latitude=51.5136, longitude=7.4653, cache_path=None)

        # Simulate the two Open-Meteo endpoints honouring their arguments:
        # archive returns [start, arch_end]; forecast returns
        # [today - past_days, today + days_ahead]. Without past_days the recent
        # window would be uncovered (the bug); the fix requests it.
        def fake_archive(s, e, timezone="UTC"):
            return _hourly_frame(s, e)

        def fake_forecast(days_ahead, timezone="UTC", past_days=0):
            f_start = now.normalize() - pd.Timedelta(days=past_days)
            f_end = now.normalize() + pd.Timedelta(days=days_ahead)
            return _hourly_frame(f_start, f_end)

        with patch.object(
            service, "fetch_archive", side_effect=fake_archive
        ), patch.object(service, "fetch_forecast", side_effect=fake_forecast):
            merged = service._fetch_hybrid(start, end, "UTC")

        recent = pd.date_range(archive_cutoff, now, freq="h")
        missing = recent.difference(merged.index)
        assert len(missing) == 0, f"{len(missing)} recent hours not fetched"


# ---------------------------------------------------------------------------
# Defect B1 — _finalize_df must not let an absent-row gap pass silently
# ---------------------------------------------------------------------------


class TestFinalizeGapDetection:
    """`_finalize_df(fill_missing=False)` must refuse gaps, however they appear."""

    def _service(self):
        return WeatherService(latitude=51.5136, longitude=7.4653, cache_path=None)

    def test_explicit_nan_gap_is_rejected(self):
        """Already sound: a NaN row is rejected with the gap timestamps."""
        df = _hourly_frame("2026-05-20", "2026-05-25")
        df.iloc[10] = np.nan  # an explicit NaN gap
        with pytest.raises(ValueError, match="missing row"):
            self._service()._finalize_df(df, freq="h", fill_missing=False)

    def test_absent_hour_gap_is_rejected(self):
        """Sound behaviour: a missing hourly row is detected, not ignored.

        (Defect B1: previously at freq='h' the reindex was skipped, so an absent
        row never became NaN and the isna() gap-check missed it.)
        """
        df = _hourly_frame("2026-05-20", "2026-05-25")
        df = df.drop(df.index[10])  # an *absent* row (a real fetch hole)
        with pytest.raises(ValueError, match="missing row"):
            self._service()._finalize_df(df, freq="h", fill_missing=False)


# ---------------------------------------------------------------------------
# Defect B (core) — get_weather_features must not synthesise a flat tail
# ---------------------------------------------------------------------------


class TestWeatherFeaturesNoSilentLOCF:
    """The pipeline's weather alignment must not carry a value forward for days
    (Defect B): a long fetch gap is refused loudly, not flat-lined."""

    def test_recent_gap_is_refused_not_flat_filled(self):
        start = pd.Timestamp("2026-05-01 00:00", tz="UTC")
        last_real = pd.Timestamp("2026-05-25 23:00", tz="UTC")  # archive ends here
        cov_end = pd.Timestamp("2026-05-31 23:00", tz="UTC")  # demanded coverage

        # Fetched weather covers only [start, last_real]; 26-30 May is absent.
        holed_weather = _hourly_frame(start, last_real)

        # Reference target frame (used by curate_weather for shape validation only).
        data = pd.DataFrame(
            {"load": 1.0},
            index=pd.date_range(start, cov_end - pd.Timedelta(hours=24), freq="h", tz="UTC"),
        )

        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=holed_weather,
        ):
            # A multi-day hole must raise, not be silently carried forward into a
            # constant temperature. WeatherFetchError is catchable by the
            # pipeline's on_weather_failure="skip" path.
            with pytest.raises(WeatherFetchError, match="coverage gap"):
                get_weather_features(
                    data=data,
                    start=start,
                    cov_end=cov_end,
                    forecast_horizon=24,
                )

    def test_small_gap_is_still_forward_filled(self):
        """A short benign gap (<= tolerance) is tolerated, not refused."""
        start = pd.Timestamp("2026-05-01 00:00", tz="UTC")
        cov_end = pd.Timestamp("2026-05-10 00:00", tz="UTC")
        # Drop a single hour — a benign boundary-style gap.
        weather = _hourly_frame(start, cov_end)
        weather = weather.drop(weather.index[100])

        data = pd.DataFrame(
            {"load": 1.0},
            index=pd.date_range(start, cov_end - pd.Timedelta(hours=24), freq="h", tz="UTC"),
        )
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=weather,
        ):
            _features, weather_aligned = get_weather_features(
                data=data, start=start, cov_end=cov_end, forecast_horizon=24
            )
        assert "temperature_2m" in weather_aligned.columns
        assert not weather_aligned["temperature_2m"].isna().any()

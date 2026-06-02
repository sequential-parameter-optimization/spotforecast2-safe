# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Leaf-builder failure-mode tests for spotforecast2-safe.

These mirror the orchestration-layer suite in
``spotforecast2/tests/test_exogenous_failure_scenarios.py`` but exercise
the *real* leaf builders rather than patching them.  Together the two
suites lock in the failure-mode matrix documented in
``Publications/bart26b_spotforecast2_entsoe/index.qmd``
(``@tbl-exo-failure-scenarios``).

Scope here — the contracts at the safe-layer boundary:

- S1: ``get_calendar_features`` is the bullet-proof group; pure
  datetime arithmetic, no I/O, no config-derived externalities.
- S2 / S3: ``get_weather_features`` (the leaf) does **not** swallow
  ``WeatherFetchError``.  Only ``BaseTask`` decides to skip; the safe
  layer always propagates so the orchestrator can choose its policy.
- S4: ``get_holiday_features(country_code="ZZ", ...)`` raises
  ``NotImplementedError`` from the ``holidays`` library.  No graceful
  handling exists by design.
- S5: ``get_day_night_features`` with an out-of-range latitude raises
  ``ValueError`` from ``astral``.  No graceful handling exists by
  design.

S6 (train-with-weather → predict-without-weather) and S7 (master
toggle) are orchestration concerns and live in the sf2 suite — not
duplicated here.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from spotforecast2_safe.weather import WeatherFetchError

HOURLY_INDEX = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# S1 — calendar features always succeed
# ---------------------------------------------------------------------------


class TestS1CalendarFeaturesAlwaysSucceed:
    def test_returns_expected_columns(self):
        from spotforecast2_safe.calendar import get_calendar_features

        df = get_calendar_features(
            start=HOURLY_INDEX[0],
            cov_end=HOURLY_INDEX[-1],
            freq="h",
            timezone="UTC",
        )
        assert df.shape == (48, 4)
        assert df.columns.tolist() == ["month", "week", "day_of_week", "hour"]


# ---------------------------------------------------------------------------
# S2 / S3 — weather-leaf does NOT swallow WeatherFetchError
# ---------------------------------------------------------------------------


class TestS2S3WeatherLeafPropagatesFailure:
    """The graceful-degradation policy lives at the orchestrator, not the leaf.

    ``BaseTask.build_exogenous_features`` is the single place that
    decides ``raise`` vs ``skip``.  The safe-layer leaf must therefore
    always surface ``WeatherFetchError`` so that callers retain control.
    """

    def test_get_weather_features_propagates_weather_fetch_error(self):
        from spotforecast2_safe.weather import get_weather_features

        data = pd.DataFrame({"load": range(48)}, index=HOURLY_INDEX)
        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            side_effect=WeatherFetchError("simulated outage"),
        ):
            with pytest.raises(WeatherFetchError, match="simulated outage"):
                get_weather_features(
                    data=data,
                    start=HOURLY_INDEX[0],
                    cov_end=HOURLY_INDEX[-1],
                    forecast_horizon=2,
                )


# ---------------------------------------------------------------------------
# S4 — invalid country_code propagates from the holidays library
# ---------------------------------------------------------------------------


class TestS4InvalidCountryCodeRaisesAtLeaf:
    """Unsupported ISO 3166-1 alpha-2 code surfaces ``NotImplementedError``."""

    def test_create_holiday_df_invalid_country(self):
        from spotforecast2_safe.calendar.holiday import create_holiday_df

        with pytest.raises(NotImplementedError, match="Country ZZ"):
            create_holiday_df(
                start=HOURLY_INDEX[0],
                end=HOURLY_INDEX[-1],
                freq="h",
                country_code="ZZ",
                state="NW",
            )

    def test_get_holiday_features_invalid_country(self):
        from spotforecast2_safe.calendar import get_holiday_features

        data = pd.DataFrame({"load": range(48)}, index=HOURLY_INDEX)
        with pytest.raises(NotImplementedError, match="Country ZZ"):
            get_holiday_features(
                data=data,
                start=HOURLY_INDEX[0],
                cov_end=HOURLY_INDEX[-1] + pd.Timedelta(hours=2),
                forecast_horizon=2,
                country_code="ZZ",
                state="NW",
            )


# ---------------------------------------------------------------------------
# S5 — invalid coordinates propagate from astral
# ---------------------------------------------------------------------------


class TestS5InvalidCoordinatesRaiseAtLeaf:
    """An out-of-range latitude breaks ``astral.sun.sun`` and propagates."""

    def test_get_day_night_features_polar_latitude_raises(self):
        from astral import LocationInfo

        from spotforecast2_safe.calendar import get_day_night_features

        # Lat 89.999 sits inside the polar circle in mid-summer; astral
        # is then unable to compute civil twilight and raises ValueError.
        location = LocationInfo(latitude=89.999, longitude=0.0, timezone="UTC")
        summer_idx = pd.date_range("2024-06-20", periods=48, freq="h", tz="UTC")
        with pytest.raises(ValueError):
            get_day_night_features(
                start=summer_idx[0],
                cov_end=summer_idx[-1],
                location=location,
                freq="h",
                timezone="UTC",
            )

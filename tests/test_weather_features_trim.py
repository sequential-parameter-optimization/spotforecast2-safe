# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the ``get_weather_features`` trim-before-curate fix.

Open-Meteo can return whole-day forecast blocks aligned to its own clock,
so the raw payload occasionally extends a few hours past ``cov_end``. In
that case ``curate_weather`` used to print a spurious
``"Weather dataframe has wrong shape."`` even though the very next
reindex aligned the frame correctly. The fix trims the raw frame to
``[start, cov_end]`` before the curate check.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from spotforecast2_safe.weather import get_weather_features


def _make_data(start: str, periods: int) -> pd.DataFrame:
    """Build an hourly UTC-indexed reference frame."""
    idx = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame({"load": range(periods)}, index=idx)


def _make_raw_weather(start: str, periods: int) -> pd.DataFrame:
    """Build a raw weather frame at hourly UTC resolution."""
    idx = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "temperature_2m": [10.0] * periods,
            "wind_speed_10m": [3.0] * periods,
        },
        index=idx,
    )


class TestWeatherTrimBeforeCurate:
    """The raw weather frame is trimmed to ``[start, cov_end]`` before curate."""

    def test_no_warning_when_raw_extends_past_cov_end(self, capsys):
        """Extra rows past ``cov_end`` must not trigger the wrong-shape print."""
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        cov_end = pd.Timestamp("2024-01-03 23:00", tz="UTC")
        forecast_horizon = 24
        data = _make_data("2024-01-01 00:00", periods=48)  # 48 + 24 == 72

        # Raw weather extends 6 hours past cov_end — the "whole-day block"
        # behaviour that surfaced the spurious warning in production.
        raw = _make_raw_weather("2024-01-01 00:00", periods=78)

        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=raw,
        ):
            features, aligned = get_weather_features(
                data=data,
                start=start,
                cov_end=cov_end,
                forecast_horizon=forecast_horizon,
            )

        captured = capsys.readouterr()
        assert "Weather dataframe has wrong shape." not in captured.out
        assert len(aligned) == 72  # data.shape[0] + forecast_horizon
        assert features.shape[0] == 72

    def test_warning_still_fires_on_genuine_gap(self, capsys):
        """A real coverage gap inside the window must still warn."""
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        cov_end = pd.Timestamp("2024-01-03 23:00", tz="UTC")
        forecast_horizon = 24
        data = _make_data("2024-01-01 00:00", periods=48)

        # Drop a row from inside the window so the trimmed frame is short.
        raw = _make_raw_weather("2024-01-01 00:00", periods=72)
        raw = raw.drop(raw.index[10])

        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=raw,
        ):
            get_weather_features(
                data=data,
                start=start,
                cov_end=cov_end,
                forecast_horizon=forecast_horizon,
            )

        captured = capsys.readouterr()
        assert "Weather dataframe has wrong shape." in captured.out

    def test_trim_handles_exact_match(self, capsys):
        """Raw frame already exactly the requested length: no warning, no change."""
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        cov_end = pd.Timestamp("2024-01-03 23:00", tz="UTC")
        forecast_horizon = 24
        data = _make_data("2024-01-01 00:00", periods=48)
        raw = _make_raw_weather("2024-01-01 00:00", periods=72)

        with patch(
            "spotforecast2_safe.data.fetch_data.fetch_weather_data",
            return_value=raw,
        ):
            _, aligned = get_weather_features(
                data=data,
                start=start,
                cov_end=cov_end,
                forecast_horizon=forecast_horizon,
            )

        captured = capsys.readouterr()
        assert "Weather dataframe has wrong shape." not in captured.out
        assert len(aligned) == 72


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

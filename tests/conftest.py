# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared pytest fixtures."""

import pytest

import spotforecast2_safe.weather.client as _weather_client


@pytest.fixture(autouse=True)
def _disable_open_meteo_throttle():
    """Disable the proactive Open-Meteo request throttle during tests.

    The throttle only spaces real HTTP requests; tests mock the network, so the
    spacing would add wall-clock sleeps without changing behaviour. Individual
    tests that exercise the throttle set the interval explicitly.
    """
    saved = _weather_client._MIN_REQUEST_INTERVAL_S
    _weather_client._MIN_REQUEST_INTERVAL_S = 0.0
    try:
        yield
    finally:
        _weather_client._MIN_REQUEST_INTERVAL_S = saved

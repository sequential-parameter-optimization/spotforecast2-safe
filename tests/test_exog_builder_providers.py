# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ExogBuilder's provider integration and failure policy."""

import pandas as pd
import pytest

from spotforecast2_safe.configurator._base_config import default_periods
from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder
from spotforecast2_safe.preprocessing.exog_providers import (
    CovidInfectionRateProvider,
    EntsoeForecastLoadProvider,
)

START = pd.Timestamp("2021-12-01", tz="UTC")
END = pd.Timestamp("2021-12-02", tz="UTC")


def _columns_without_providers():
    return set(
        ExogBuilder(periods=default_periods(), country_code="DE")
        .build(START, END)
        .columns
    )


def test_providers_none_is_unchanged():
    builder = ExogBuilder(periods=default_periods(), country_code="DE", providers=None)
    out = builder.build(START, END)
    assert set(out.columns) == _columns_without_providers()


def test_provider_columns_appended():
    builder = ExogBuilder(
        periods=default_periods(),
        country_code="DE",
        providers=[CovidInfectionRateProvider()],
    )
    out = builder.build(START, END)
    assert "covid_infection_rate" in out.columns
    assert set(out.columns) - {"covid_infection_rate"} == _columns_without_providers()
    assert not out["covid_infection_rate"].isna().any()


def test_on_provider_failure_raise(tmp_path):
    builder = ExogBuilder(
        periods=default_periods(),
        country_code="DE",
        providers=[EntsoeForecastLoadProvider(data_home=tmp_path)],
        on_provider_failure="raise",
    )
    with pytest.raises(Exception):
        builder.build(START, END)


def test_on_provider_failure_skip(tmp_path, caplog):
    builder = ExogBuilder(
        periods=default_periods(),
        country_code="DE",
        providers=[EntsoeForecastLoadProvider(data_home=tmp_path)],
        on_provider_failure="skip",
    )
    out = builder.build(START, END)
    assert "entsoe_forecasted_load" not in out.columns
    assert set(out.columns) == _columns_without_providers()


def test_skip_keeps_other_providers(tmp_path):
    builder = ExogBuilder(
        periods=default_periods(),
        country_code="DE",
        providers=[
            EntsoeForecastLoadProvider(data_home=tmp_path),  # will fail
            CovidInfectionRateProvider(),  # will succeed
        ],
        on_provider_failure="skip",
    )
    out = builder.build(START, END)
    assert "entsoe_forecasted_load" not in out.columns
    assert "covid_infection_rate" in out.columns


def test_invalid_on_provider_failure_rejected():
    with pytest.raises(ValueError):
        ExogBuilder(on_provider_failure="nope")

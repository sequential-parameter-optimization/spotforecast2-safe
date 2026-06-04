# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ExogBuilder's provider integration and failure policy."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator._base_config import default_periods
from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder
from spotforecast2_safe.preprocessing.exog_providers import (
    CovidInfectionRateProvider,
    EntsoeDayAheadPriceProvider,
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


# ---------------------------------------------------------------------------
# provider_window: price provider with only the back-half populated
# ---------------------------------------------------------------------------


def test_price_provider_window_back_half_only(tmp_path):
    """Price CSV whose first half is absent; provider_window covers only the
    back half → build succeeds and the result is NaN-free within the window."""
    full_index = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
    back_half = full_index[24:]

    # Write a CSV that only has the back 24 hours
    interim = tmp_path / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Day-ahead Price": 50.0}, index=back_half).rename_axis(
        "Time (UTC)"
    ).to_csv(interim / "day_ahead_price.csv")

    provider = EntsoeDayAheadPriceProvider(
        data_home=tmp_path,
        max_gap=0,
        provider_window=back_half,
    )
    result = provider.build(full_index)
    # The first 24 hours are NaN (out-of-window, not validated)
    assert result.iloc[:24].isna().all().all()
    # The back 24 hours are populated and NaN-free
    back_vals = result.loc[back_half, "entsoe_day_ahead_price"]
    assert not back_vals.isna().any()
    assert float(back_vals.mean()) == pytest.approx(50.0, abs=1e-3)
    assert result.dtypes.iloc[0] == np.dtype("float32")

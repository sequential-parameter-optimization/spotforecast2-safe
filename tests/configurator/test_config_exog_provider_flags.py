# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the provider-based exogenous toggles on the config classes."""

import pytest

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.preprocessing.exog_providers import EXOG_PROVIDER_REGISTRY

PROVIDER_FLAGS = tuple(EXOG_PROVIDER_REGISTRY)


@pytest.fixture(params=[ConfigEntsoe, ConfigMulti])
def config_cls(request):
    return request.param


def test_flags_default_false(config_cls):
    cfg = config_cls()
    for flag in PROVIDER_FLAGS:
        assert getattr(cfg, flag) is False, flag
    assert cfg.on_exog_provider_failure == "raise"


def test_flags_in_param_names(config_cls):
    names = config_cls._PARAM_NAMES
    for flag in PROVIDER_FLAGS:
        assert flag in names, flag
    assert "on_exog_provider_failure" in names


def test_get_params_round_trips_flags(config_cls):
    cfg = config_cls(
        include_covid_infection_rate=True,
        include_entsoe_forecast_load=True,
        on_exog_provider_failure="skip",
    )
    params = cfg.get_params()
    assert params["include_covid_infection_rate"] is True
    assert params["include_entsoe_forecast_load"] is True
    assert params["include_entsoe_net_load"] is False
    assert params["on_exog_provider_failure"] == "skip"


def test_set_params_updates_flags(config_cls):
    cfg = config_cls()
    cfg.set_params(
        include_entsoe_renewable_forecast=True, on_exog_provider_failure="skip"
    )
    assert cfg.include_entsoe_renewable_forecast is True
    assert cfg.on_exog_provider_failure == "skip"


def test_invalid_provider_failure_policy_rejected(config_cls):
    with pytest.raises(ValueError):
        config_cls(on_exog_provider_failure="nope")


def test_invalid_provider_failure_policy_rejected_on_set(config_cls):
    cfg = config_cls()
    with pytest.raises(ValueError):
        cfg.set_params(on_exog_provider_failure="bogus")

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


# ---------------------------------------------------------------------------
# exog_max_gap_hours and exog_provider_window (new params)
# ---------------------------------------------------------------------------


def test_exog_gap_defaults(config_cls):
    cfg = config_cls()
    assert cfg.exog_max_gap_hours == 0
    assert cfg.exog_max_tail_gap_hours == 0
    assert cfg.exog_provider_window == "full"


def test_exog_gap_in_param_names(config_cls):
    assert "exog_max_gap_hours" in config_cls._PARAM_NAMES
    assert "exog_max_tail_gap_hours" in config_cls._PARAM_NAMES
    assert "exog_provider_window" in config_cls._PARAM_NAMES


def test_exog_gap_get_set_round_trip(config_cls):
    cfg = config_cls(exog_max_gap_hours=6, exog_provider_window="train")
    params = cfg.get_params()
    assert params["exog_max_gap_hours"] == 6
    assert params["exog_provider_window"] == "train"
    cfg2 = config_cls()
    cfg2.set_params(exog_max_gap_hours=6, exog_provider_window="train")
    assert cfg2.exog_max_gap_hours == 6
    assert cfg2.exog_provider_window == "train"


def test_exog_tail_gap_default_zero(config_cls):
    """exog_max_tail_gap_hours defaults to 0."""
    cfg = config_cls()
    assert cfg.exog_max_tail_gap_hours == 0
    assert "exog_max_tail_gap_hours" in config_cls._PARAM_NAMES


def test_exog_tail_gap_get_set_round_trip(config_cls):
    """exog_max_tail_gap_hours is round-tripped through get_params/set_params."""
    cfg = config_cls(exog_max_gap_hours=3, exog_max_tail_gap_hours=48)
    params = cfg.get_params()
    assert params["exog_max_tail_gap_hours"] == 48
    cfg2 = config_cls()
    cfg2.set_params(exog_max_tail_gap_hours=48)
    assert cfg2.exog_max_tail_gap_hours == 48


# ---------------------------------------------------------------------------
# New event-window flags
# ---------------------------------------------------------------------------


def test_new_event_window_flags_default_false(config_cls):
    """include_football_match_window and include_energy_saving_window default to False."""
    cfg = config_cls()
    assert cfg.include_football_match_window is False
    assert cfg.include_energy_saving_window is False


def test_new_event_window_flags_in_param_names(config_cls):
    """New flags appear in _PARAM_NAMES."""
    assert "include_football_match_window" in config_cls._PARAM_NAMES
    assert "include_energy_saving_window" in config_cls._PARAM_NAMES


def test_new_event_window_flags_get_params_round_trip(config_cls):
    """New flags are round-tripped through get_params/set_params."""
    cfg = config_cls(
        include_football_match_window=True,
        include_energy_saving_window=True,
    )
    params = cfg.get_params()
    assert params["include_football_match_window"] is True
    assert params["include_energy_saving_window"] is True
    assert params["include_covid_infection_rate"] is False

    cfg2 = config_cls()
    cfg2.set_params(
        include_football_match_window=True,
        include_energy_saving_window=True,
    )
    assert cfg2.include_football_match_window is True
    assert cfg2.include_energy_saving_window is True

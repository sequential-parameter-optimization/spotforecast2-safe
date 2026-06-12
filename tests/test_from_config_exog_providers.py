# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""from_config forwards the provider flags into the model's ExogBuilder."""

import pytest

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.forecaster.wrappers import (
    ForecasterRecursiveLGBM,
    ForecasterRecursiveModel,
)


@pytest.mark.parametrize("config_cls", [ConfigEntsoe, ConfigMulti])
@pytest.mark.parametrize(
    "model_cls", [ForecasterRecursiveModel, ForecasterRecursiveLGBM]
)
def test_flags_forwarded_into_preprocessor(config_cls, model_cls):
    cfg = config_cls(
        include_covid_infection_rate=True,
        include_entsoe_forecast_load=True,
        on_exog_provider_failure="skip",
    )
    model = model_cls.from_config(iteration=0, config=cfg)
    names = [p.name for p in model.preprocessor.providers]
    assert names == ["covid_infection_rate", "entsoe_forecasted_load"]
    assert model.preprocessor.on_provider_failure == "skip"


def test_no_flags_means_no_providers():
    model = ForecasterRecursiveLGBM.from_config(iteration=0, config=ConfigEntsoe())
    assert model.preprocessor.providers == []


def test_direct_construction_default_has_no_providers():
    model = ForecasterRecursiveModel(iteration=0)
    assert model.preprocessor.providers == []
    assert model.preprocessor.on_provider_failure == "raise"


# ---------------------------------------------------------------------------
# exog_max_gap_hours threading through from_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_cls", [ConfigEntsoe, ConfigMulti])
def test_from_config_max_gap_forwarded_to_providers(config_cls):
    """Config with include_entsoe_forecast_load=True and exog_max_gap_hours=3
    → the built providers carry max_gap==3."""
    cfg = config_cls(
        include_entsoe_forecast_load=True,
        exog_max_gap_hours=3,
    )
    model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
    assert len(model.preprocessor.providers) == 1
    provider = model.preprocessor.providers[0]
    assert provider.max_gap == 3


@pytest.mark.parametrize("config_cls", [ConfigEntsoe, ConfigMulti])
def test_from_config_default_max_gap_is_zero(config_cls):
    """Default config → max_gap==0 and no providers."""
    cfg = config_cls()
    model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
    assert model.preprocessor.providers == []
    assert model.exog_max_gap_hours == 0


# ---------------------------------------------------------------------------
# exog_max_tail_gap_hours threading through from_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_cls", [ConfigEntsoe, ConfigMulti])
def test_from_config_max_tail_gap_forwarded_to_providers(config_cls):
    """Config with exog_max_tail_gap_hours=48 → providers carry max_tail_gap==48."""
    cfg = config_cls(
        include_entsoe_forecast_load=True,
        exog_max_gap_hours=3,
        exog_max_tail_gap_hours=48,
    )
    model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
    assert len(model.preprocessor.providers) == 1
    provider = model.preprocessor.providers[0]
    assert provider.max_tail_gap == 48


@pytest.mark.parametrize("config_cls", [ConfigEntsoe, ConfigMulti])
def test_from_config_default_max_tail_gap_is_zero(config_cls):
    """Default config → max_tail_gap==0."""
    cfg = config_cls(include_entsoe_forecast_load=True)
    model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
    assert len(model.preprocessor.providers) == 1
    assert model.preprocessor.providers[0].max_tail_gap == 0


# ---------------------------------------------------------------------------
# New event-window flags flow through from_config in registry order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag, expected_name",
    [
        ("include_football_match_window", "football_match_window"),
        ("include_energy_saving_window", "energy_saving_window"),
    ],
)
@pytest.mark.parametrize("config_cls", [ConfigEntsoe, ConfigMulti])
def test_event_window_flags_produce_providers(config_cls, flag, expected_name):
    """Setting an event-window flag produces the correct provider."""
    cfg = config_cls(**{flag: True})
    model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
    names = [p.name for p in model.preprocessor.providers]
    assert expected_name in names


@pytest.mark.parametrize("config_cls", [ConfigEntsoe, ConfigMulti])
def test_event_window_flags_registry_order(config_cls):
    """Both new flags enabled → providers appear in registry order (football before energy)."""
    cfg = config_cls(
        include_football_match_window=True,
        include_energy_saving_window=True,
    )
    model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
    names = [p.name for p in model.preprocessor.providers]
    assert names == ["football_match_window", "energy_saving_window"]

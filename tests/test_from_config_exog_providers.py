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

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.data import Period


def test_config_entsoe_defaults():
    """Test that ConfigEntsoe initializes with correct defaults."""
    config = ConfigEntsoe()
    assert config.country_code == "DE"
    assert config.predict_size == 24
    assert config.refit_size == 7
    assert config.random_state == 314159
    assert len(config.periods) == 5
    assert isinstance(config.train_size, pd.Timedelta)
    assert config.train_size == pd.Timedelta(days=3 * 365)


def test_config_entsoe_custom():
    """Test custom overrides in ConfigEntsoe."""
    custom_periods = [
        Period(name="test", n_periods=1, column="col", input_range=(0, 1))
    ]
    config = ConfigEntsoe(
        country_code="FR", predict_size=48, random_state=42, periods=custom_periods
    )
    assert config.country_code == "FR"
    assert config.predict_size == 48
    assert config.random_state == 42
    assert config.periods == custom_periods


def test_config_entsoe_timedeltas():
    """Test timedelta parameters."""
    train_size = pd.Timedelta(days=10)
    delta_val = pd.Timedelta(hours=5)
    config = ConfigEntsoe(train_size=train_size, delta_val=delta_val)
    assert config.train_size == train_size
    assert config.delta_val == delta_val


def test_config_entsoe_root_import():
    """Test that Config can be imported from the root package."""
    from spotforecast2_safe import Config
    from spotforecast2_safe import ConfigEntsoe as CE

    assert Config is CE

    config = Config()
    assert isinstance(config, CE)


def test_config_entsoe_retrain_max_age_defaults_to_seven_days():
    """retrain_max_age defaults to 7 days for the cadence gate."""
    config = ConfigEntsoe()
    assert config.retrain_max_age == pd.Timedelta(days=7)


def test_config_entsoe_retrain_max_age_override():
    """retrain_max_age can be overridden via the constructor."""
    config = ConfigEntsoe(retrain_max_age=pd.Timedelta(days=3))
    assert config.retrain_max_age == pd.Timedelta(days=3)


def test_config_entsoe_retrain_max_age_in_get_params():
    """retrain_max_age is exposed via get_params() for serialization."""
    config = ConfigEntsoe()
    params = config.get_params()
    assert params["retrain_max_age"] == pd.Timedelta(days=7)


def test_config_entsoe_imputation_window_size_default_none():
    """imputation_window_size defaults to None (falls back to window_size)."""
    config = ConfigEntsoe()
    assert config.imputation_window_size is None


def test_config_entsoe_imputation_window_size_custom():
    """imputation_window_size is stored when set and is independent of window_size."""
    config = ConfigEntsoe(window_size=72, imputation_window_size=6)
    assert config.window_size == 72
    assert config.imputation_window_size == 6


def test_config_entsoe_imputation_window_size_in_get_params():
    """imputation_window_size is exposed via get_params()."""
    params = ConfigEntsoe(imputation_window_size=6).get_params()
    assert params["imputation_window_size"] == 6

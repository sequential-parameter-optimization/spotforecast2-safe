# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ConfigEntsoe must carry the same opt-in feature flags as ConfigMulti.

ConfigEntsoe is an independent dataclass (not a ConfigMulti subclass), so new
feature toggles added to ConfigMulti must be mirrored here or the ENTSO-E
single-target pipeline silently cannot enable them.
"""

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.configurator.config_multi import ConfigMulti

NEW_FEATURE_FLAGS = (
    "use_population_weighted_weather",
    "include_degree_hours",
    "include_apparent_temperature",
    "include_ephemeris_features",
    "include_day_type_features",
)
NEW_FLOAT_FIELDS = ("degree_hours_base_heating", "degree_hours_base_cooling")


class TestFeatureFlagParity:
    def test_flags_present_and_default_off(self):
        cfg = ConfigEntsoe()
        for flag in NEW_FEATURE_FLAGS:
            assert hasattr(cfg, flag), flag
            assert getattr(cfg, flag) is False, flag

    def test_float_defaults_match_configmulti(self):
        entsoe, multi = ConfigEntsoe(), ConfigMulti()
        for fld in NEW_FLOAT_FIELDS:
            assert getattr(entsoe, fld) == getattr(multi, fld), fld

    def test_flags_in_param_names(self):
        for flag in NEW_FEATURE_FLAGS + NEW_FLOAT_FIELDS:
            assert flag in ConfigEntsoe._PARAM_NAMES, flag

    def test_constructs_with_flags_enabled(self):
        cfg = ConfigEntsoe(
            use_population_weighted_weather=True,
            include_degree_hours=True,
            include_apparent_temperature=True,
            include_ephemeris_features=True,
            include_day_type_features=True,
            degree_hours_base_heating=16.0,
            degree_hours_base_cooling=21.0,
        )
        assert cfg.include_ephemeris_features is True
        assert cfg.include_day_type_features is True
        assert cfg.degree_hours_base_heating == 16.0

    def test_configmulti_and_entsoe_agree_on_flag_set(self):
        multi = set(ConfigMulti._PARAM_NAMES)
        entsoe = set(ConfigEntsoe._PARAM_NAMES)
        for flag in NEW_FEATURE_FLAGS + NEW_FLOAT_FIELDS:
            assert flag in multi and flag in entsoe, flag


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

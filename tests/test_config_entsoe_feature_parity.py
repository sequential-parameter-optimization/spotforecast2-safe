# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ConfigEntsoe inherits ConfigMulti — the parity gap is closed structurally.

Since 20.0.0 ``ConfigEntsoe`` is a subclass of ``ConfigMulti``, so every feature
flag (and every other field) on ``ConfigMulti`` is present on ``ConfigEntsoe``
automatically. These tests lock in that relationship and the two — and only two
— sanctioned differences (``index_name`` default + the ``retrain_max_age``
field), so a future asymmetric edit fails loudly.
"""

import pandas as pd

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.configurator.config_multi import ConfigMulti

# Feature flags that previously had to be hand-mirrored (regression guard).
INHERITED_FEATURE_FLAGS = (
    "use_population_weighted_weather",
    "include_degree_hours",
    "include_apparent_temperature",
    "include_ephemeris_features",
    "include_day_type_features",
)


class TestStructuralParity:
    def test_entsoe_is_configmulti_subclass(self):
        assert issubclass(ConfigEntsoe, ConfigMulti)
        assert isinstance(ConfigEntsoe(), ConfigMulti)

    def test_entsoe_is_superset_plus_only_known_extra(self):
        multi = set(ConfigMulti._PARAM_NAMES)
        entsoe = set(ConfigEntsoe._PARAM_NAMES)
        # No ConfigMulti field can be missing from ConfigEntsoe.
        assert multi <= entsoe
        # The only sanctioned ENTSO-E-specific field.
        assert entsoe - multi == {"retrain_max_age"}

    def test_feature_flags_inherited(self):
        cfg = ConfigEntsoe()
        for flag in INHERITED_FEATURE_FLAGS:
            assert hasattr(cfg, flag), flag
            assert getattr(cfg, flag) is False, flag

    def test_constructs_with_inherited_flags(self):
        cfg = ConfigEntsoe(
            use_population_weighted_weather=True,
            include_degree_hours=True,
            include_apparent_temperature=True,
            include_ephemeris_features=True,
            include_day_type_features=True,
        )
        assert cfg.include_ephemeris_features is True
        assert cfg.include_day_type_features is True


class TestSanctionedDifferences:
    def test_index_name_default_override(self):
        assert ConfigMulti().index_name == "DateTime"
        assert ConfigEntsoe().index_name == "Time (UTC)"

    def test_retrain_max_age(self):
        assert ConfigEntsoe().retrain_max_age == pd.Timedelta(days=7)
        assert not hasattr(ConfigMulti(), "retrain_max_age")

    def test_shared_field_defaults_identical(self):
        """Every shared field has the same default on both classes (only
        index_name is allowed to differ)."""
        multi, entsoe = ConfigMulti(), ConfigEntsoe()
        shared = set(ConfigMulti._PARAM_NAMES) & set(ConfigEntsoe._PARAM_NAMES)
        diffs = {
            k for k in shared if repr(getattr(multi, k)) != repr(getattr(entsoe, k))
        }
        assert diffs == {"index_name"}, diffs


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

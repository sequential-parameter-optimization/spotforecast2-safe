# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-safe hyperparameter validation for the pipeline config classes.

Clearly-invalid values must raise at construction AND via set_params (the latter
previously bypassed the one existing poly_features_degree check). Validation is
deliberately conservative — only unambiguously-invalid values are rejected.
"""

import pytest

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.configurator.config_multi import ConfigMulti

CLASSES = [ConfigEntsoe, ConfigMulti]


@pytest.mark.parametrize("cls", CLASSES)
class TestConstructorValidation:
    def test_predict_size_non_positive(self, cls):
        with pytest.raises(ValueError, match="predict_size must be positive"):
            cls(predict_size=0)

    def test_contamination_out_of_range(self, cls):
        with pytest.raises(ValueError, match="contamination must be"):
            cls(contamination=0.6)

    def test_poly_features_degree_below_one(self, cls):
        with pytest.raises(ValueError, match="poly_features_degree must be"):
            cls(poly_features_degree=0)

    def test_on_weather_failure_invalid(self, cls):
        with pytest.raises(ValueError, match="on_weather_failure must be"):
            cls(on_weather_failure="bogus")

    def test_valid_boundaries_accepted(self, cls):
        cfg = cls(
            predict_size=1,
            contamination=0.0,
            poly_features_degree=1,
            on_weather_failure="skip",
        )
        assert cfg.predict_size == 1
        assert cfg.contamination == 0.0


@pytest.mark.parametrize("cls", CLASSES)
class TestSetParamsValidation:
    def test_rejects_invalid_predict_size(self, cls):
        with pytest.raises(ValueError, match="predict_size must be positive"):
            cls().set_params(predict_size=-1)

    def test_rejects_invalid_poly_degree(self, cls):
        # set_params previously bypassed this guard entirely.
        with pytest.raises(ValueError, match="poly_features_degree must be"):
            cls().set_params(poly_features_degree=0)

    def test_rejects_invalid_on_weather_failure(self, cls):
        with pytest.raises(ValueError, match="on_weather_failure must be"):
            cls().set_params(on_weather_failure="nope")

    def test_valid_set_params_still_works(self, cls):
        cfg = cls().set_params(predict_size=48, contamination=0.05)
        assert cfg.predict_size == 48
        assert cfg.contamination == 0.05

    def test_negative_exog_max_gap_rejected(self, cls):
        with pytest.raises(ValueError, match="exog_max_gap_hours must be >= 0"):
            cls().set_params(exog_max_gap_hours=-1)

    def test_negative_exog_max_tail_gap_rejected(self, cls):
        with pytest.raises(ValueError, match="exog_max_tail_gap_hours must be >= 0"):
            cls().set_params(exog_max_tail_gap_hours=-1)

    def test_bogus_exog_provider_window_rejected(self, cls):
        with pytest.raises(ValueError, match="exog_provider_window must be"):
            cls(exog_provider_window="yesterday")

    def test_valid_exog_gap_accepted(self, cls):
        cfg = cls(exog_max_gap_hours=0, exog_provider_window="full")
        assert cfg.exog_max_gap_hours == 0
        cfg2 = cls(exog_max_gap_hours=12, exog_provider_window="train")
        assert cfg2.exog_max_gap_hours == 12
        assert cfg2.exog_provider_window == "train"

    def test_inert_tail_gap_warns(self, cls, caplog):
        """0 < exog_max_tail_gap_hours <= exog_max_gap_hours triggers a warning."""
        import logging

        with caplog.at_level(
            logging.WARNING, logger="spotforecast2_safe.configurator._base_config"
        ):
            cfg = cls(exog_max_gap_hours=6, exog_max_tail_gap_hours=3)
        assert cfg.exog_max_tail_gap_hours == 3
        assert any(
            "inert" in r.message for r in caplog.records
        ), f"Expected 'inert' in warning; got: {caplog.records!r}"

    def test_valid_tail_gap_larger_than_interior_accepted(self, cls):
        """exog_max_tail_gap_hours=48 with exog_max_gap_hours=3 is valid (no warning)."""
        # Construct without raising or warning.
        cfg = cls(exog_max_gap_hours=3, exog_max_tail_gap_hours=48)
        assert cfg.exog_max_gap_hours == 3
        assert cfg.exog_max_tail_gap_hours == 48

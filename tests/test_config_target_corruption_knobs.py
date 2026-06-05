# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the target-corruption-detector config knobs on both config classes.

Follows the pattern of ``tests/test_config_poly_mi_knobs.py``.
"""

import pytest

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.configurator.config_multi import ConfigMulti


def _entsoe(**kw):
    return ConfigEntsoe(**kw)


def _multi(**kw):
    return ConfigMulti(**kw)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_defaults(factory):
    cfg = factory()
    assert cfg.target_qc_range_mw is None
    assert cfg.target_qc_step_mw is None
    assert cfg.target_qc_window_days is None
    assert cfg.target_corruption_policy == "abort"
    assert cfg.target_max_heal_hours == 0
    assert cfg.target_anchor_zone_hours == 168


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_explicit_values_round_trip(factory):
    cfg = factory(
        target_qc_range_mw=5_000.0,
        target_qc_step_mw=8_000.0,
        target_qc_window_days=7,
        target_corruption_policy="truncate",
        target_max_heal_hours=24,
        target_anchor_zone_hours=48,
    )
    assert cfg.target_qc_range_mw == 5_000.0
    assert cfg.target_qc_step_mw == 8_000.0
    assert cfg.target_qc_window_days == 7
    assert cfg.target_corruption_policy == "truncate"
    assert cfg.target_max_heal_hours == 24
    assert cfg.target_anchor_zone_hours == 48


# ---------------------------------------------------------------------------
# set_params round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_set_params_round_trip(factory):
    cfg = factory()
    cfg.set_params(
        target_qc_range_mw=3_000.0,
        target_qc_step_mw=6_000.0,
        target_qc_window_days=5,
        target_corruption_policy="heal",
        target_max_heal_hours=12,
        target_anchor_zone_hours=72,
    )
    assert cfg.target_qc_range_mw == 3_000.0
    assert cfg.target_qc_step_mw == 6_000.0
    assert cfg.target_qc_window_days == 5
    assert cfg.target_corruption_policy == "heal"
    assert cfg.target_max_heal_hours == 12
    assert cfg.target_anchor_zone_hours == 72


# ---------------------------------------------------------------------------
# Validation: accept valid values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_entsoe, _multi])
@pytest.mark.parametrize("policy", ["abort", "heal", "truncate"])
def test_valid_policy(factory, policy):
    cfg = factory(target_corruption_policy=policy)
    assert cfg.target_corruption_policy == policy


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_zero_range_mw_accepted(factory):
    cfg = factory(target_qc_range_mw=0.0)
    assert cfg.target_qc_range_mw == 0.0


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_zero_step_mw_accepted(factory):
    cfg = factory(target_qc_step_mw=0.0)
    assert cfg.target_qc_step_mw == 0.0


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_window_days_1_accepted(factory):
    cfg = factory(target_qc_window_days=1)
    assert cfg.target_qc_window_days == 1


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_zero_anchor_zone_accepted(factory):
    cfg = factory(target_anchor_zone_hours=0)
    assert cfg.target_anchor_zone_hours == 0


# ---------------------------------------------------------------------------
# Validation: reject invalid values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_negative_range_mw_rejected(factory):
    with pytest.raises(ValueError, match="target_qc_range_mw"):
        factory(target_qc_range_mw=-1.0)


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_negative_step_mw_rejected(factory):
    with pytest.raises(ValueError, match="target_qc_step_mw"):
        factory(target_qc_step_mw=-100.0)


@pytest.mark.parametrize("factory", [_entsoe, _multi])
@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_window_days_rejected(factory, bad):
    with pytest.raises(ValueError, match="target_qc_window_days"):
        factory(target_qc_window_days=bad)


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_invalid_policy_rejected(factory):
    with pytest.raises(ValueError, match="target_corruption_policy"):
        factory(target_corruption_policy="ignore")


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_negative_max_heal_hours_rejected(factory):
    with pytest.raises(ValueError, match="target_max_heal_hours"):
        factory(target_max_heal_hours=-1)


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_negative_anchor_zone_rejected(factory):
    with pytest.raises(ValueError, match="target_anchor_zone_hours"):
        factory(target_anchor_zone_hours=-5)


# ---------------------------------------------------------------------------
# get_params includes new knobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_entsoe, _multi])
def test_get_params_includes_knobs(factory):
    cfg = factory(target_qc_range_mw=4_000.0)
    p = cfg.get_params()
    assert "target_qc_range_mw" in p
    assert "target_qc_step_mw" in p
    assert "target_qc_window_days" in p
    assert "target_corruption_policy" in p
    assert "target_max_heal_hours" in p
    assert "target_anchor_zone_hours" in p
    assert p["target_qc_range_mw"] == 4_000.0

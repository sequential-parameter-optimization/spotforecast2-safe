# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the poly-MI performance knobs on the config classes.

``poly_mi_n_jobs`` / ``poly_mi_sample_size`` tune the mutual-information
ranking that enforces ``max_poly_features`` (see
``manager.features.select_top_poly_features``). Both configs must expose them
with the same defaults, and ``validate_config`` must reject a non-positive
sample size (fail-safe).
"""

import pytest

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.configurator.config_multi import ConfigMulti


def _minimal_entsoe(**overrides):
    return ConfigEntsoe(**overrides)


def _minimal_multi(**overrides):
    return ConfigMulti(**overrides)


@pytest.mark.parametrize("factory", [_minimal_entsoe, _minimal_multi])
def test_defaults(factory):
    cfg = factory()
    assert cfg.poly_mi_n_jobs == -1
    assert cfg.poly_mi_sample_size == 4000


@pytest.mark.parametrize("factory", [_minimal_entsoe, _minimal_multi])
def test_explicit_values_round_trip(factory):
    cfg = factory(poly_mi_n_jobs=2, poly_mi_sample_size=10_000)
    assert cfg.poly_mi_n_jobs == 2
    assert cfg.poly_mi_sample_size == 10_000


@pytest.mark.parametrize("factory", [_minimal_entsoe, _minimal_multi])
def test_none_disables_subsampling_and_parallelism(factory):
    cfg = factory(poly_mi_n_jobs=None, poly_mi_sample_size=None)
    assert cfg.poly_mi_n_jobs is None
    assert cfg.poly_mi_sample_size is None


@pytest.mark.parametrize("factory", [_minimal_entsoe, _minimal_multi])
@pytest.mark.parametrize("bad", [0, -10])
def test_non_positive_sample_size_rejected(factory, bad):
    with pytest.raises(ValueError, match="poly_mi_sample_size"):
        factory(poly_mi_sample_size=bad)

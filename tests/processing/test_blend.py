# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for processing.blend.blend_with_prior (pure convex blend)."""

import pandas as pd
import pytest

from spotforecast2_safe.processing.blend import blend_with_prior

IDX = pd.date_range("2026-06-13 00:00", periods=4, freq="h", tz="UTC")
MODEL = pd.Series([100.0, 110.0, 120.0, 130.0], index=IDX, name="y0")
PRIOR = pd.Series([140.0, 140.0, 140.0, 140.0], index=IDX)


class TestBlendWithPrior:
    def test_weight_zero_returns_model(self):
        out = blend_with_prior(MODEL, PRIOR, weight=0.0)
        pd.testing.assert_series_equal(out, MODEL)

    def test_weight_one_returns_prior_values(self):
        out = blend_with_prior(MODEL, PRIOR, weight=1.0)
        assert out.tolist() == PRIOR.tolist()

    def test_half_weight_is_midpoint(self):
        out = blend_with_prior(MODEL, PRIOR, weight=0.5)
        assert out.tolist() == [120.0, 125.0, 130.0, 135.0]

    def test_name_preserved_from_model(self):
        assert blend_with_prior(MODEL, PRIOR, weight=0.3).name == "y0"

    def test_intersection_only(self):
        prior_short = PRIOR.iloc[1:]
        out = blend_with_prior(MODEL, prior_short, weight=0.5)
        assert out.index.equals(prior_short.index)
        assert len(out) == 3

    def test_does_not_mutate_inputs(self):
        m_before, p_before = MODEL.copy(), PRIOR.copy()
        blend_with_prior(MODEL, PRIOR, weight=0.4)
        pd.testing.assert_series_equal(MODEL, m_before)
        pd.testing.assert_series_equal(PRIOR, p_before)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_weight_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            blend_with_prior(MODEL, PRIOR, weight=bad)

    def test_non_series_model_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            blend_with_prior([1, 2], PRIOR, weight=0.5)  # type: ignore[arg-type]

    def test_non_series_prior_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            blend_with_prior(MODEL, [1, 2], weight=0.5)  # type: ignore[arg-type]

    def test_disjoint_index_raises(self):
        other = pd.Series(
            [1.0, 2.0],
            index=pd.date_range("2027-01-01", periods=2, freq="h", tz="UTC"),
        )
        with pytest.raises(ValueError, match="no index positions"):
            blend_with_prior(MODEL, other, weight=0.5)

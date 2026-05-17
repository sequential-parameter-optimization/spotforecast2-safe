# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2_safe.security.masking.

Validates the mask_estimator helper used to keep estimator details out of
production logs (CWE-532, CWE-312).
"""

from lightgbm import LGBMRegressor

from spotforecast2_safe.security.masking import mask_estimator


class _DummyEstimator:
    """Minimal stand-in to verify object-branch handling without importing scikit-learn."""


class TestMaskEstimator:
    """Test suite for mask_estimator."""

    def test_none_returns_default_label(self):
        """None must yield the documented project-default label."""
        assert mask_estimator(None) == "LGBMRegressor (default)"

    def test_string_returned_verbatim(self):
        """A string input is treated as a pre-masked label and passed through."""
        assert mask_estimator("ForecasterRecursive") == "ForecasterRecursive"

    def test_empty_string_returned_verbatim(self):
        """Empty strings are also returned as-is; no implicit fallback."""
        assert mask_estimator("") == ""

    def test_dummy_object_returns_type_name(self):
        """Arbitrary objects collapse to their class name only."""
        assert mask_estimator(_DummyEstimator()) == "_DummyEstimator"

    def test_lgbm_instance_returns_type_name(self):
        """A real LGBMRegressor instance never leaks its hyperparameters."""
        result = mask_estimator(LGBMRegressor(n_estimators=200, learning_rate=0.01))
        assert result == "LGBMRegressor"

    def test_hyperparameters_never_leak(self):
        """Sanity check that no hyperparameter values appear in the mask."""
        masked = mask_estimator(LGBMRegressor(n_estimators=12345, max_depth=99))
        assert "12345" not in masked
        assert "99" not in masked
        assert "max_depth" not in masked

    def test_deterministic_repeated_calls(self):
        """Same input must produce the same masked output across calls."""
        est = LGBMRegressor()
        assert mask_estimator(est) == mask_estimator(est)


class TestMaskEstimatorReExport:
    """The package-level re-export must remain stable for downstream imports."""

    def test_reexport_from_security_init(self):
        from spotforecast2_safe import security

        assert security.mask_estimator(None) == "LGBMRegressor (default)"

    def test_all_lists_mask_estimator(self):
        from spotforecast2_safe.security import __all__ as security_all
        from spotforecast2_safe.security.masking import __all__ as masking_all

        assert "mask_estimator" in security_all
        assert "mask_estimator" in masking_all

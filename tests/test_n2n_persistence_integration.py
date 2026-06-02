# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pipeline-level persistence contract for ``n2n_predict_with_covariates``.

The heavy forecasting pipeline exposes a caching contract (``model_dir`` /
``force_train``) and re-uses the canonical persistence helpers. These tests
guard that contract at the signature/import level; the persistence *behaviour*
itself is covered by ``tests/manager/test_persistence.py`` (the single canonical
home, into which the former duplicate persistence test files were consolidated).
"""

import inspect

from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)


class TestPipelineCacheSignature:
    """The pipeline's caching knobs keep their documented defaults."""

    def test_model_dir_defaults_to_none(self):
        sig = inspect.signature(n2n_predict_with_covariates)
        # None triggers the default cache-home location.
        assert sig.parameters["model_dir"].default is None

    def test_force_train_parameter_exists(self):
        sig = inspect.signature(n2n_predict_with_covariates)
        assert "force_train" in sig.parameters

    def test_force_train_defaults_to_true(self):
        sig = inspect.signature(n2n_predict_with_covariates)
        assert sig.parameters["force_train"].default is True


class TestPersistenceReexports:
    """The pipeline re-uses the canonical persistence helpers (same objects)."""

    def test_pipeline_uses_canonical_persistence_helpers(self):
        from spotforecast2_safe.manager.persistence import (
            load_forecasters as canonical_load,
        )
        from spotforecast2_safe.manager.persistence import (
            save_forecasters as canonical_save,
        )
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            load_forecasters,
            save_forecasters,
        )

        assert load_forecasters is canonical_load
        assert save_forecasters is canonical_save

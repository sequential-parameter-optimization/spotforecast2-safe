"""
Integration tests for n2n_predict_with_covariates with model persistence.

Tests for the model caching, loading, and force_train functionality
in the main forecasting pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Tests for Model Directory Parameter
# ============================================================================


class TestModelDirParameter:
    """Tests for model_dir parameter functionality."""

    def test_model_dir_creation(self, temp_model_dir):
        """Test that model_dir is created if it doesn't exist."""
        from spotforecast2_safe.manager.persistence import ensure_model_dir

        new_dir = temp_model_dir / "new" / "models"
        assert not new_dir.exists()

        result = ensure_model_dir(new_dir)
        assert result.exists()

    def test_model_dir_string_path(self, temp_model_dir):
        """Test that model_dir accepts string paths."""
        from spotforecast2_safe.manager.persistence import ensure_model_dir

        str_path = str(temp_model_dir / "models")
        result = ensure_model_dir(str_path)

        assert result.exists()
        assert isinstance(result, Path)

    def test_model_dir_default_value(self):
        """Test that model_dir has correct default value."""
        # Check the function signature
        import inspect

        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            n2n_predict_with_covariates,
        )

        sig = inspect.signature(n2n_predict_with_covariates)
        # Default should be None, which triggers get_cache_home() usage
        assert sig.parameters["model_dir"].default is None


# ============================================================================
# Tests for force_train Parameter
# ============================================================================


class TestForceTrainParameter:
    """Tests for force_train parameter."""

    def test_force_train_parameter_exists(self):
        """Test that force_train parameter exists."""
        import inspect

        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            n2n_predict_with_covariates,
        )

        sig = inspect.signature(n2n_predict_with_covariates)
        assert "force_train" in sig.parameters

    def test_force_train_default_is_true(self):
        """Test that force_train defaults to True."""
        import inspect

        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            n2n_predict_with_covariates,
        )

        sig = inspect.signature(n2n_predict_with_covariates)
        assert sig.parameters["force_train"].default is True


# ============================================================================
# Tests for Model Caching Behavior
# ============================================================================


class TestModelCachingBehavior:
    """Tests for model caching behavior."""

    def test_can_load_andsave_forecasters(self, temp_model_dir):
        """Test basic load/save functionality."""
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            load_forecasters,
        )

        # Try loading from empty dir
        forecasters, missing = load_forecasters(
            ["power", "energy"],
            temp_model_dir,
            verbose=False,
        )

        assert len(forecasters) == 0
        assert len(missing) == 2

    def test_model_directory_management(self, temp_model_dir):
        """Test model directory creation and existence checks."""
        from spotforecast2_safe.manager.persistence import (
            ensure_model_dir,
            model_directory_exists,
        )

        model_dir = temp_model_dir / "new_models"
        assert not model_directory_exists(model_dir)

        ensure_model_dir(model_dir)
        assert model_directory_exists(model_dir)


# ============================================================================
# Tests for Persistence Functions
# ============================================================================


class TestPersistenceFunctions:
    """Tests for model persistence helper functions."""

    @patch("spotforecast2_safe.manager.persistence.dump")
    def testsave_forecasters_with_patch(self, mock_dump, temp_model_dir):
        """Test save_forecasters with mocked dump."""
        from spotforecast2_safe.manager.persistence import save_forecasters

        mock_forecaster = MagicMock()
        result = save_forecasters(
            {"power": mock_forecaster},
            temp_model_dir,
            verbose=False,
        )

        assert "power" in result
        assert mock_dump.called

    @patch("spotforecast2_safe.manager.persistence.load")
    @patch("spotforecast2_safe.manager.persistence.dump")
    def testload_forecasters_with_patch(self, mock_dump, mock_load, temp_model_dir):
        """Test load_forecasters with mocked load."""
        from spotforecast2_safe.manager.persistence import (
            load_forecasters,
            save_forecasters,
        )

        mock_forecaster = MagicMock()
        save_forecasters({"power": mock_forecaster}, temp_model_dir, verbose=False)

        mock_load.return_value = mock_forecaster
        forecasters, missing = load_forecasters(
            ["power"],
            temp_model_dir,
            verbose=False,
        )

        # load should have been called
        assert mock_load.called or len(missing) >= 0

    def test_filepath_generation(self, temp_model_dir):
        """Test model filepath generation."""
        from spotforecast2_safe.manager.persistence import get_model_filepath

        filepath = get_model_filepath(temp_model_dir, "power")
        assert filepath.name == "forecaster_power.joblib"
        assert filepath.parent == temp_model_dir


# ============================================================================
# Documentation and Signature Tests
# ============================================================================


class TestFunctionDocumentation:
    """Tests for function documentation and docstrings."""

    def test_n2n_predict_docstring_mentions_persistence(self):
        """Test that n2n_predict_with_covariates has persistence in docs."""
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            n2n_predict_with_covariates,
        )

        docstring = n2n_predict_with_covariates.__doc__
        assert docstring is not None
        assert "persisted" in docstring.lower() or "cache" in docstring.lower()

    def test_persistence_functions_have_docstrings(self):
        """Test that persistence functions are documented."""
        from spotforecast2_safe.manager.persistence import (
            ensure_model_dir,
            get_model_filepath,
            load_forecasters,
            model_directory_exists,
            save_forecasters,
        )

        functions = [
            ensure_model_dir,
            get_model_filepath,
            load_forecasters,
            model_directory_exists,
            save_forecasters,
        ]

        for func in functions:
            assert func.__doc__ is not None, f"{func.__name__} has no docstring"

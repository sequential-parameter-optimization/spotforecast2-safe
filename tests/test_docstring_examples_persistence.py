# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests to validate docstring examples for the persistence module.

Uses pytest-doctest to verify that all Examples in the docstrings are correct
and self-contained with all required imports.
"""

import doctest
import pytest

from spotforecast2_safe.manager import persistence


def test_ensure_model_dir_docstring_examples():
    """Test that _ensure_model_dir docstring examples are valid."""
    results = doctest.testmod(
        persistence,
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        extraglobs={"_ensure_model_dir": persistence._ensure_model_dir},
    )
    # This tests all module examples at once
    # We'll add specific tests below for better error reporting


def test_ensure_model_dir_example():
    """Validate _ensure_model_dir example is runnable."""
    import tempfile
    from pathlib import Path

    from spotforecast2_safe.manager.persistence import _ensure_model_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _ensure_model_dir(Path(tmpdir) / "models")
        assert model_path.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        nested = _ensure_model_dir(Path(tmpdir) / "a" / "b" / "c")
        assert nested.is_dir()


def test_get_model_filepath_example():
    """Validate _get_model_filepath example is runnable."""
    from pathlib import Path

    from spotforecast2_safe.manager.persistence import _get_model_filepath

    path = _get_model_filepath(Path("./models"), "power")
    assert str(path) == "models/forecaster_power.joblib"

    path = _get_model_filepath(Path("/tmp/models"), "energy")
    assert path.name == "forecaster_energy.joblib"
    assert path.suffix == ".joblib"


def test_save_forecasters_example():
    """Validate _save_forecasters example is runnable."""
    import tempfile
    from pathlib import Path

    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.manager.persistence import _save_forecasters

    mock_model = LinearRegression()

    with tempfile.TemporaryDirectory() as tmpdir:
        forecasters = {"power": mock_model, "energy": mock_model}
        paths = _save_forecasters(forecasters, tmpdir, verbose=False)
        assert "power" in paths
        assert paths["power"].exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = _save_forecasters({"demand": mock_model}, tmpdir)
        assert paths["demand"].suffix == ".joblib"


def test_load_forecasters_example():
    """Validate _load_forecasters example is runnable."""
    import tempfile
    from pathlib import Path

    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.manager.persistence import (
        _load_forecasters,
        _save_forecasters,
    )

    mock_model = LinearRegression()

    with tempfile.TemporaryDirectory() as tmpdir:
        _ = _save_forecasters({"power": mock_model}, tmpdir)
        forecasters, missing = _load_forecasters(
            ["power", "energy"],
            tmpdir,
            verbose=False,
        )
        assert "power" in forecasters
        assert "energy" in missing

    with tempfile.TemporaryDirectory() as tmpdir:
        forecasters, missing = _load_forecasters(["nonexistent"], tmpdir)
        assert len(forecasters) == 0
        assert len(missing) == 1


def test_model_directory_exists_example():
    """Validate _model_directory_exists example is runnable."""
    import tempfile
    from pathlib import Path

    from spotforecast2_safe.manager.persistence import _model_directory_exists

    with tempfile.TemporaryDirectory() as tmpdir:
        assert _model_directory_exists(tmpdir) is True

    assert _model_directory_exists("/nonexistent/path/to/models") is False


def test_all_docstring_examples_via_doctest():
    """Run all docstring examples via doctest module."""
    results = doctest.testmod(
        persistence,
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert results.failed == 0, f"Doctest failures: {results.failed}"

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Smoke tests for the persistence module public helpers.

The legacy `>>>`-style docstring examples were converted to live
`{python}` Quarto cells (executed at `quarto render` time), so this
file only carries direct call-based assertions for each renamed public
helper.
"""


def test_ensure_model_dir_example():
    """Validate ensure_model_dir example is runnable."""
    import tempfile
    from pathlib import Path

    from spotforecast2_safe.manager.persistence import ensure_model_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = ensure_model_dir(Path(tmpdir) / "models")
        assert model_path.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        nested = ensure_model_dir(Path(tmpdir) / "a" / "b" / "c")
        assert nested.is_dir()


def test_get_model_filepath_example():
    """Validate get_model_filepath example is runnable."""
    from pathlib import Path

    from spotforecast2_safe.manager.persistence import get_model_filepath

    path = get_model_filepath(Path("./models"), "power")
    assert str(path) == "models/forecaster_power.joblib"

    path = get_model_filepath(Path("/tmp/models"), "energy")
    assert path.name == "forecaster_energy.joblib"
    assert path.suffix == ".joblib"


def test_save_forecasters_example():
    """Validate save_forecasters example is runnable."""
    import tempfile

    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.manager.persistence import save_forecasters

    mock_model = LinearRegression()

    with tempfile.TemporaryDirectory() as tmpdir:
        forecasters = {"power": mock_model, "energy": mock_model}
        paths = save_forecasters(forecasters, tmpdir, verbose=False)
        assert "power" in paths
        assert paths["power"].exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_forecasters({"demand": mock_model}, tmpdir)
        assert paths["demand"].suffix == ".joblib"


def test_load_forecasters_example():
    """Validate load_forecasters example is runnable."""
    import tempfile

    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.manager.persistence import (
        load_forecasters,
        save_forecasters,
    )

    mock_model = LinearRegression()

    with tempfile.TemporaryDirectory() as tmpdir:
        _ = save_forecasters({"power": mock_model}, tmpdir)
        forecasters, missing = load_forecasters(
            ["power", "energy"],
            tmpdir,
            verbose=False,
        )
        assert "power" in forecasters
        assert "energy" in missing

    with tempfile.TemporaryDirectory() as tmpdir:
        forecasters, missing = load_forecasters(["nonexistent"], tmpdir)
        assert len(forecasters) == 0
        assert len(missing) == 1


def test_model_directory_exists_example():
    """Validate model_directory_exists example is runnable."""
    import tempfile

    from spotforecast2_safe.manager.persistence import model_directory_exists

    with tempfile.TemporaryDirectory() as tmpdir:
        assert model_directory_exists(tmpdir) is True

    assert model_directory_exists("/nonexistent/path/to/models") is False

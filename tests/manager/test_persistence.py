# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``manager.persistence`` save/load.

Covers the public single-model ``save_forecaster`` (default and task-name
filenames) and the safety-critical load contract: a *missing* model file is
reported for retraining, but a model file that exists yet fails to deserialise
raises ``OSError`` so the caller cannot silently retrain over a corrupt model.

This is the canonical home for the persistence-helper contract; it absorbs the
real coverage that previously lived in the duplicate ``test_model_persistence``
and ``test_n2n_predict_persistence`` files (helpers ``ensure_model_dir`` /
``get_model_filepath`` / ``model_directory_exists`` and the multi-model
``save_forecasters`` / ``load_forecasters`` paths), with real assertions instead
of MagicMock tautologies.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.manager.persistence import (
    ensure_model_dir,
    get_model_filepath,
    load_forecasters,
    model_directory_exists,
    save_forecaster,
    save_forecasters,
)


def test_save_forecaster_default_name(tmp_path):
    path = save_forecaster(LinearRegression(), tmp_path, "power")
    assert path.name == "forecaster_power.joblib"
    assert path.exists()


def test_save_forecaster_with_task_name(tmp_path):
    path = save_forecaster(LinearRegression(), tmp_path, "power", task_name="task_1")
    assert path.name == "task_1_power.joblib"
    assert path.exists()


def test_load_forecasters_missing_is_reported_not_raised(tmp_path):
    forecasters, missing = load_forecasters(["nope"], tmp_path)
    assert forecasters == {}
    assert missing == ["nope"]


def test_load_forecasters_corrupt_file_raises_oserror(tmp_path):
    # A model file that exists but cannot be deserialised must raise, so the
    # caller cannot silently retrain over a corrupt model.
    bad = get_model_filepath(tmp_path, "power")
    bad.write_bytes(b"this is not a valid joblib payload")
    with pytest.raises(OSError, match="Failed to load"):
        load_forecasters(["power"], tmp_path)


def test_round_trip_save_and_load(tmp_path):
    model = LinearRegression().fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
    save_forecaster(model, tmp_path, "power")

    forecasters, missing = load_forecasters(["power"], tmp_path)
    assert "power" in forecasters
    assert missing == []
    assert isinstance(forecasters["power"], LinearRegression)


def test_save_forecaster_non_serializable_raises_typeerror(tmp_path):
    # A non-serializable object must raise the documented TypeError (distinct
    # from a disk OSError), not be masked as OSError.
    with pytest.raises(TypeError, match="not serializable"):
        save_forecaster(lambda x: x, tmp_path, "power")


def test_save_forecasters_non_serializable_raises_typeerror(tmp_path):
    with pytest.raises(TypeError, match="not serializable"):
        save_forecasters({"power": lambda x: x}, tmp_path)


# ---------------------------------------------------------------------------
# Helper-function contract (consolidated from removed duplicate test files).
# ---------------------------------------------------------------------------


class TestEnsureModelDir:
    """``ensure_model_dir`` creates the target and returns it as a ``Path``."""

    def test_creates_new_directory(self, tmp_path):
        new_dir = tmp_path / "models"
        assert not new_dir.exists()

        result = ensure_model_dir(new_dir)

        assert result == new_dir
        assert result.is_dir()

    def test_creates_nested_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"

        result = ensure_model_dir(nested)

        assert result.is_dir()

    def test_existing_directory_is_returned(self, tmp_path):
        assert ensure_model_dir(tmp_path) == tmp_path

    def test_accepts_string_path(self, tmp_path):
        result = ensure_model_dir(str(tmp_path / "models"))

        assert isinstance(result, Path)
        assert result.is_dir()


class TestGetModelFilepath:
    """``get_model_filepath`` follows the ``forecaster_{target}.joblib`` rule."""

    def test_filename_format(self, tmp_path):
        fp = get_model_filepath(tmp_path, "power")

        assert fp.name == "forecaster_power.joblib"
        assert fp.suffix == ".joblib"
        assert fp.parent == tmp_path

    def test_distinct_targets_give_distinct_paths(self, tmp_path):
        assert get_model_filepath(tmp_path, "power") != get_model_filepath(
            tmp_path, "energy"
        )


class TestModelDirectoryExists:
    """``model_directory_exists`` is a thin, string-tolerant existence check."""

    def test_true_for_existing_directory(self, tmp_path):
        assert model_directory_exists(tmp_path) is True

    def test_false_for_missing_directory(self, tmp_path):
        assert model_directory_exists(tmp_path / "nope") is False

    def test_accepts_string_path(self, tmp_path):
        assert model_directory_exists(str(tmp_path)) is True


class TestSaveForecastersMultiModel:
    """Multi-model ``save_forecasters`` happy paths and disk-failure contract."""

    def test_saves_multiple_and_returns_existing_paths(self, tmp_path):
        models = {"power": LinearRegression(), "energy": LinearRegression()}

        paths = save_forecasters(models, tmp_path)

        assert set(paths) == {"power", "energy"}
        assert all(p.exists() for p in paths.values())
        assert paths["power"].name == "forecaster_power.joblib"

    def test_creates_missing_directory(self, tmp_path):
        target = tmp_path / "new" / "models"
        assert not target.exists()

        save_forecasters({"power": LinearRegression()}, target)

        assert target.is_dir()

    def test_verbose_prints_confirmation(self, tmp_path, capsys):
        save_forecasters({"power": LinearRegression()}, tmp_path, verbose=True)

        assert "Saved forecaster for power" in capsys.readouterr().out

    def test_accepts_string_path(self, tmp_path):
        paths = save_forecasters({"power": LinearRegression()}, str(tmp_path))

        assert paths["power"].exists()

    @patch(
        "spotforecast2_safe.manager.persistence.dump",
        side_effect=OSError("disk full"),
    )
    def test_disk_failure_raises_oserror(self, _mock_dump, tmp_path):
        with pytest.raises(OSError, match="Failed to write model"):
            save_forecasters({"power": LinearRegression()}, tmp_path)


class TestLoadForecastersEdges:
    """Edge cases for ``load_forecasters`` selective-retraining reporting."""

    def test_empty_target_list(self, tmp_path):
        forecasters, missing = load_forecasters([], tmp_path)

        assert forecasters == {}
        assert missing == []

    def test_nonexistent_directory_reports_all_missing(self, tmp_path):
        forecasters, missing = load_forecasters(["power", "energy"], tmp_path / "nope")

        assert forecasters == {}
        assert set(missing) == {"power", "energy"}

    def test_partial_cache_reports_only_uncached(self, tmp_path):
        save_forecaster(LinearRegression(), tmp_path, "power")

        forecasters, missing = load_forecasters(["power", "energy"], tmp_path)

        assert "power" in forecasters
        assert missing == ["energy"]

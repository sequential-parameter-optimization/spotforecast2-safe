# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``manager.persistence`` save/load.

Covers the public single-model ``save_forecaster`` (default and task-name
filenames) and the safety-critical load contract: a *missing* model file is
reported for retraining, but a model file that exists yet fails to deserialise
raises ``OSError`` so the caller cannot silently retrain over a corrupt model.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.manager.persistence import (
    get_model_filepath,
    load_forecasters,
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

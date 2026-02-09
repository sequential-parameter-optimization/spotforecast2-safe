# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Module for managing model training.
"""

import glob
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from joblib import dump, load

from spotforecast2_safe.data.fetch_data import fetch_data, get_cache_home

logger = logging.getLogger(__name__)


def train_new_model(
    model_class: type,
    n_iteration: int,
    model_name: Optional[str] = None,
    train_size: Optional[pd.Timedelta] = None,
    save_to_file: bool = True,
    model_dir: Optional[Union[str, Path]] = None,
    end_dev: Optional[Union[str, pd.Timestamp]] = None,
) -> Any:
    """
    Train a new forecaster model and optionally save it to disk.

    This function fetches the latest data, calculates the training cutoff,
    initializes a model of the given class, triggers the tuning process,
    and saves the model following the naming convention:
    `{model_name}_forecaster_{n_iteration}.joblib`.

    Args:
        model_class: The class of the forecaster model to train.
            The class should accept `iteration`, `end_dev`, and `train_size`
            in its constructor and provide a `tune()` method.
        n_iteration: The iteration number for this training run.
        train_size: Optional size of the training set as a pandas Timedelta.
            Defaults to None.
        save_to_file: If True, saves the model to disk after training.
            Defaults to True.
        model_dir: Directory where the model should be saved. If None, defaults to
            the library's cache home.
        end_dev: Optional cutoff date for training. If None, it is calculated
            from the latest available data.

    Returns:
        The trained model instance.

    Examples:
        >>> import tempfile
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> from unittest.mock import patch
        >>> from spotforecast2_safe.manager.trainer import train_new_model
        >>>
        >>> # Example 1: Train without saving to file
        >>> class MockModel:  # doctest: +SKIP
        ...     '''Mock model class for testing'''
        ...     def __init__(self, iteration, end_dev, train_size=None):
        ...         self.iteration = iteration
        ...         self.end_dev = end_dev
        ...         self.train_size = train_size
        ...         self.name = 'mock'
        ...     def tune(self):
        ...         '''Simulate tuning process'''
        ...         pass
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     with patch('spotforecast2_safe.manager.trainer.fetch_data') as mock_fetch:
        ...         mock_data = pd.DataFrame(
        ...             {'value': [1, 2, 3]},
        ...             index=pd.date_range('2024-01-01', periods=3, freq='H')
        ...         )
        ...         mock_fetch.return_value = mock_data
        ...         model = train_new_model(
        ...             MockModel,
        ...             n_iteration=0,
        ...             save_to_file=False
        ...         )
        ...         print(f"Model trained: {model is not None}")
        ...         print(f"Model iteration: {model.iteration}")
        Model trained: True
        Model iteration: 0
        >>>
        >>> # Example 2: Train and save to custom directory
        >>> class TestModel:  # doctest: +SKIP
        ...     '''Test model class'''
        ...     def __init__(self, iteration, end_dev, train_size=None):
        ...         self.iteration = iteration
        ...         self.end_dev = end_dev
        ...         self.name = 'test'
        ...     def tune(self):
        ...         pass
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     with patch('spotforecast2_safe.manager.trainer.fetch_data') as mock_fetch:
        ...         mock_fetch.return_value = pd.DataFrame(
        ...             {'value': [10, 20, 30]},
        ...             index=pd.date_range('2024-06-01', periods=3, freq='D')
        ...         )
        ...         model = train_new_model(
        ...             TestModel,
        ...             n_iteration=1,
        ...             save_to_file=True,
        ...             model_dir=tmpdir
        ...         )
        ...         saved_file = Path(tmpdir) / 'test_forecaster_1.joblib'
        ...         print(f"Model saved: {saved_file.exists()}")
        Model saved: True
    """
    logger.info("Training new model (iteration %d)...", n_iteration)

    # Fetch data using the library's utility
    current_data = fetch_data()
    if current_data.empty:
        logger.error("No data fetched. Aborting training.")
        return None

    if end_dev is None:
        latest_idx = current_data.index[-1]
        # Calculate training cutoff. In this implementation, we use data up to one day
        # before the latest recorded index to ensure we have a full day's data for
        # validation or the last training window.
        end_train_cutoff = latest_idx - pd.Timedelta(days=1)
        logger.debug("Latest data index: %s", latest_idx)
        logger.debug("Calculated training cutoff: %s", end_train_cutoff)
    else:
        end_train_cutoff = pd.to_datetime(end_dev, utc=True)
        logger.debug("Using provided training cutoff: %s", end_train_cutoff)

    # Initialize the model instance
    model = model_class(
        iteration=n_iteration, end_dev=end_train_cutoff, train_size=train_size
    )

    # Perform hyperparameter tuning and fitting as implemented in model_class
    logger.info("Starting model tuning...")
    model.tune()
    logger.info("Training and tuning completed for iteration %d.", n_iteration)

    if save_to_file:
        if model_dir is None:
            model_dir = get_cache_home()
        else:
            model_dir = Path(model_dir)

        model_dir.mkdir(parents=True, exist_ok=True)

        # Use provided model_name, or model's 'name' attribute,
        # otherwise use lowercase class name
        if model_name is None:
            model_name = getattr(model, "name", model_class.__name__.lower())
        else:
            # Update model's internal name for consistency
            model.name = model_name

        file_path = model_dir / f"{model_name}_forecaster_{n_iteration}.joblib"

        try:
            dump(model, file_path, compress=3)
            logger.info("Saved model to %s", file_path)
        except Exception as e:
            logger.error("Failed to save model to %s: %s", file_path, e)

    return model


def get_last_model(
    model_name: str, model_dir: Optional[Union[str, Path]] = None
) -> tuple[int, Any]:
    """
    Get the latest trained model from the cache.

    Args:
        model_name: Name of the model (e.g., 'lgbm', 'xgb').
        model_dir: Directory where models are stored. If None, defaults to
            the library's cache home.

    Returns:
        A tuple (iteration, model_instance). If no model is found,
        returns (-1, None).

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.trainer import get_last_model
        >>> from joblib import dump
        >>>
        >>> # Example 1: No model found
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     iteration, model = get_last_model('lgbm', model_dir=tmpdir)
        ...     print(f"Iteration: {iteration}, Model: {model}")
        Iteration: -1, Model: None
        >>>
        >>> # Example 2: Single model exists
        >>> class SimpleModel:  # doctest: +SKIP
        ...     '''Simple model class'''
        ...     def __init__(self, name):
        ...         self.name = name
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     simple_model = SimpleModel('xgb')
        ...     dump(simple_model, model_dir / 'xgb_forecaster_0.joblib')
        ...     iteration, model = get_last_model('xgb', model_dir=model_dir)
        ...     print(f"Found iteration: {iteration}")
        ...     print(f"Model loaded: {model is not None}")
        Found iteration: 0
        Model loaded: True
        >>>
        >>> # Example 3: Multiple iterations - gets latest
        >>> class IterModel:  # doctest: +SKIP
        ...     '''Model with iteration tracking'''
        ...     def __init__(self, iteration):
        ...         self.iteration = iteration
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     for i in range(5):
        ...         iter_model = IterModel(i)
        ...         dump(iter_model, model_dir / f'lgbm_forecaster_{i}.joblib')
        ...     iteration, model = get_last_model('lgbm', model_dir=model_dir)
        ...     print(f"Latest iteration: {iteration}")
        ...     print(f"Model iteration attribute: {model.iteration}")
        Latest iteration: 4
        Model iteration attribute: 4
        >>>
        >>> # Example 4: Safety-critical - verify model integrity
        >>> class SafetyModel:  # doctest: +SKIP
        ...     '''Safety model with validation attributes'''
        ...     def __init__(self):
        ...         self.validated = True
        ...         self.checksum = 'abc123'
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     safety_model = SafetyModel()
        ...     dump(safety_model, model_dir / 'safety_forecaster_1.joblib')
        ...     iteration, model = get_last_model('safety', model_dir=model_dir)
        ...     if model:
        ...         print(f"Model validated: {model.validated}")
        ...         print(f"Checksum present: {hasattr(model, 'checksum')}")
        Model validated: True
        Checksum present: True
    """
    if model_dir is None:
        model_dir = get_cache_home()
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        return -1, None

    list_files = glob.glob(str(model_dir / f"{model_name}_forecaster_*.joblib"))
    if not list_files:
        return -1, None

    searches = [
        re.search(rf"{model_name}_forecaster_(\d+)\.joblib", x) for x in list_files
    ]
    iterations = [int(search.group(1)) for search in searches if search is not None]

    if not iterations:
        return -1, None

    max_iter = max(iterations)
    file_path = model_dir / f"{model_name}_forecaster_{max_iter}.joblib"

    try:
        model = load(file_path)
        return max_iter, model
    except Exception as e:
        logger.error("Failed to load model from %s: %s", file_path, e)
        return -1, None


def handle_training(
    model_class: type,
    model_name: Optional[str] = None,
    model_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    train_size: Optional[pd.Timedelta] = None,
    end_dev: Optional[Union[str, pd.Timestamp]] = None,
) -> None:
    """
    Check if a new model needs to be trained and trigger training if necessary.

    Trains a new model if no model exists, if the existing model is older than
    7 days, or if retraining is forced.

    Args:
        model_class: The class of the forecaster model to train.
        model_name: Name of the model (e.g., 'lgbm'). If None, it is inferred
            from the model_class name.
        model_dir: Directory where models are stored.
        force: If True, force retraining even if the current model is recent.
        train_size: Optional size of the training set.
        end_dev: Optional cutoff date for training.

    Examples:
        >>> import tempfile
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> from unittest.mock import patch
        >>> from spotforecast2_safe.manager.trainer import handle_training
        >>>
        >>> # Example 1: No existing model - triggers training
        >>> class MockModel:  # doctest: +SKIP
        ...     '''Mock model class'''
        ...     def __init__(self, iteration, end_dev, train_size=None):
        ...         self.iteration = iteration
        ...         self.end_dev = end_dev
        ...         self.name = 'test'
        ...     def tune(self):
        ...         pass
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     with patch('spotforecast2_safe.manager.trainer.get_last_model') as mock_get:
        ...         with patch('spotforecast2_safe.manager.trainer.train_new_model') as mock_train:
        ...             mock_get.return_value = (-1, None)
        ...             handle_training(MockModel, model_name='test', model_dir=tmpdir)
        ...             print(f"Training called: {mock_train.called}")
        ...             print(f"Iteration: {mock_train.call_args[0][1]}")
        Training called: True
        Iteration: 0
        >>>
        >>> # Example 2: Recent model exists - no retraining
        >>> class RecentModel:  # doctest: +SKIP
        ...     '''Model with recent training'''
        ...     def __init__(self):
        ...         self.end_dev = pd.Timestamp.now('UTC') - pd.Timedelta(hours=24)
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     with patch('spotforecast2_safe.manager.trainer.get_last_model') as mock_get:
        ...         with patch('spotforecast2_safe.manager.trainer.train_new_model') as mock_train:
        ...             mock_existing = RecentModel()
        ...             mock_get.return_value = (1, mock_existing)
        ...             handle_training(MockModel, model_name='recent', model_dir=tmpdir)
        ...             print(f"Training skipped: {not mock_train.called}")
        Training skipped: True
        >>>
        >>> # Example 3: Old model exists - triggers retraining
        >>> class OldModel:  # doctest: +SKIP
        ...     '''Model with old training'''
        ...     def __init__(self):
        ...         self.end_dev = pd.Timestamp.now('UTC') - pd.Timedelta(days=10)
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     with patch('spotforecast2_safe.manager.trainer.get_last_model') as mock_get:
        ...         with patch('spotforecast2_safe.manager.trainer.train_new_model') as mock_train:
        ...             mock_old = OldModel()
        ...             mock_get.return_value = (2, mock_old)
        ...             handle_training(MockModel, model_name='old', model_dir=tmpdir)
        ...             print(f"Retraining triggered: {mock_train.called}")
        ...             print(f"New iteration: {mock_train.call_args[0][1]}")
        Retraining triggered: True
        New iteration: 3
        >>>
        >>> # Example 4: Force retraining even with recent model
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     with patch('spotforecast2_safe.manager.trainer.get_last_model') as mock_get:
        ...         with patch('spotforecast2_safe.manager.trainer.train_new_model') as mock_train:
        ...             mock_recent = RecentModel()
        ...             mock_get.return_value = (0, mock_recent)
        ...             handle_training(MockModel, model_name='forced', model_dir=tmpdir, force=True)
        ...             print(f"Force training executed: {mock_train.called}")
        Force training executed: True
    """
    if model_name is None:
        model_name = model_class.__name__.lower()

    n_iteration, current_model = get_last_model(model_name, model_dir)

    if current_model is None:
        logger.info("No model found for %s. Training iteration 0...", model_name)
        train_new_model(
            model_class,
            0,
            model_name=model_name,
            train_size=train_size,
            model_dir=model_dir,
            end_dev=end_dev,
        )
        return

    # Check how long since the model has been trained
    # Note: We expect the model instance to have an 'end_dev' attribute
    last_training_date = getattr(current_model, "end_dev", None)
    if last_training_date is None:
        logger.warning(
            "Current model has no 'end_dev' attribute. Cannot determine age. Forcing retraining."
        )
        train_new_model(
            model_class,
            n_iteration + 1,
            model_name=model_name,
            train_size=train_size,
            model_dir=model_dir,
            end_dev=end_dev,
        )
        return

    # Ensure last_training_date is a pandas Timestamp and timezone aware
    last_training_date = pd.to_datetime(last_training_date)
    if last_training_date.tzinfo is None:
        last_training_date = last_training_date.tz_localize("UTC")

    today = pd.Timestamp.now("UTC")
    hours_since_last_training = (today - last_training_date).total_seconds() // 3600

    # Train a new model every seven days (168 hours)
    if hours_since_last_training >= 168 or force:
        logger.info(
            "Model for %s is old enough (%.0f hours) or retraining forced. "
            "Training iteration %d...",
            model_name,
            hours_since_last_training,
            n_iteration + 1,
        )
        train_new_model(
            model_class,
            n_iteration + 1,
            model_name=model_name,
            train_size=train_size,
            model_dir=model_dir,
            end_dev=end_dev,
        )
    else:
        logger.info(
            "The current %s model was trained up to %s (%.0f hours ago). "
            "No retraining necessary.",
            model_name,
            last_training_date,
            hours_since_last_training,
        )

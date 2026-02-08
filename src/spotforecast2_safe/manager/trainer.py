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
    train_size: Optional[pd.Timedelta] = None,
    save_to_file: bool = True,
    model_dir: Optional[Union[str, Path]] = None,
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

    Returns:
        The trained model instance.

    Examples:
        >>> from spotforecast2_safe.manager.trainer import train_new_model
        >>> # Assuming MyLGBMModel is defined correctly
        >>> # model = train_new_model(MyLGBMModel, n_iteration=1)
    """
    logger.info("Training new model (iteration %d)...", n_iteration)

    # Fetch data using the library's utility
    current_data = fetch_data()
    if current_data.empty:
        logger.error("No data fetched. Aborting training.")
        return None

    latest_idx = current_data.index[-1]

    # Calculate training cutoff. In this implementation, we use data up to one day
    # before the latest recorded index to ensure we have a full day's data for
    # validation or the last training window.
    end_train_cutoff = latest_idx - pd.Timedelta(days=1)

    logger.debug("Latest data index: %s", latest_idx)
    logger.debug("Training cutoff: %s", end_train_cutoff)

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

        # Get model name if available, otherwise use lowercase class name
        model_name = getattr(model, "name", model_class.__name__.lower())
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
    """
    if model_name is None:
        model_name = model_class.__name__.lower()

    n_iteration, current_model = get_last_model(model_name, model_dir)

    if current_model is None:
        logger.info("No model found for %s. Training iteration 0...", model_name)
        train_new_model(model_class, 0, train_size=train_size, model_dir=model_dir)
        return

    # Check how long since the model has been trained
    # Note: We expect the model instance to have an 'end_dev' attribute
    last_training_date = getattr(current_model, "end_dev", None)
    if last_training_date is None:
        logger.warning(
            "Current model has no 'end_dev' attribute. Cannot determine age. Forcing retraining."
        )
        train_new_model(
            model_class, n_iteration + 1, train_size=train_size, model_dir=model_dir
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
            model_class, n_iteration + 1, train_size=train_size, model_dir=model_dir
        )
    else:
        logger.info(
            "The current %s model was trained up to %s (%.0f hours ago). "
            "No retraining necessary.",
            model_name,
            last_training_date,
            hours_since_last_training,
        )

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
from spotforecast2_safe.preprocessing import RollingFeatures

logger = logging.getLogger(__name__)


#: Candidate lag values for hyperparameter search.
LAGS_CONSIDER: list[int] = list(range(1, 24))

#: Default rolling window features matching the original chag25a configuration.
#: Each entry is a separate RollingFeatures instance to avoid duplicate-name
#: collisions in spotforecast2-safe's ``initialize_window_features``.
window_features = [
    RollingFeatures(stats="mean", window_sizes=24),
    RollingFeatures(stats="mean", window_sizes=24 * 7),
    RollingFeatures(stats="mean", window_sizes=24 * 30),
    RollingFeatures(stats="min", window_sizes=24),
    RollingFeatures(stats="max", window_sizes=24),
]


def get_path_model(
    name: str,
    iteration: int,
    model_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Yield the path to a model file for a given iteration and model name.

    Args:
        name: Model name (e.g. ``"lgbm"``, ``"xgb"``).
        iteration: Iteration of the model.
        model_dir: Directory where models are stored.
            If *None*, defaults to :func:`get_cache_home`.

    Returns:
        Path: Full path where the model file should be stored.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.trainer import get_path_model
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     p = get_path_model("lgbm", 3, model_dir=tmpdir)
        ...     p.name
        'lgbm_forecaster_3.joblib'
    """
    if model_dir is None:
        model_dir = get_cache_home()
    else:
        model_dir = Path(model_dir)
    return model_dir / f"{name}_forecaster_{iteration}.joblib"


def load_iteration(
    name: str,
    iteration: int,
    model_dir: Optional[Union[str, Path]] = None,
) -> Optional[Any]:
    """Load a saved model at a given iteration.

    Args:
        name: Model name (e.g. ``"lgbm"``).
        iteration: Iteration of the model.
        model_dir: Directory where models are stored.
            If *None*, defaults to :func:`get_cache_home`.

    Returns:
        The loaded model instance, or *None* if the file does not exist.

    Examples:
        >>> import tempfile
        >>> from spotforecast2_safe.manager.trainer import load_iteration
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     result = load_iteration("lgbm", 99, model_dir=tmpdir)
        ...     result is None
        True
    """
    path_file = get_path_model(name, iteration, model_dir=model_dir)
    if not path_file.exists():
        logger.error("Iteration %d does not exist at %s!", iteration, path_file)
        return None
    try:
        model = load(path_file)
        return model
    except Exception as e:
        logger.error("Failed to load model from %s: %s", path_file, e)
        return None


def search_space_lgbm(trial: Any) -> dict:
    """Optuna search space for LightGBM hyperparameters.

    Args:
        trial: An :class:`optuna.trial.Trial` instance.

    Returns:
        dict: Suggested hyperparameters for the current trial.

    Examples:
        >>> from spotforecast2_safe.manager.trainer import search_space_lgbm
        >>> # Without Optuna, verify the function signature exists
        >>> callable(search_space_lgbm)
        True
    """
    search_space = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100),
        "lags": trial.suggest_categorical("lags", LAGS_CONSIDER),
    }
    return search_space


def search_space_xgb(trial: Any) -> dict:
    """Optuna search space for XGBoost hyperparameters.

    Args:
        trial: An :class:`optuna.trial.Trial` instance.

    Returns:
        dict: Suggested hyperparameters for the current trial.

    Examples:
        >>> from spotforecast2_safe.manager.trainer import search_space_xgb
        >>> callable(search_space_xgb)
        True
    """
    search_space = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "n_estimators": trial.suggest_int("n_estimators", 50, 600, step=50),
        "alpha": trial.suggest_float("alpha", 0.0, 0.5),
        "lambda": trial.suggest_float("lambda", 0.0, 0.5),
        "lags": trial.suggest_categorical("lags", LAGS_CONSIDER),
    }
    return search_space


#: Registry mapping model names to their search space functions.
SEARCH_SPACES: dict[str, Any] = {
    "lgbm": search_space_lgbm,
    "xgb": search_space_xgb,
}


def train_new_model(
    model_class: type,
    n_iteration: int,
    model_name: Optional[str] = None,
    train_size: Optional[pd.Timedelta] = None,
    save_to_file: bool = True,
    model_dir: Optional[Union[str, Path]] = None,
    end_dev: Optional[Union[str, pd.Timestamp]] = None,
    data_filename: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Train a new forecaster model and optionally save it to disk.

    This function fetches the latest data, calculates the training cutoff,
    initializes a model of the given class, triggers the tuning process,
    and saves the model following the naming convention:
    `{model_name}_forecaster_{n_iteration}.joblib`.

    Args:
        model_class (type):
            The class of the forecaster model to train.
            The class should accept `iteration`, `end_dev`, and `train_size`
            in its constructor and provide a `tune()` method.
        n_iteration (int):
            The iteration number for this training run.
            This acts as an incrementing version number for the model. 
            When using `handle_training`, the first model starts at iteration 0. 
            Upon subsequent forced or scheduled retrainings, it is incremented 
            by 1 (`get_last_model_iteration + 1`). It is primarily used to 
            determine the filename when saving the model to disk 
            (e.g., `lgbm_forecaster_0.joblib`, `lgbm_forecaster_1.joblib`).
        model_name (Optional[str]):
            Optional name of the model to train.
            If None, the name is inferred from the model class.
            Defaults to None.
        train_size (Optional[pd.Timedelta]):
            Optional size of the training set as a pandas Timedelta.
            Determines the lookback window length from `end_dev`. If provided, the training data
            will start at `end_dev - train_size`. If None, all available data up to `end_dev` is used.
            Defaults to None.
        save_to_file (bool):
            If True, saves the model to disk after training.
            Defaults to True.
        model_dir (Optional[Union[str, Path]]):
            Directory where the model should be saved. If None, defaults to
            the library's cache home.
        end_dev (Optional[Union[str, pd.Timestamp]]):
            Optional cutoff date for training.
            This represents the absolute point in time separating training/development data
            from unseen future data. If None, it is calculated automatically to be one day
            before the latest available index in the data.
        data_filename (Optional[str]):
            Optional filename for the data to be used for training, e.g., 'interim/energy_load.csv'.
            If None, the default data file is used. Defaults to None.
        **kwargs (Any):
            Additional keyword arguments to be passed to the model constructor.

    Notes:
        Relationship between ``train_size`` and ``end_dev``:
        The actual training data spans from ``max(dataset_start, end_dev - train_size)`` to ``end_dev``.
        - If ``train_size`` is larger than the available history before ``end_dev``, the framework
          gracefully clips the start date to the beginning of the dataset without throwing an error.
        - If ``end_dev`` is set to a time before the start of the dataset, the training subset will
          be empty and the forecaster will fail to fit.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.manager.trainer import train_new_model
        >>> # Train using all available history up to the end of 2025:
        >>> # train_new_model(..., train_size=None, end_dev="2025-12-31 00:00+00:00")
        >>>
        >>> # Train using exactly 3 years of data leading up to the end of 2025:
        >>> # train_new_model(..., train_size=pd.Timedelta(days=3*365), end_dev="2025-12-31 00:00+00:00")
        >>>
        >>> # Train using the latest available data minus 1 day (default behavior):
        >>> # train_new_model(..., train_size=pd.Timedelta(days=3*365), end_dev=None)

    Returns:
        The trained model instance.

    """
    logger.info("Training new model (iteration %d)...", n_iteration)

    # Fetch data using the library's utility
    current_data = fetch_data(filename=data_filename)
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
        iteration=n_iteration,
        end_dev=end_train_cutoff,
        train_size=train_size,
        **kwargs,
    )
    logger.debug("Model initialized: %s", model)
    logger.debug("Model parameters: %s", model.get_params())

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
    data_filename: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Check if a new model needs to be trained and trigger training if necessary.

    Trains a new model if no model exists, if the existing model is older than
    7 days, or if retraining is forced.

    Args:
        model_class: The class of the forecaster model to train, for example
            `spotforecast2_safe.forecaster.ForecasterLGBM`.
        model_name: Name of the model (e.g., 'lgbm'). If None, it is inferred
            from the model_class name.
        model_dir:
            Directory where models are stored, see also get_cache_home().
        force:
            If True, force retraining even if the current model is recent. Default is False.
        train_size:
            Optional size of the training set. Default is None.
        end_dev:
            Optional cutoff date for training. Default is None.
        data_filename:
            Optional filename of the data used for training. Default is None.
        **kwargs:
            Additional keyword arguments passed to the model constructor.

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
            data_filename=data_filename,
            **kwargs,
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
            data_filename=data_filename,
            **kwargs,
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
            data_filename=data_filename,
            **kwargs,
        )
    else:
        logger.info(
            "The current %s model was trained up to %s (%.0f hours ago). "
            "No retraining necessary.",
            model_name,
            last_training_date,
            hours_since_last_training,
        )

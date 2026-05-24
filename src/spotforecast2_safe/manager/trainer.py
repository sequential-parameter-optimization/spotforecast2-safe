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
from joblib import load

from spotforecast2_safe.data.fetch_data import get_cache_home

logger = logging.getLogger(__name__)


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
            If *None*, defaults to `get_cache_home()`.

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
            If *None*, defaults to `get_cache_home()`.

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


def should_retrain(
    last_end_dev: Optional[pd.Timestamp],
    *,
    max_age: pd.Timedelta,
    force: bool = False,
    now: Optional[pd.Timestamp] = None,
) -> bool:
    """Decide whether a forecaster should be retrained.

    The cadence policy lives in `ConfigEntsoe.retrain_max_age`.  Callers
    pass `max_age` in explicitly so the policy stays in config, not in
    code.

    Args:
        last_end_dev: End-of-development-window timestamp of the last
            trained model.  ``None`` indicates no previous model.
        max_age: Maximum allowed age before retraining is required.
        force: Bypass the cadence gate and always return ``True``.
        now: Override "now" — used for deterministic tests.  Defaults
            to ``pd.Timestamp.now(tz="UTC")``.

    Returns:
        ``True`` when retraining should proceed (no previous model,
        ``force``, or age above ``max_age``); ``False`` otherwise.

    Examples:
        ```{python}
        import pandas as pd

        from spotforecast2_safe.manager.trainer import should_retrain

        now = pd.Timestamp("2026-05-24 12:00", tz="UTC")
        max_age = pd.Timedelta(days=7)

        # No previous model -> retrain.
        print(should_retrain(None, max_age=max_age, now=now))

        # Recent model -> skip.
        recent = now - pd.Timedelta(days=2)
        print(should_retrain(recent, max_age=max_age, now=now))

        # Old model -> retrain.
        old = now - pd.Timedelta(days=10)
        print(should_retrain(old, max_age=max_age, now=now))
        ```
    """
    if force or last_end_dev is None:
        return True

    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if last_end_dev.tzinfo is None:
        last_end_dev = last_end_dev.tz_localize("UTC")
    if current.tzinfo is None:
        current = current.tz_localize("UTC")

    age = current - last_end_dev
    if age >= max_age:
        return True

    hours = age.total_seconds() / 3600.0
    logger.info(
        "Last model trained %.0f h ago (max_age=%s); no retraining necessary.",
        hours,
        max_age,
    )
    return False

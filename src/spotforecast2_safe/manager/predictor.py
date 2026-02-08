# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Module for managing model predictions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from spotforecast2_safe.manager.trainer import get_last_model

logger = logging.getLogger(__name__)


def get_model_prediction(
    model_name: str, model_dir: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the prediction package from the latest trained model.

    This function retrieves the latest iteration of a specified model from the
    cache and calls its `package_prediction` method to obtain a comprehensive
    set of predictions and metrics.

    Args:
        model_name: Name of the model to use (e.g., 'lgbm', 'xgb').
        model_dir: Directory where models are stored. If None, defaults to
            the library's cache home.

    Returns:
        A dictionary containing predictions and metrics if a model is found and
        successfully executes `package_prediction`. Returns None otherwise.

    Examples:
        >>> from spotforecast2_safe.manager.predictor import get_model_prediction
        >>> # prediction_pkg = get_model_prediction('lgbm')
    """
    n_iteration, model = get_last_model(model_name, model_dir)

    if n_iteration < 0 or model is None:
        logger.error(
            "No trained model found for '%s'. Please train a model first.", model_name
        )
        return None

    logger.info(
        "Making predictions using %s model (iteration %d)...",
        model_name,
        n_iteration,
    )

    if not hasattr(model, "package_prediction"):
        logger.error(
            "Model '%s' (iteration %d) does not implement 'package_prediction' method.",
            model_name,
            n_iteration,
        )
        return None

    try:
        prediction_package = model.package_prediction()
        return prediction_package
    except Exception as e:
        logger.error(
            "Error occurred while generating prediction package for '%s': %s",
            model_name,
            e,
        )
        return None

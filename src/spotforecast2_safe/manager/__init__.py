# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Manager module for model persistence, prediction, and training.

This module provides utilities for:
- Logging configuration for safety-critical systems
- Model persistence (save/load)
- Model prediction management
- Model training and retraining workflows
- Model evaluation metrics
"""

from spotforecast2_safe.manager.demo_metrics import calculate_metrics
from spotforecast2_safe.manager.features import (
    apply_cyclical_encoding,
    create_interaction_features,
    get_target_data,
    merge_data_and_covariates,
    select_exogenous_features,
    select_top_poly_features,
)
from spotforecast2_safe.manager.logger import setup_logging
from spotforecast2_safe.manager.persistence import (
    ensure_model_dir,
    get_model_filepath,
    load_forecasters,
    model_directory_exists,
    save_forecaster,
    save_forecasters,
)
from spotforecast2_safe.manager.predictor import (
    build_prediction_package,
    get_model_prediction,
)
from spotforecast2_safe.manager.trainer import get_last_model, should_retrain

__all__ = [
    # Logger
    "setup_logging",
    # Metrics
    "calculate_metrics",
    # Persistence
    "ensure_model_dir",
    "get_model_filepath",
    "save_forecasters",
    "load_forecasters",
    "model_directory_exists",
    "save_forecaster",
    # Predictor
    "build_prediction_package",
    "get_model_prediction",
    # Trainer
    "get_last_model",
    "should_retrain",
    # Feature engineering helpers
    "apply_cyclical_encoding",
    "create_interaction_features",
    "get_target_data",
    "merge_data_and_covariates",
    "select_exogenous_features",
    "select_top_poly_features",
]

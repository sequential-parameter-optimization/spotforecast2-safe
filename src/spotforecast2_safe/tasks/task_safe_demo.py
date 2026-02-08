"""
Task demo: compare baseline, covariate, and custom LightGBM forecasts against ground truth.

This script executes the baseline N-to-1 task, the covariate-enhanced N-to-1
pipeline, and a custom LightGBM model with optimized hyperparameters, then loads
the ground truth from a specified data directory.

Logging Mechanism:
    This script uses a dual-handler logging system designed for safety-critical MLOps:
    1.  **Console Handler**: Provides real-time progress updates to `stdout`.
    2.  **File Handler**: Persists all log messages (including debug/tracebacks) to a
        timestamped file in `{model_root}/logs/`.

    Log File Location:
        By default, logs are saved to `~/spotforecast2_safe_models/logs/task_safe_demo_YYYYMMDD_HHMMSS.log`.

Safety-Critical Features:
- Persistent file-based logging for auditability.
- Path management using pathlib for cross-platform reliability.
- Explicit input validation and existence checks.
- Comprehensive error handling with traceback logging.
- Deterministic random seeding where applicable.
- Minimal dependency footprint (no plotting libraries).

Examples:
    Run the demo:

    >>> python tasks/task_safe_demo.py

    Force training (case-insensitive boolean):

    >>> python tasks/task_safe_demo.py --force_train false
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from lightgbm import LGBMRegressor

# Corrected package name from spotforecast2_safe_safe to spotforecast2_safe
from spotforecast2_safe.processing.agg_predict import agg_predict
from spotforecast2_safe.processing.n2n_predict import n2n_predict
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """Calculate MAE and MSE for numeric evaluation."""
    diff = actual - predicted
    mae = diff.abs().mean()
    mse = (diff**2).mean()
    return {"MAE": mae, "MSE": mse}


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the safety-critical demo task."""

    data_path: Path = field(
        default_factory=lambda: Path.home() / "spotforecast2_data" / "data_test.csv"
    )
    model_root: Path = field(
        default_factory=lambda: Path.home() / "spotforecast2_safe_models"
    )
    log_root: Path = field(
        default_factory=lambda: Path.home() / "spotforecast2_safe_models" / "logs"
    )
    forecast_horizon: int = 24
    contamination: float = 0.01
    window_size: int = 72
    lags: int = 24
    train_ratio: float = 0.8
    random_seed: int = 42

    weights: List[float] = field(
        default_factory=lambda: [
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
        ]
    )


def setup_logging(
    level: int = logging.INFO, log_dir: Optional[Path] = None
) -> Tuple[logging.Logger, Optional[Path]]:
    """
    Configure robust logging for safety-critical execution.
    Sets up both a stream (stdout) and an optional informative file handler.
    Always logs INFO level or higher to the file, regardless of console level.
    """
    logger = logging.getLogger("task_safe_demo")
    logger.setLevel(logging.DEBUG)  # Root level allows handlers to filter

    # Avoid duplicate handlers if main is called multiple times
    if logger.handlers:
        existing_path = None
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                existing_path = Path(h.baseFilename)
        return logger, existing_path

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 1. Console Handler (Respects the requested level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # 2. File Handler (Always INFO+ for audit durability)
    log_file_path = None
    if log_dir:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = log_dir / f"task_safe_demo_{timestamp}.log"

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.info(f"Persistent logging initialized at: {log_file_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize file logging in {log_dir}: {e}")

    return logger, log_file_path


def _parse_bool(value: str) -> bool:
    """Parse case-insensitive boolean strings for CLI arguments."""
    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "1"}:
        return True
    if normalized in {"false", "f", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def load_actual_combined(
    config: DemoConfig,
    columns: List[str],
) -> pd.Series:
    """Load ground truth and compute combined actual series with validation."""
    if not config.data_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {config.data_path}")

    data_test = pd.read_csv(config.data_path, index_col=0, parse_dates=True)

    # Safety Check: Ensure all requested columns exist in the test data
    missing_cols = [col for col in columns if col not in data_test.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in test data: {missing_cols}")

    actual_df = data_test[columns].iloc[: config.forecast_horizon]

    # Weight validation happens inside agg_predict
    return agg_predict(actual_df, weights=config.weights)


def main(
    force_train: bool = True,
    data_path: Optional[Path] = None,
    logging_enabled: bool = False,
) -> int:
    """
    Main execution entry point.
    Returns 0 on success, non-zero on failure.
    """
    # Initialize configuration first to get log path
    default_config = DemoConfig()
    config = DemoConfig(
        data_path=data_path if data_path else default_config.data_path,
        model_root=default_config.model_root,
        log_root=default_config.log_root,
        forecast_horizon=default_config.forecast_horizon,
        contamination=default_config.contamination,
        window_size=default_config.window_size,
        lags=default_config.lags,
        train_ratio=default_config.train_ratio,
        random_seed=default_config.random_seed,
        weights=default_config.weights,
    )

    # Setup Logging if enabled
    log_file = None
    if logging_enabled:
        logger, log_file = setup_logging(log_dir=config.log_root)
    else:
        logger = logging.getLogger("task_safe_demo")
        logger.addHandler(logging.NullHandler())

    # --- Fail-Fast Check: Validate ground truth availability before compute ---
    logger.info(f"Validating ground truth availability: {config.data_path}")
    if not config.data_path.is_file():
        logger.error(f"FAIL-FAST: Ground truth file not found at {config.data_path}")
        logger.error(
            "Please ensure the file exists or specify a valid path via --data_path"
        )
        return 1
    logger.info("Ground truth file verified.")

    try:
        # --- Baseline predictions ---
        logger.info("Executing baseline forecast...")
        baseline_predictions, _ = n2n_predict(
            columns=None,
            forecast_horizon=config.forecast_horizon,
            contamination=config.contamination,
            window_size=config.window_size,
            verbose=False,  # Use our own logging
            show_progress=True,
            force_train=force_train,
            model_dir=config.model_root / "task_demo_baseline",
        )

        baseline_combined = agg_predict(baseline_predictions, weights=config.weights)
        logger.info(f"Baseline combined generated. Shape: {baseline_combined.shape}")

        # --- Covariate-enhanced predictions ---
        logger.info("Executing covariate-enhanced forecast...")
        cov_predictions, _, _ = n2n_predict_with_covariates(
            forecast_horizon=config.forecast_horizon,
            contamination=config.contamination,
            window_size=config.window_size,
            lags=config.lags,
            train_ratio=config.train_ratio,
            verbose=False,
            show_progress=True,
            force_train=force_train,
            model_dir=config.model_root / "task_demo_covariates",
        )

        covariates_combined = agg_predict(cov_predictions, weights=config.weights)
        logger.info(
            f"Covariates combined generated. Shape: {covariates_combined.shape}"
        )

        # --- Custom LightGBM predictions ---
        logger.info("Executing custom LightGBM forecast...")
        custom_lgbm = LGBMRegressor(
            n_estimators=1059,
            learning_rate=0.04191323446625026,
            num_leaves=212,
            min_child_samples=54,
            subsample=0.5014650987802548,
            colsample_bytree=0.6080926628683118,
            random_state=config.random_seed,
            verbose=-1,
        )
        custom_lgbm_predictions, _, _ = n2n_predict_with_covariates(
            forecast_horizon=config.forecast_horizon,
            contamination=config.contamination,
            window_size=config.window_size,
            lags=config.lags,
            train_ratio=config.train_ratio,
            timezone="UTC",
            estimator=custom_lgbm,
            verbose=False,
            show_progress=True,
            force_train=force_train,
            model_dir=config.model_root / "task_demo_custom_lgbm",
        )
        custom_lgbm_combined = agg_predict(
            custom_lgbm_predictions, weights=config.weights
        )
        logger.info(
            f"Custom LightGBM combined generated. Shape: {custom_lgbm_combined.shape}"
        )

        # --- Comparative Metrics ---
        logger.info("Comparing models...")
        idx_match = (baseline_combined.index == covariates_combined.index).all()
        logger.info(f"Indices align between baseline and covariates: {idx_match}")

        # --- Ground truth and Evaluation ---
        # Note: We checked for file existence above, but load_actual_combined
        # also performs schema validation.
        try:
            columns = list(baseline_predictions.columns)
            actual_combined = load_actual_combined(config=config, columns=columns)
            actual_combined = actual_combined.reindex(baseline_combined.index)

            # Use fillna(0) or dropna() based on safety policy; here we assume test data should be complete
            if actual_combined.isnull().any():
                logger.warning("Actual combined series contains NaNs after reindexing.")
                actual_combined = actual_combined.dropna()
                common_idx = actual_combined.index.intersection(baseline_combined.index)
                actual_combined = actual_combined.loc[common_idx]
                baseline_combined = baseline_combined.loc[common_idx]
                covariates_combined = covariates_combined.loc[common_idx]
                custom_lgbm_combined = custom_lgbm_combined.loc[common_idx]

            metrics_baseline = calculate_metrics(actual_combined, baseline_combined)
            metrics_cov = calculate_metrics(actual_combined, covariates_combined)
            metrics_lgbm = calculate_metrics(actual_combined, custom_lgbm_combined)

            logger.info("\n=== EVALUATION RESULTS ===")
            logger.info(f"Baseline:       {metrics_baseline}")
            logger.info(f"Covariates:     {metrics_cov}")
            logger.info(f"Custom LGBM:    {metrics_lgbm}")
            logger.info("==========================\n")

        except Exception as e:
            logger.error(f"Error during evaluation (checking data again): {e}")

        logger.info("--- task_safe_demo completed successfully ---")
        if log_file:
            print(f"Finalized logging info saved to: {log_file}")
        return 0

    except Exception as _e:  # noqa: F841
        logger.critical("Task failed with an unexpected error:")
        logger.critical(traceback.format_exc())
        if log_file:
            print(f"Finalized logging info saved to: {log_file}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the spotforecast2_safe demo task."
    )
    parser.add_argument(
        "--force_train",
        type=_parse_bool,
        default=True,
        help="Force training (true/false, case-insensitive).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the ground truth CSV file.",
    )
    parser.add_argument(
        "--logging",
        type=_parse_bool,
        default=False,
        help="Enable logging (both console and file).",
    )
    args = parser.parse_args()

    # Convert string path to Path object if provided
    specified_data_path = Path(args.data_path) if args.data_path else None

    sys.exit(
        main(
            force_train=args.force_train,
            data_path=specified_data_path,
            logging_enabled=args.logging,
        )
    )

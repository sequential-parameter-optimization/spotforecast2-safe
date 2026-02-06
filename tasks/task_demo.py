"""
Task demo: compare baseline, covariate, and custom LightGBM forecasts against ground truth.

This script executes the baseline N-to-1 task, the covariate-enhanced N-to-1
pipeline, and a custom LightGBM model with optimized hyperparameters, then loads
the ground truth from ~/spotforecast2_data/data_test.csv and plots Actual vs
Predicted using Plotly.

The plot includes:
    - Actual combined values (ground truth)
    - Baseline combined prediction (n2n_predict)
    - Covariate combined prediction (n2n_predict_with_covariates, default LGBM)
    - Custom LightGBM combined prediction (optimized hyperparameters, Europe/Berlin tz)

Examples:
    Run the demo:

    >>> python tasks/task_demo.py

    Force training (case-insensitive boolean):

    >>> python tasks/task_demo.py --force_train false

    Save the plot as a single HTML file (default: task_demo_plot.html):

    >>> python tasks/task_demo.py --html

    Save to a specific path:

    >>> python tasks/task_demo.py --html results/plot.html
"""

from __future__ import annotations

import argparse
import warnings
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMRegressor

from spotforecast2_safe.processing.agg_predict import agg_predict
from spotforecast2_safe.processing.n2n_predict import n2n_predict
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)

warnings.simplefilter("ignore")


def _parse_bool(value: str) -> bool:
    """Parse case-insensitive boolean strings for CLI arguments."""
    normalized = value.strip().lower()
    if normalized in {"true", "t"}:
        return True
    if normalized in {"false", "f"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value: true/false (case-insensitive)."
    )


def _load_actual_combined(
    data_path: str,
    columns: List[str],
    weights: List[float],
    forecast_horizon: int,
) -> pd.Series:
    """Load ground truth and compute combined actual series.

    Args:
        data_path: Path to the data_test.csv file.
        columns: Column names to use for aggregation.
        weights: Weight list aligned with columns.
        forecast_horizon: Number of steps to take from the start of test data.

    Returns:
        Combined actual values as a Series.
    """
    data_test = pd.read_csv(data_path, index_col=0, parse_dates=True)
    actual_df = data_test[columns].iloc[:forecast_horizon]
    return agg_predict(actual_df, weights=weights)


def _plot_actual_vs_predicted(
    actual_combined: pd.Series,
    baseline_combined: pd.Series,
    covariates_combined: pd.Series,
    custom_lgbm_combined: pd.Series,
    html_path: Optional[str] = None,
) -> None:
    """Plot actual vs predicted combined values.

    Args:
        actual_combined: Ground truth combined series.
        baseline_combined: Baseline combined prediction series.
        covariates_combined: Covariate-enhanced combined prediction series.
        custom_lgbm_combined: Custom LightGBM (optimized params) combined prediction series.
        html_path: If set, save the plot as a single self-contained HTML file to this path.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_combined.index,
            y=actual_combined.values,
            mode="lines+markers",
            name="Actual",
            line=dict(color="green", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_combined.index,
            y=baseline_combined.values,
            mode="lines+markers",
            name="Predicted (Baseline)",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=covariates_combined.index,
            y=covariates_combined.values,
            mode="lines+markers",
            name="Predicted (Covariates)",
            line=dict(color="blue", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=custom_lgbm_combined.index,
            y=custom_lgbm_combined.values,
            mode="lines+markers",
            name="Predicted (Custom LightGBM)",
            line=dict(color="orange", width=2, dash="dashdot"),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="Combined Values: Actual vs. Predicted",
        xaxis_title="Time",
        yaxis_title="Combined Value",
        width=1000,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )

    if html_path:
        fig.write_html(html_path)
        print(f"Plot saved to {html_path}")

    fig.show()


def main(
    force_train: bool = True,
    html_path: Optional[str] = None,
) -> None:
    """Run the demo, compute predictions for three models, and plot actual vs predicted."""
    DATA_PATH = "~/spotforecast2_data/data_test.csv"
    FORECAST_HORIZON = 24
    CONTAMINATION = 0.01
    WINDOW_SIZE = 72
    LAGS = 24
    TRAIN_RATIO = 0.8
    VERBOSE = True
    SHOW_PROGRESS = True
    FORCE_TRAIN = force_train

    WEIGHTS = [
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

    print("--- Starting task_demo: baseline, covariates, and custom LightGBM ---")

    # --- Baseline predictions ---
    baseline_predictions, _ = n2n_predict(
        columns=None,
        forecast_horizon=FORECAST_HORIZON,
        contamination=CONTAMINATION,
        window_size=WINDOW_SIZE,
        verbose=VERBOSE,
        show_progress=SHOW_PROGRESS,
        force_train=FORCE_TRAIN,
        model_dir="~/spotforecast2_models/task_demo_baseline",
    )

    baseline_combined = agg_predict(baseline_predictions, weights=WEIGHTS)

    # --- Covariate-enhanced predictions ---
    cov_predictions, _, _ = n2n_predict_with_covariates(
        forecast_horizon=FORECAST_HORIZON,
        contamination=CONTAMINATION,
        window_size=WINDOW_SIZE,
        lags=LAGS,
        train_ratio=TRAIN_RATIO,
        verbose=VERBOSE,
        show_progress=SHOW_PROGRESS,
        force_train=FORCE_TRAIN,
        model_dir="~/spotforecast2_models/task_demo_covariates",
    )

    covariates_combined = agg_predict(cov_predictions, weights=WEIGHTS)

    # --- Custom LightGBM predictions (optimized hyperparameters) ---
    custom_lgbm = LGBMRegressor(
        n_estimators=1059,
        learning_rate=0.04191323446625026,
        num_leaves=212,
        min_child_samples=54,
        subsample=0.5014650987802548,
        colsample_bytree=0.6080926628683118,
        random_state=42,
        verbose=-1,
    )
    custom_lgbm_predictions, _, _ = n2n_predict_with_covariates(
        forecast_horizon=FORECAST_HORIZON,
        contamination=CONTAMINATION,
        window_size=WINDOW_SIZE,
        lags=LAGS,
        train_ratio=TRAIN_RATIO,
        timezone="UTC",
        estimator=custom_lgbm,
        verbose=VERBOSE,
        show_progress=SHOW_PROGRESS,
        force_train=FORCE_TRAIN,
        model_dir="~/spotforecast2_models/task_demo_custom_lgbm",
    )
    custom_lgbm_combined = agg_predict(custom_lgbm_predictions, weights=WEIGHTS)

    # --- Debug output ---
    print("\n=== DEBUG INFO ===")
    print(f"Baseline combined shape: {baseline_combined.shape}")
    print(f"Baseline index: {baseline_combined.index[0]} to {baseline_combined.index[-1]}")
    print(f"Baseline values (first 5): {baseline_combined.head().values}")
    print(f"\nCovariates combined shape: {covariates_combined.shape}")
    print(f"Covariates index: {covariates_combined.index[0]} to {covariates_combined.index[-1]}")
    print(f"Covariates values (first 5): {covariates_combined.head().values}")
    print(f"\nCustom LightGBM combined shape: {custom_lgbm_combined.shape}")
    print(f"Custom LightGBM index: {custom_lgbm_combined.index[0]} to {custom_lgbm_combined.index[-1]}")
    print(f"Custom LightGBM values (first 5): {custom_lgbm_combined.head().values}")
    print(f"\nAre indices aligned? {(baseline_combined.index == covariates_combined.index).all()}")
    print(f"Baseline vs Covariates identical? {(baseline_combined.values == covariates_combined.values).all()}")
    print(f"Baseline vs Custom LightGBM identical? {(baseline_combined.values == custom_lgbm_combined.values).all()}")
    print(f"Covariates vs Custom LightGBM identical? {(covariates_combined.values == custom_lgbm_combined.values).all()}")
    if not (baseline_combined.values == covariates_combined.values).all():
        diff = baseline_combined - covariates_combined
        print(f"Baseline - Covariates diff stats:\n{diff.describe()}")
    if not (covariates_combined.values == custom_lgbm_combined.values).all():
        diff_lgbm = covariates_combined - custom_lgbm_combined
        print(f"Covariates - Custom LightGBM diff stats:\n{diff_lgbm.describe()}")
    print("==================\n")

    # --- Ground truth ---
    columns = list(baseline_predictions.columns)
    actual_combined = _load_actual_combined(
        data_path=DATA_PATH,
        columns=columns,
        weights=WEIGHTS,
        forecast_horizon=FORECAST_HORIZON,
    )

    # Align indices to predictions for clean plotting
    actual_combined = actual_combined.reindex(baseline_combined.index)

    # --- Plot ---
    _plot_actual_vs_predicted(
        actual_combined=actual_combined,
        baseline_combined=baseline_combined,
        covariates_combined=covariates_combined,
        custom_lgbm_combined=custom_lgbm_combined,
        html_path=html_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the spotforecast2 demo task.")
    parser.add_argument(
        "--force_train",
        type=_parse_bool,
        default=True,
        help="Force training (true/false, case-insensitive).",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const="task_demo_plot.html",
        default=None,
        metavar="PATH",
        help="Save the plot as a single self-contained HTML file. Default path: task_demo_plot.html",
    )
    args = parser.parse_args()
    main(force_train=args.force_train, html_path=args.html)

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Task demo: four-zone bottom-up total-load forecast vs. a direct aggregate.

This task demonstrates the four-German-TSO-zone forecasting design: fit one
model per control-zone load series (Amprion, TenneT, TransnetBW, 50Hertz) and
sum the four forecasts into the total-German-load forecast (bottom-up
aggregation), then compare it against a single model trained on the aggregate
load directly. The bottom-up combination reuses the existing multi-target
machinery -- the four zones become `ConfigMulti.targets` and the per-target
forecasts are summed via the `MultiTask` aggregation with
``agg_weights=[1.0, ...]``.

Data source. With ``data_path`` pointing at the assembled four-column interim
file (see `spotforecast2_safe.downloader.entsoe.assemble_zone_loads`), the task
runs on real ENTSO-E control-zone load. With ``data_path=None`` (the default) it
builds a deterministic, seeded synthetic four-zone dataset so the demo and its
smoke test run offline without an ENTSO-E API key.

The comparison uses the project's own backtesting engine
(`backtesting_forecaster` + `TimeSeriesFold`) over a 24-hour horizon and reports
MAE / RMSE / MAPE for both methods.

Per-zone weather (``--per_zone_weather``). When enabled, each zone model
receives weather from its own TSO control-area cities instead of the shared
single-point baseline. The pipeline uses ``on_weather_failure="skip"`` so the
smoke test stays network-free: if Open-Meteo is unreachable the per-zone
weather degrades to no-weather rather than failing. The default path
(``per_zone_weather=False``) is exactly the pre-feature baseline
(``use_exogenous_features=False``).

Examples:
    ```{python}
    #| eval: false
    # Shell commands; they cannot run inside a Python kernel.
    # Synthetic offline demo (no API key needed):
    #   uv run spotforecast-safe-zone-load-demo
    # Run on assembled real ENTSO-E zone data:
    #   uv run spotforecast-safe-zone-load-demo \
    #       --data_path ~/spotforecast2_data/interim/energy_load_zones.csv
    # Enable per-zone weather (requires Open-Meteo access; degrades gracefully):
    #   uv run spotforecast-safe-zone-load-demo --per_zone_weather true
    ```
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from spotforecast2_safe.backtesting import backtesting_forecaster
from spotforecast2_safe.configurator import ConfigMulti
from spotforecast2_safe.downloader.entsoe import GERMAN_TSO_ZONES
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.demo_metrics import calculate_metrics
from spotforecast2_safe.manager.logger import setup_logging
from spotforecast2_safe.multitask import MultiTask
from spotforecast2_safe.splitter import TimeSeriesFold
from spotforecast2_safe.utils.parse import parse_bool

_LAGS = 24


def _make_estimator(random_seed: int) -> LGBMRegressor:
    """Return a small, deterministic LightGBM regressor for the demo."""
    return LGBMRegressor(
        n_estimators=50,
        random_state=random_seed,
        n_jobs=1,
        verbose=-1,
    )


def _synthetic_zone_frame(
    periods: int, random_seed: int, zones: List[str]
) -> pd.DataFrame:
    """Build a deterministic synthetic hourly load frame, one column per zone.

    Each zone gets a base level plus a daily sinusoid, a weekend dip, and seeded
    Gaussian noise. Values are strictly positive (safe for MAPE).
    """
    rng = np.random.default_rng(random_seed)
    idx = pd.date_range("2023-01-01", periods=periods, freq="h", tz="UTC")
    hours = idx.hour.to_numpy()
    is_weekend = (idx.dayofweek.to_numpy() >= 5).astype(float)
    base_levels = {
        "load_amprion": 22000.0,
        "load_tennet": 25000.0,
        "load_transnetbw": 9000.0,
        "load_50hertz": 11000.0,
    }
    data: Dict[str, np.ndarray] = {}
    for col in zones:
        base = base_levels.get(col, 15000.0)
        daily = 0.15 * base * np.sin(2 * np.pi * (hours - 3) / 24)
        weekend = -0.05 * base * is_weekend
        noise = rng.normal(0.0, 0.02 * base, periods)
        data[col] = base + daily + weekend + noise
    frame = pd.DataFrame(data, index=idx)
    frame.index.name = "DateTime"
    return frame


def _load_zone_frame(path: Path) -> pd.DataFrame:
    """Load an assembled four-column zone-load CSV into a datetime-indexed frame."""
    frame = pd.read_csv(path)
    time_col = "Time (UTC)" if "Time (UTC)" in frame.columns else frame.columns[0]
    frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
    frame = frame.set_index(time_col).sort_index()
    # Restore the hourly frequency lost on CSV round-trip; the recursive
    # forecaster needs index.freq to expand the prediction horizon.
    frame = frame.asfreq("h")
    frame.index.name = "DateTime"
    # Keep only the known zone columns (drop any *_forecast side columns).
    zone_cols = [c for c in GERMAN_TSO_ZONES if c in frame.columns]
    return frame[zone_cols] if zone_cols else frame


def _metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """MAE / RMSE / MAPE for an aligned (actual, predicted) pair."""
    base = calculate_metrics(actual, predicted)  # MAE + MSE; raises on NaN
    rmse = float(np.sqrt(base["MSE"]))
    mape = float((actual - predicted).abs().div(actual.abs()).mean() * 100.0)
    return {"MAE": float(base["MAE"]), "RMSE": rmse, "MAPE": mape}


def _compare_bottom_up_vs_aggregate(
    df: pd.DataFrame,
    zones: List[str],
    total_actual: pd.Series,
    predict_size: int,
    random_seed: int,
    n_folds: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Backtest the bottom-up sum vs. the direct aggregate over a rolling horizon.

    Treatment (bottom-up): backtest one `ForecasterRecursive` per zone over
    identical folds, then sum the per-fold predictions. Baseline (direct): one
    forecaster on the aggregate total. Both are scored against ``total_actual``.
    """
    initial_train_size = len(df) - predict_size * n_folds
    if initial_train_size < predict_size * 3:
        raise ValueError(
            "Not enough history for the requested backtest: need at least "
            f"{predict_size * (n_folds + 3)} hours, have {len(df)}."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = TimeSeriesFold(
            steps=predict_size,
            initial_train_size=initial_train_size,
            refit=False,
            fixed_train_size=True,
            verbose=False,
        )

        zone_preds = []
        for col in zones:
            forecaster = ForecasterRecursive(
                estimator=_make_estimator(random_seed), lags=_LAGS
            )
            _, preds = backtesting_forecaster(
                forecaster,
                df[col],
                cv,
                metric="mean_absolute_error",
                interval=None,
                n_jobs=1,
                show_progress=False,
                suppress_warnings=True,
                verbose=False,
            )
            zone_preds.append(preds["pred"].rename(col))

        baseline_forecaster = ForecasterRecursive(
            estimator=_make_estimator(random_seed), lags=_LAGS
        )
        _, baseline_preds = backtesting_forecaster(
            baseline_forecaster,
            total_actual,
            cv,
            metric="mean_absolute_error",
            interval=None,
            n_jobs=1,
            show_progress=False,
            suppress_warnings=True,
            verbose=False,
        )

    bottom_up_pred = pd.concat(zone_preds, axis=1, sort=False).sum(axis=1)
    baseline_pred = baseline_preds["pred"]

    idx = baseline_pred.index.intersection(bottom_up_pred.index)
    truth = total_actual.reindex(idx)
    return {
        "direct_aggregate": _metrics(truth, baseline_pred.reindex(idx)),
        "bottom_up_sum": _metrics(truth, bottom_up_pred.reindex(idx)),
    }


def main(
    synthetic: bool = True,
    data_path: Optional[Path] = None,
    predict_size: int = 24,
    history_hours: int = 24 * 100,
    random_seed: int = 314159,
    logging_enabled: bool = False,
    per_zone_weather: bool = False,
) -> int:
    """Run the four-zone bottom-up load demo. Returns 0 on success, 1 on failure.

    Args:
        synthetic: If True (and ``data_path`` is None), use seeded synthetic
            data. Ignored when ``data_path`` is provided.
        data_path: Path to an assembled four-column zone-load CSV. When given,
            real data is used and ``synthetic`` is ignored.
        predict_size: Forecast horizon in hours (also the backtest fold size).
        history_hours: Length of the synthetic series in hours.
        random_seed: Seed for the synthetic data and the estimators.
        logging_enabled: If True, attach the dual console/file handlers.
        per_zone_weather: If True, each zone model fetches weather from its own
            TSO control-area cities (``config.per_zone_weather=True``,
            ``on_weather_failure="skip"``). Requires Open-Meteo access; if
            unreachable the pipeline degrades to no-weather. Default ``False``
            keeps the pipeline byte-identical to the pre-feature baseline.

    Examples:
        ```{python}
        from spotforecast2_safe.tasks.task_safe_zone_load_demo import main

        # Fail-fast: a non-existent data_path returns 1 without computing.
        from pathlib import Path
        rc = main(data_path=Path("/nonexistent/zones.csv"))
        print(f"Return code (missing data): {rc}")
        assert rc == 1
        ```
    """
    if logging_enabled:
        logger, log_file = setup_logging(
            log_dir=Path.home() / "spotforecast2_safe_models" / "logs"
        )
    else:
        logger = logging.getLogger("task_safe_zone_load_demo")
        logger.addHandler(logging.NullHandler())
        log_file = None

    try:
        # --- Obtain the four-zone load frame ---
        if data_path is not None:
            if not Path(data_path).is_file():
                logger.error("FAIL-FAST: zone-load file not found at %s", data_path)
                return 1
            df = _load_zone_frame(Path(data_path))
            logger.info("Loaded real zone-load frame from %s", data_path)
        else:
            df = _synthetic_zone_frame(
                history_hours, random_seed, list(GERMAN_TSO_ZONES)
            )
            logger.info("Using synthetic four-zone load frame (%d hours).", len(df))

        zones = [c for c in GERMAN_TSO_ZONES if c in df.columns] or list(df.columns)
        if len(zones) < 2:
            logger.error("FAIL-FAST: need >= 2 zone columns, found %s", zones)
            return 1
        total_actual = df[zones].sum(axis=1)
        logger.info("Zones: %s", zones)

        # --- Operational bottom-up forecast (one model per zone, summed) ---
        with tempfile.TemporaryDirectory() as tmp:
            if per_zone_weather:
                config = ConfigMulti(
                    targets=zones,
                    agg_weights=[1.0] * len(zones),
                    predict_size=predict_size,
                    use_exogenous_features=True,
                    per_zone_weather=True,
                    on_weather_failure="skip",
                    use_outlier_detection=False,
                    auto_save_models=False,
                    number_folds=2,
                    random_state=random_seed,
                    cache_home=tmp,
                )
            else:
                config = ConfigMulti(
                    targets=zones,
                    agg_weights=[1.0] * len(zones),  # SUM -> total (not the mean)
                    predict_size=predict_size,
                    use_exogenous_features=False,
                    use_outlier_detection=False,
                    auto_save_models=False,
                    number_folds=2,
                    random_state=random_seed,
                    cache_home=tmp,
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                task = MultiTask(config, dataframe=df, task="lazy")
                task.prepare_data().detect_outliers().impute().build_exogenous_features()
                result = task.run(task="lazy")
            bottom_up_future = result["future_pred"]
        logger.info(
            "Bottom-up total forecast (next %d h) = sum of %d per-zone forecasts; "
            "shape=%s, mean=%.0f MW",
            predict_size,
            len(zones),
            bottom_up_future.shape,
            float(bottom_up_future.mean()),
        )

        # --- Research comparison: bottom-up vs. direct aggregate ---
        comparison = _compare_bottom_up_vs_aggregate(
            df, zones, total_actual, predict_size, random_seed
        )
        logger.info("=== Bottom-up sum vs. direct aggregate (24 h backtest) ===")
        for name, m in comparison.items():
            logger.info(
                "%-18s MAE=%10.2f  RMSE=%10.2f  MAPE=%6.3f%%",
                name,
                m["MAE"],
                m["RMSE"],
                m["MAPE"],
            )
        winner = (
            "bottom-up sum"
            if comparison["bottom_up_sum"]["MAE"]
            < comparison["direct_aggregate"]["MAE"]
            else "direct aggregate"
        )
        logger.info("Lower-MAE method: %s", winner)
        logger.info("--- task_safe_zone_load_demo completed successfully ---")
        if log_file:
            print(f"Finalized logging info saved to: {log_file}")
        return 0

    except Exception:  # noqa: BLE001 - reported with traceback, returns 1
        logger.critical("Task failed with an unexpected error:")
        logger.critical(traceback.format_exc())
        if log_file:
            print(f"Finalized logging info saved to: {log_file}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Four-zone bottom-up total-load forecast demo."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to an assembled four-column zone-load CSV. "
        "Omit to use seeded synthetic data (runs offline).",
    )
    parser.add_argument(
        "--predict_size",
        type=int,
        default=24,
        help="Forecast horizon in hours (also the backtest fold size).",
    )
    parser.add_argument(
        "--logging",
        type=parse_bool,
        default=False,
        help="Enable logging (both console and file).",
    )
    parser.add_argument(
        "--per_zone_weather",
        type=parse_bool,
        default=False,
        help="When true, each zone model fetches weather from its own TSO "
        "control-area cities. Requires Open-Meteo access; if unreachable "
        "the pipeline degrades to no-weather (on_weather_failure='skip'). "
        "Default false keeps the pre-feature baseline.",
    )
    args = parser.parse_args()

    specified_data_path = Path(args.data_path) if args.data_path else None

    sys.exit(
        main(
            data_path=specified_data_path,
            predict_size=args.predict_size,
            logging_enabled=args.logging,
            per_zone_weather=args.per_zone_weather,
        )
    )

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Thin ConfigMulti-driven entry point for N-to-1 forecasting.

This module is a single-call wrapper around the ``multitask`` pipeline.
It delegates all heavy lifting to
``spotforecast2_safe.multitask.runner.run`` with ``task="lazy"`` and
returns the forecast DataFrame directly.

``run_pipeline`` requires an explicit ``ConfigMulti`` instance.  Outlier
``bounds`` and aggregation ``agg_weights`` are domain-specific calibrations
and must be supplied by the caller on ``ConfigMulti``; there are no
dataset-specific presets.  Input data must always be passed explicitly via
the ``dataframe`` argument.  The CLI flag ``--weights`` maps to
``ConfigMulti.agg_weights``; the flag ``--train_ratio`` derives
``train_size`` from the extent of the bundled ``demo10.csv`` (Python API
callers supply ``train_size`` explicitly on ``ConfigMulti``).

CLI entry point: ``spotforecast-safe-n2o1-cov-df``
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.data.fetch_data import get_package_data_home
from spotforecast2_safe.multitask.runner import run
from spotforecast2_safe.utils.parse import parse_bool

# ---------------------------------------------------------------------------
# Argument parser (shared by CLI entry-point and _build_config_from_cli)
# ---------------------------------------------------------------------------


def _make_arg_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the safety-critical N-to-1 forecasting pipeline "
            "(ConfigMulti-driven, multitask runner)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Forecast parameters
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=24,
        help="Number of steps ahead to forecast.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=24,
        help="Lag depth N; expands to lags_consider=range(1, N+1).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data used for training [0, 1].",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Outlier contamination parameter [0, 0.5).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=72,
        help="Rolling window size for weighted imputation (hours).",
    )

    # Location parameters
    parser.add_argument(
        "--latitude",
        type=float,
        default=51.5136,
        help="Location latitude for solar/weather features.",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=7.4653,
        help="Location longitude for solar/weather features.",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone string.",
    )
    parser.add_argument(
        "--country_code",
        type=str,
        default="DE",
        help="ISO 3166-1 alpha-2 country code for holidays.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="NW",
        help="ISO 3166-2 subdivision code for regional holidays.",
    )

    # Feature engineering flags
    parser.add_argument(
        "--include_weather_windows",
        type=parse_bool,
        default=False,
        help="Enable rolling weather-window features.",
    )
    parser.add_argument(
        "--include_holiday_features",
        type=parse_bool,
        default=False,
        help="Enable public-holiday indicator features.",
    )
    parser.add_argument(
        "--include_holiday_adjacency_features",
        type=parse_bool,
        default=False,
        help="Enable Brückentag and before/after-holiday features.",
    )
    parser.add_argument(
        "--poly_features_degree",
        type=int,
        default=1,
        help="Polynomial-interaction degree (1 = off).",
    )
    parser.add_argument(
        "--max_poly_features",
        type=int,
        default=10,
        help="Cap on kept polynomial columns (top-K by mutual information).",
    )

    # Execution controls
    parser.add_argument(
        "--verbose",
        type=parse_bool,
        default=False,
        help="Enable verbose pipeline output.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Space-separated aggregation weights (one per target column).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Cache directory for models and logs (maps to cache_home).",
    )

    return parser


def _build_config_from_cli(args: argparse.Namespace) -> ConfigMulti:
    """Translate parsed CLI arguments into a ``ConfigMulti`` instance.

    CLI flag to ``ConfigMulti`` field mapping:

    - ``--forecast_horizon N`` -> ``predict_size=N``
    - ``--lags N`` -> ``lags_consider=list(range(1, N+1))``
    - ``--train_ratio R`` -> ``train_size`` derived from demo10 data extent
    - ``--contamination`` -> ``contamination``
    - ``--window_size`` -> ``window_size``
    - ``--latitude`` -> ``latitude``
    - ``--longitude`` -> ``longitude``
    - ``--timezone`` -> ``timezone``
    - ``--country_code`` -> ``country_code``
    - ``--state`` -> ``state``
    - ``--include_weather_windows`` -> ``include_weather_windows``
    - ``--include_holiday_features`` -> ``include_holiday_features``
    - ``--include_holiday_adjacency_features`` -> ``include_holiday_adjacency_features``
    - ``--poly_features_degree`` -> ``poly_features_degree``
    - ``--max_poly_features`` -> ``max_poly_features``
    - ``--verbose`` -> ``verbose``
    - ``--weights w1 w2 ...`` -> ``agg_weights=[w1, w2, ...]``
    - ``--log_dir PATH`` -> forwarded as ``cache_home`` to ``main()``

    ``--lags N`` expands to ``lags_consider=list(range(1, N+1))`` because
    ``default_lgbm_forecaster_factory`` reads ``config.lags_consider[-1]``
    to set the lag depth of ``ForecasterRecursive``, preserving the old
    ``ForecasterRecursive(lags=N)`` behaviour.

    ``--train_ratio R`` is translated into ``train_size`` as a
    ``pd.Timedelta`` derived from the demo10 dataset extent.

    Args:
        args: Parsed ``argparse.Namespace`` from ``_make_arg_parser()``.

    Returns:
        A new ``ConfigMulti`` instance with the CLI arguments applied.
    """
    # --lags N -> lags_consider=list(range(1, N+1))
    lags_consider = list(range(1, args.lags + 1))

    # --train_ratio R -> train_size derived from demo10 extent
    data_home = get_package_data_home()
    try:
        _df = pd.read_csv(data_home / "demo10.csv", index_col=0, parse_dates=True)
    except OSError as exc:
        # Fail-safe: never silently substitute a train_size unrelated to
        # the data the pipeline will actually load.
        raise FileNotFoundError(
            "Cannot derive train_size from --train_ratio: the bundled "
            f"demo10.csv could not be read from {data_home}: {exc}"
        ) from exc
    first_ts = pd.to_datetime(_df.index.min(), utc=True)
    last_ts = pd.to_datetime(_df.index.max(), utc=True)
    total_span = last_ts - first_ts
    train_size = pd.Timedelta(
        seconds=int(total_span.total_seconds() * args.train_ratio)
    )

    return ConfigMulti(
        predict_size=args.forecast_horizon,
        lags_consider=lags_consider,
        train_size=train_size,
        contamination=args.contamination,
        window_size=args.window_size,
        latitude=args.latitude,
        longitude=args.longitude,
        timezone=args.timezone,
        country_code=args.country_code,
        state=args.state,
        include_weather_windows=args.include_weather_windows,
        include_holiday_features=args.include_holiday_features,
        include_holiday_adjacency_features=args.include_holiday_adjacency_features,
        poly_features_degree=args.poly_features_degree,
        max_poly_features=args.max_poly_features,
        verbose=args.verbose,
        agg_weights=args.weights,
    )


def run_pipeline(
    config: Optional[ConfigMulti] = None,
    *,
    dataframe: Optional[pd.DataFrame] = None,
    data_test: Optional[pd.DataFrame] = None,
    project_name: str = "demo10",
    cache_home: Optional[Path] = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Execute the N-to-1 forecasting pipeline and return the forecast DataFrame.

    Execution is delegated to ``spotforecast2_safe.multitask.runner.run``
    with ``task="lazy"``.  A ``ConfigMulti`` instance must be supplied
    explicitly; there is no implicit fallback.  Outlier ``bounds`` and
    aggregation ``agg_weights`` are domain-specific calibrations and must
    be provided by the caller — no preset values are substituted.  Input
    data must likewise be supplied via ``dataframe``; auto-loading is not
    performed.

    Args:
        config: A ``ConfigMulti`` instance.  Must not be ``None``; passing
            ``None`` raises ``ValueError``.
        dataframe: Input time-series DataFrame.  Must contain a datetime
            column matching ``config.index_name`` and at least one numeric
            target column.
        data_test: Ground-truth DataFrame covering the prediction horizon.
            Optional; passed through to the runner for metric computation.
        project_name: Cache subdirectory and model-file identifier.
            Defaults to ``"demo10"``.
        cache_home: Cache directory override.  When ``None``, the package
            default (``~/.spotforecast2_cache/``) is used.
        show_progress: Whether to emit progress messages during pipeline
            execution.

    Returns:
        DataFrame with a ``"forecast"`` column indexed by the forecast
        horizon timestamps.

    Raises:
        ValueError: If ``config`` is ``None``, or if the supplied
            ``config`` or ``dataframe`` is invalid (propagated from
            ``runner.run`` / ``BaseTask``).
        TypeError: If ``config`` is not ``None`` and not a ``ConfigMulti``
            instance.

    Examples:
        ```{python}
        import tempfile
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.configurator.config_multi import ConfigMulti
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import run_pipeline

        rng = np.random.default_rng(0)
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            rng.uniform(0, 100, size=(n, 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        df.index.name = "DateTime"

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(
                predict_size=4,
                train_size=pd.Timedelta(days=14),
                use_exogenous_features=False,
                use_outlier_detection=False,
                imputation_method="linear",
                agg_weights=[1.0, 1.0, -1.0],
            )
            result = run_pipeline(config=cfg, dataframe=df, project_name="doctest", cache_home=tmp)

        print(type(result))
        print(len(result))
        ```
    """
    if config is not None and not isinstance(config, ConfigMulti):
        raise TypeError(
            f"config must be a ConfigMulti instance or None; "
            f"got {type(config).__name__!r}."
        )

    if config is None:
        raise ValueError(
            "config is required: build a ConfigMulti and pass it explicitly, e.g.\n"
            "    ConfigMulti(\n"
            "        predict_size=24,\n"
            "        agg_weights=[...],   # one weight per target column; None = equal 1/n\n"
            "        bounds=[...],        # one (lower, upper) per target; None = no clipping\n"
            "    )\n"
            "Outlier `bounds` and aggregation `agg_weights` are domain-specific "
            "calibrations and are never defaulted to demo-dataset values."
        )

    return run(
        config,
        task="lazy",
        dataframe=dataframe,
        data_test=data_test,
        project_name=project_name,
        cache_home=cache_home,
        show_progress=show_progress,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for the N-to-1 forecasting pipeline.

    Parses command-line arguments, builds a ``ConfigMulti`` via
    ``_build_config_from_cli``, and delegates to ``run_pipeline``.
    Prints the forecast head to stdout.

    When ``argv`` is ``None``, ``sys.argv[1:]`` is used.  Pass an explicit
    list of strings to invoke programmatically with a specific argv (useful
    for testing without touching ``sys.argv``).

    Args:
        argv: Argument list.  ``None`` means read from ``sys.argv``.
    """
    parser = _make_arg_parser()
    args = parser.parse_args(argv)

    cache_home_path = Path(args.log_dir) if args.log_dir else None
    cfg = _build_config_from_cli(args)

    try:
        forecast = run_pipeline(config=cfg, cache_home=cache_home_path)
        print("\nForecast head:")
        print(forecast.head())
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nCritical failure: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

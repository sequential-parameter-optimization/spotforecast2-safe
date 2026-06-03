# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-step benchmark of the exogenous-feature pipeline.

Reconstructs, on synthetic data shaped like a real ENTSO-E load-forecasting
run (~17 months of hourly data, 15 weather variables, degree-2 polynomial
interactions capped by mutual information), the sub-steps that
``spotforecast2.multitask.BaseTask.build_exogenous_features`` executes through
the ``spotforecast2_safe`` helpers, and times each step in isolation:

1.  ``weather_windows`` — :class:`WindowFeatures` rolling 1D/7D mean/max/min,
    mirroring ``spotforecast2_safe.weather.features.get_weather_features``
    (the network fetch is replaced by a seeded synthetic frame).
2.  ``calendar`` — ``get_calendar_features``.
3.  ``day_night`` — ``get_day_night_features`` (astral).
4.  ``holidays`` — ``get_holiday_features``.
5.  ``combine`` — ``pd.concat`` + bfill/ffill of the four blocks.
6.  ``cyclical`` — ``apply_cyclical_encoding``.
7.  ``interactions`` — ``create_interaction_features`` (degree-2 pairwise).
8.  ``mi_cap`` — ``select_top_poly_features`` (mutual-information top-K).
9.  ``select`` — ``select_exogenous_features``.
10. ``merge`` — ``merge_data_and_covariates``.

The synthetic generator is fully seeded, so two runs on the same machine and
environment are directly comparable; this script is the single before/after
instrument for any optimisation of these helpers.

Usage::

    uv run python benchmarks/bench_exog_pipeline.py
    uv run python benchmarks/bench_exog_pipeline.py --profile
    uv run python benchmarks/bench_exog_pipeline.py --repeats 5 --out base.json

Steps whose first run exceeds ``--slow-threshold`` seconds are not repeated
(the figure of merit for slow steps is unambiguous anyway); fast steps report
the median and min over ``--repeats`` runs.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import statistics
import sys
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from astral import LocationInfo
from feature_engine.timeseries.forecasting import WindowFeatures

from spotforecast2_safe.calendar.features import (
    get_calendar_features,
    get_day_night_features,
)
from spotforecast2_safe.calendar.holiday import get_holiday_features
from spotforecast2_safe.manager.features import (
    apply_cyclical_encoding,
    create_interaction_features,
    merge_data_and_covariates,
    select_exogenous_features,
    select_top_poly_features,
)

# Shape of the chapter-14 ("team 4") live-submission run.
START = pd.Timestamp("2024-01-01 00:00", tz="UTC")
DATA_END = pd.Timestamp("2025-05-31 23:00", tz="UTC")  # ~17 months hourly
FORECAST_HORIZON = 24
LATITUDE = 51.5136  # Dortmund default used by the book pipeline
LONGITUDE = 7.4653
TIMEZONE = "UTC"
COUNTRY_CODE = "DE"
STATE = "NW"

# The 15 hourly variables the Open-Meteo fetch returns for the default config.
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]


def build_synthetic_inputs(seed: int, n_weather_vars: int) -> Dict[str, pd.DataFrame]:
    """Build a seeded synthetic target + weather frame of the real run's shape."""
    rng = np.random.default_rng(seed)
    cov_end = DATA_END + pd.Timedelta(hours=FORECAST_HORIZON)
    data_idx = pd.date_range(START, DATA_END, freq="h", tz="UTC")
    full_idx = pd.date_range(START, cov_end, freq="h", tz="UTC")

    hours = np.arange(len(full_idx))
    daily = np.sin(2 * np.pi * hours / 24)
    weekly = np.sin(2 * np.pi * hours / (24 * 7))
    yearly = np.sin(2 * np.pi * hours / (24 * 365))

    names = WEATHER_VARS[:n_weather_vars]
    weather = pd.DataFrame(
        {
            name: (
                10.0 * yearly
                + 3.0 * daily
                + rng.normal(0.0, 1.0, len(full_idx)).cumsum() * 0.01
                + rng.normal(0.0, 0.5, len(full_idx))
            )
            for name in names
        },
        index=full_idx,
    )

    # Load: weekly/daily seasonality + temperature dependence + noise.
    temp = weather[names[0]].to_numpy()[: len(data_idx)]
    load = (
        55_000.0
        + 8_000.0 * daily[: len(data_idx)]
        + 4_000.0 * weekly[: len(data_idx)]
        - 300.0 * temp
        + rng.normal(0.0, 1_000.0, len(data_idx))
    )
    data = pd.DataFrame({"Actual Load": load}, index=data_idx)

    return {"data": data, "weather_raw": weather, "cov_end": cov_end}


def run_pipeline(
    state: Dict,
    degree: int,
    max_poly_features: int,
    random_state: int,
    timings: Dict[str, List[float]],
    repeats: int,
    slow_threshold: float,
) -> Dict:
    """Execute the 10 steps once, timing each (with repeats for fast steps)."""

    def timed(name: str, fn: Callable[[], object]) -> object:
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        runs = [elapsed]
        if elapsed < slow_threshold:
            for _ in range(repeats - 1):
                t0 = time.perf_counter()
                fn()
                runs.append(time.perf_counter() - t0)
        timings[name] = runs
        print(
            f"  {name:<16s} {min(runs):>9.3f}s min  "
            f"{statistics.median(runs):>9.3f}s median  ({len(runs)} run(s))",
            flush=True,
        )
        return result

    data = state["data"]
    weather_raw = state["weather_raw"]
    cov_end = state["cov_end"]

    # 1. Weather windows (mirror of get_weather_features lines 212-239,
    #    minus the network fetch).
    def weather_step():
        wf = WindowFeatures(
            variables=list(weather_raw.columns),
            window=["1D", "7D"],
            functions=["mean", "max", "min"],
            freq="h",
        )
        out = wf.fit_transform(weather_raw)
        return out.bfill()

    weather_features = timed("weather_windows", weather_step)

    # 2-4. Calendar, day/night, holidays (real helpers, no network).
    calendar_features = timed(
        "calendar",
        lambda: get_calendar_features(
            start=START, cov_end=cov_end, freq="h", timezone=TIMEZONE
        ),
    )
    location = LocationInfo(latitude=LATITUDE, longitude=LONGITUDE, timezone=TIMEZONE)
    sun_light_features = timed(
        "day_night",
        lambda: get_day_night_features(
            start=START, cov_end=cov_end, location=location, freq="h", timezone=TIMEZONE
        ),
    )
    holiday_features = timed(
        "holidays",
        lambda: get_holiday_features(
            data=data,
            start=START,
            cov_end=cov_end,
            forecast_horizon=FORECAST_HORIZON,
            tz=TIMEZONE,
            freq="h",
            country_code=COUNTRY_CODE,
            state=STATE,
        ),
    )

    # 5. Combine (mirror of base.py step 5).
    def combine_step():
        out = pd.concat(
            [calendar_features, sun_light_features, weather_features, holiday_features],
            axis=1,
        )
        if out.isnull().sum().sum() != 0:
            out = out.bfill().ffill()
        return out

    exog = timed("combine", combine_step)

    # 6. Cyclical encoding.
    exog = timed(
        "cyclical", lambda: apply_cyclical_encoding(data=exog, drop_original=False)
    )

    # 7. Degree-2 interactions.
    exog_cyc = exog
    exog = timed(
        "interactions",
        lambda: create_interaction_features(
            exogenous_features=exog_cyc,
            weather_aligned=weather_raw,
            degree=degree,
        ),
    )

    poly_cols = [c for c in exog.columns if c.startswith("poly_")]
    print(f"  -> {len(poly_cols)} poly columns, {exog.shape[1]} total columns")

    # 8. Mutual-information cap (the suspected hotspot).
    keep = timed(
        "mi_cap",
        lambda: select_top_poly_features(
            exog[poly_cols],
            data["Actual Load"],
            max_poly_features=max_poly_features,
            random_state=random_state,
        ),
    )
    exog = exog.drop(columns=[c for c in poly_cols if c not in keep])

    # 9. Selection.
    exog_capped = exog
    exog_names = timed(
        "select",
        lambda: select_exogenous_features(
            exogenous_features=exog_capped,
            weather_aligned=weather_raw,
            include_weather_windows=True,
            include_holiday_features=True,
            poly_features_degree=degree,
        ),
    )

    # 10. Merge.
    merged = timed(
        "merge",
        lambda: merge_data_and_covariates(
            data=data,
            exogenous_features=exog_capped,
            target_columns=["Actual Load"],
            exog_features=exog_names,
            start=START,
            end=DATA_END,
            cov_end=cov_end,
            forecast_horizon=FORECAST_HORIZON,
            cast_dtype="float32",
        ),
    )

    return {
        "n_poly_cols": len(poly_cols),
        "n_exog_cols": exog_capped.shape[1],
        "n_selected": len(exog_names),
        "kept_poly": list(keep),
        "merged_shape": list(merged[0].shape),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repeats", type=int, default=3, help="runs per fast step")
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=30.0,
        help="steps slower than this (s) on their first run are not repeated",
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--max-poly-features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--weather-vars",
        type=int,
        default=len(WEATHER_VARS),
        help="number of synthetic weather variables (reduce for a quick run)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run one full pass under cProfile and print the top 25 by cumtime",
    )
    parser.add_argument("--out", type=str, default=None, help="write JSON results")
    args = parser.parse_args(argv)

    import sklearn

    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
    }
    print("environment:", json.dumps(env))
    print(
        f"shape: {START} .. {DATA_END} (+{FORECAST_HORIZON}h horizon), "
        f"{args.weather_vars} weather vars, degree={args.degree}, "
        f"max_poly_features={args.max_poly_features}"
    )

    state = build_synthetic_inputs(args.seed, args.weather_vars)
    timings: Dict[str, List[float]] = {}
    profiler: cProfile.Profile | None = None

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    t_total = time.perf_counter()
    meta = run_pipeline(
        state,
        degree=args.degree,
        max_poly_features=args.max_poly_features,
        random_state=args.random_state,
        timings=timings,
        repeats=1 if args.profile else args.repeats,
        slow_threshold=args.slow_threshold,
    )
    total = time.perf_counter() - t_total

    if args.profile and profiler is not None:
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(25)
        print("\ncProfile (top 25 by cumulative time):")
        print(stream.getvalue())

    print(f"\ntotal wall time (single pass + repeats of fast steps): {total:.3f}s")
    print(f"meta: {json.dumps(meta)}")

    if args.out:
        payload = {
            "env": env,
            "args": vars(args),
            "timings_s": timings,
            "meta": meta,
            "total_s": total,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

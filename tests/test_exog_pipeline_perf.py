# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Performance-regression guard for the polynomial mutual-information cap.

``select_top_poly_features`` is the dominant cost of the exogenous-feature
pipeline on realistic inputs (see ``benchmarks/bench_exog_pipeline.py``). The
guard below runs the ranking on a wide, multi-year-sized matrix with the
default knobs (``n_jobs=-1``, ``mi_sample_size=4000``) and fails if it blows
past a generous wall-clock ceiling. Before those knobs existed, this shape
took minutes single-threaded on a laptop; with them it completes in seconds,
so the ceiling separates the two regimes cleanly even on slow CI runners.

Marked ``slow``: deselect with ``-m "not slow"`` for a quick local loop.
"""

import time

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.manager.features import select_top_poly_features


@pytest.mark.slow
def test_mi_cap_completes_within_ceiling():
    rng = np.random.default_rng(42)
    n_rows, n_cols = 50_000, 200
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="h", tz="UTC")
    signal = rng.normal(0, 1, n_rows)
    y = pd.Series(signal, index=idx, name="target")
    poly = pd.DataFrame(
        rng.normal(0, 1, (n_rows, n_cols)),
        index=idx,
        columns=[f"poly_{i:03d}" for i in range(n_cols)],
    )
    # Two informative columns that must survive the cap.
    poly["poly_signal_a"] = signal + rng.normal(0, 0.05, n_rows)
    poly["poly_signal_b"] = signal * 0.5 + rng.normal(0, 0.2, n_rows)

    t0 = time.perf_counter()
    top = select_top_poly_features(poly, y, max_poly_features=10, random_state=42)
    elapsed = time.perf_counter() - t0

    assert len(top) == 10
    assert "poly_signal_a" in top
    assert "poly_signal_b" in top
    # Generous ceiling: ~2 s on a laptop with the default knobs; minutes
    # without them. 60 s keeps slow CI runners green while still failing
    # loudly if the fast path regresses to full-data single-threaded scoring.
    assert elapsed < 60.0, f"MI cap took {elapsed:.1f}s (ceiling 60s)"

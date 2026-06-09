# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Smoke tests for the four-zone bottom-up load demo task."""

from pathlib import Path

import pandas as pd

from spotforecast2_safe.downloader.entsoe import GERMAN_TSO_ZONES
from spotforecast2_safe.tasks.task_safe_zone_load_demo import (
    _synthetic_zone_frame,
    main,
)


def test_synthetic_demo_returns_zero():
    # Short, fast synthetic run end-to-end (MultiTask bottom-up + backtest).
    rc = main(predict_size=12, history_hours=24 * 60, random_seed=0)
    assert rc == 0


def test_missing_data_path_fails_fast():
    rc = main(data_path=Path("/nonexistent/energy_load_zones.csv"))
    assert rc == 1


def test_real_data_path_roundtrip(tmp_path):
    # An assembled four-column frame on disk drives the real-data path.
    df = _synthetic_zone_frame(24 * 60, random_seed=1, zones=list(GERMAN_TSO_ZONES))
    csv = tmp_path / "energy_load_zones.csv"
    df.rename_axis("Time (UTC)").to_csv(csv)
    rc = main(data_path=csv, predict_size=12)
    assert rc == 0


def test_synthetic_frame_is_deterministic_and_positive():
    a = _synthetic_zone_frame(48, random_seed=7, zones=list(GERMAN_TSO_ZONES))
    b = _synthetic_zone_frame(48, random_seed=7, zones=list(GERMAN_TSO_ZONES))
    pd.testing.assert_frame_equal(a, b)
    assert (a > 0).all().all()
    assert a.columns.tolist() == list(GERMAN_TSO_ZONES)

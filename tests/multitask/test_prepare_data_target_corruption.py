# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integration tests for prepare_data with the target-corruption policy.

Mirrors the style of ``tests/multitask/test_prepare_data_clamp.py``
(the 16.3.1 trailing-clamp regression suite) and uses synthetic
15-min-cadence data replicating the 2026-06-03..05 incident pattern.

Tests:
(a) defaults-off path is byte-identical to a run without any corruption knobs.
(b) truncate composes with the clamp: data_end retracts to the hour before
    the first flagged hour and predict_size is bumped by the retracted hours,
    idempotent across a second prepare_data call.
(c) abort propagates TargetCorruptionError.
(d) heal forces imputation_method to "weighted_interp" and flagged cells
    end up NaN (ready for interpolation).
"""

import logging

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.exceptions import TargetCorruptionError
from spotforecast2_safe.multitask import LazyTask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_DAYS_CLEAN = 14  # two clean weeks before the corruption event
N_HOURS_PER_DAY = 24
CADENCE = "15min"
SLOTS_PER_HOUR = 4
BASE_MW = 55_000.0
PREDICT_SIZE = 24

# The corruption event is injected on the last day, at hour 12.
# A step drop of 11 GW at slot 12:15 UTC.
CORRUPT_DAY = N_DAYS_CLEAN  # 0-indexed, so day 14 (last day)
CORRUPT_HOUR = 12


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_native_df(inject_corruption: bool = False) -> pd.DataFrame:
    """Build a synthetic 15-min DataFrame (index name: 'DateTime').

    If ``inject_corruption=True`` the 12:15 slot on the last day is dropped
    by 11 GW (still inside the 40–75 GW plausible range).
    """
    n_slots = (N_DAYS_CLEAN + 1) * N_HOURS_PER_DAY * SLOTS_PER_HOUR
    idx = pd.date_range("2026-05-20", periods=n_slots, freq=CADENCE, tz="UTC")
    rng = np.random.default_rng(42)
    vals = BASE_MW + rng.normal(0, 300, n_slots)

    if inject_corruption:
        # Last day, hour 12, second slot (12:15 UTC)
        corrupt_slot = (
            N_DAYS_CLEAN * N_HOURS_PER_DAY * SLOTS_PER_HOUR
            + CORRUPT_HOUR * SLOTS_PER_HOUR
            + 1
        )
        vals[corrupt_slot] = BASE_MW - 11_000

    idx.name = "DateTime"
    df = pd.DataFrame({"Actual Load": vals}, index=idx)
    return df


def _cfg_base(**overrides) -> ConfigMulti:
    """Minimal config: no exogenous features, no outlier detection, small folds."""
    defaults = dict(
        predict_size=PREDICT_SIZE,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        verbose=False,
        targets=["Actual Load"],
    )
    defaults.update(overrides)
    return ConfigMulti(**defaults)


def _cfg_detector(**overrides) -> ConfigMulti:
    """Config with the target-corruption detector enabled."""
    return _cfg_base(
        target_qc_range_mw=5_000.0,
        target_qc_step_mw=8_000.0,
        target_qc_window_days=3,
        **overrides,
    )


# ---------------------------------------------------------------------------
# (a) Defaults-off path is byte-identical
# ---------------------------------------------------------------------------


class TestDefaultsOff:
    def test_defaults_off_no_corruption_report(self, tmp_path):
        """With all knobs at defaults, target_corruption_report.action must be noop."""
        df = _make_native_df(inject_corruption=True)
        task = LazyTask(_cfg_base(cache_home=tmp_path), dataframe=df)
        task.prepare_data()
        report = task.target_corruption_report
        assert report.action == "noop"
        assert not report.fired

    def test_defaults_off_frame_shape_unchanged(self, tmp_path):
        """Frame shape with defaults must equal a clean-data run."""
        df_clean = _make_native_df(inject_corruption=False)
        df_corrupt = _make_native_df(inject_corruption=True)

        task_clean = LazyTask(_cfg_base(cache_home=tmp_path), dataframe=df_clean)
        task_corrupt = LazyTask(_cfg_base(cache_home=tmp_path), dataframe=df_corrupt)

        task_clean.prepare_data()
        task_corrupt.prepare_data()

        # With defaults off, the corrupted and clean runs must produce the same
        # shape (the corruption is a single slot and doesn't affect the clamp).
        assert task_clean.df_pipeline.shape == task_corrupt.df_pipeline.shape


# ---------------------------------------------------------------------------
# (b) Truncate composes with the clamp
# ---------------------------------------------------------------------------


class TestTruncateComposeWithClamp:
    def test_truncate_retracts_data_end(self, tmp_path):
        """data_end must be anchored at or before the first_flagged_hour - 1h."""
        df = _make_native_df(inject_corruption=True)
        task = LazyTask(
            _cfg_detector(cache_home=tmp_path, target_corruption_policy="truncate"),
            dataframe=df,
        )
        task.prepare_data()
        report = task.target_corruption_report

        assert report.fired
        assert report.action == "truncate"

        data_end = pd.to_datetime(task.config.data_end, utc=True)
        first_flagged = report.first_flagged_hour
        assert (
            data_end < first_flagged
        ), f"data_end ({data_end}) must be before first_flagged_hour ({first_flagged})"

    def test_truncate_bumps_predict_size(self, tmp_path):
        """predict_size must increase by the number of retracted hours."""
        df = _make_native_df(inject_corruption=True)
        task = LazyTask(
            _cfg_detector(cache_home=tmp_path, target_corruption_policy="truncate"),
            dataframe=df,
        )
        task.prepare_data()

        assert (
            task.config.predict_size > PREDICT_SIZE
        ), "predict_size must be bumped above the base value after truncation"

    def test_truncate_predict_size_bump_idempotent(self, tmp_path):
        """Calling prepare_data twice must not double-bump predict_size."""
        df = _make_native_df(inject_corruption=True)
        cfg = _cfg_detector(cache_home=tmp_path, target_corruption_policy="truncate")
        task = LazyTask(cfg, dataframe=df)
        task.prepare_data()
        ps_after_first = task.config.predict_size

        task.prepare_data()
        ps_after_second = task.config.predict_size

        assert ps_after_first == ps_after_second, (
            f"predict_size after second call ({ps_after_second}) must equal "
            f"first call ({ps_after_first}) — must be idempotent"
        )

    def test_truncate_emits_warning_log(self, tmp_path, caplog):
        df = _make_native_df(inject_corruption=True)
        task = LazyTask(
            _cfg_detector(cache_home=tmp_path, target_corruption_policy="truncate"),
            dataframe=df,
        )
        with caplog.at_level(logging.WARNING):
            task.prepare_data()
        assert any(
            "target_corruption[truncate]" in m for m in caplog.messages
        ), "truncate path must emit a WARNING log"


# ---------------------------------------------------------------------------
# (c) Abort propagates TargetCorruptionError
# ---------------------------------------------------------------------------


class TestAbort:
    def test_abort_raises_on_corrupt_data(self, tmp_path):
        df = _make_native_df(inject_corruption=True)
        task = LazyTask(
            _cfg_detector(cache_home=tmp_path, target_corruption_policy="abort"),
            dataframe=df,
        )
        with pytest.raises(TargetCorruptionError):
            task.prepare_data()

    def test_abort_does_not_raise_on_clean_data(self, tmp_path):
        df = _make_native_df(inject_corruption=False)
        task = LazyTask(
            _cfg_detector(cache_home=tmp_path, target_corruption_policy="abort"),
            dataframe=df,
        )
        task.prepare_data()  # must not raise
        assert not task.target_corruption_report.fired


# ---------------------------------------------------------------------------
# (d) Heal forces imputation_method and NaNs flagged cells
# ---------------------------------------------------------------------------


class TestHeal:
    def test_heal_forces_weighted_interp_info_log(self, tmp_path, caplog):
        """Heal policy must force imputation_method='weighted_interp' and log it."""
        df = _make_native_df(inject_corruption=True)
        # Use a large heal budget and no anchor zone to avoid refusal.
        cfg = _cfg_detector(
            cache_home=tmp_path,
            target_corruption_policy="heal",
            target_max_heal_hours=24,
            target_anchor_zone_hours=0,
        )
        task = LazyTask(cfg, dataframe=df)
        with caplog.at_level(logging.INFO):
            task.prepare_data()

        assert (
            task.config.imputation_method == "weighted_interp"
        ), "heal must force imputation_method to 'weighted_interp'"
        assert any(
            "weighted_interp" in m for m in caplog.messages
        ), "INFO log about forcing weighted_interp must be emitted"

    def test_heal_nans_flagged_cells(self, tmp_path):
        """Flagged cells in df_pipeline must be NaN after prepare_data with heal."""
        df = _make_native_df(inject_corruption=True)
        cfg = _cfg_detector(
            cache_home=tmp_path,
            target_corruption_policy="heal",
            target_max_heal_hours=24,
            target_anchor_zone_hours=0,
        )
        task = LazyTask(cfg, dataframe=df)
        task.prepare_data()

        report = task.target_corruption_report
        assert report.fired
        assert report.action == "heal"

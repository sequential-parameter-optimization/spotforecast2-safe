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

# The corruption is injected at 12:15 UTC on the last day (CORRUPT_HOUR=12).
# The step rule flags both the 12:00 hour and its predecessor (00:45->01:00
# pair is the standard case, but here 12:00->12:15 flags only 12:00).
# All slots from 12:00 onward become NaN; the trailing clamp retracts
# data_end to 11:00.  The last slot of the native series is 23:45 UTC, so
# floor(pre_policy_last_target) = 23:00.  Bump = 23:00 - 11:00 = 12 h.
_EXPECTED_BUMP = 12


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

        data_end = pd.to_datetime(task.run_state.data_end, utc=True)
        first_flagged = report.first_flagged_hour
        assert (
            data_end < first_flagged
        ), f"data_end ({data_end}) must be before first_flagged_hour ({first_flagged})"

    def test_truncate_bumps_predict_size_exactly(self, tmp_path):
        """predict_size must increase by exactly the number of retracted hours."""
        df = _make_native_df(inject_corruption=True)
        task = LazyTask(
            _cfg_detector(cache_home=tmp_path, target_corruption_policy="truncate"),
            dataframe=df,
        )
        task.prepare_data()

        assert task.config.predict_size == PREDICT_SIZE + _EXPECTED_BUMP, (
            f"predict_size must be exactly {PREDICT_SIZE + _EXPECTED_BUMP} "
            f"(base {PREDICT_SIZE} + bump {_EXPECTED_BUMP}), "
            f"got {task.config.predict_size}"
        )

    def test_truncate_absolute_end_invariant(self, tmp_path):
        """data_end + bumped_predict_size * 1h == untruncated_data_end + PREDICT_SIZE * 1h.

        The forecast window covers the same absolute span as it would have
        without the corruption.
        """
        df = _make_native_df(inject_corruption=True)
        # Untruncated data_end is the hourly floor of the last native slot.
        untruncated_data_end = df.index.max().tz_convert("UTC").floor("h")

        task = LazyTask(
            _cfg_detector(cache_home=tmp_path, target_corruption_policy="truncate"),
            dataframe=df,
        )
        task.prepare_data()

        data_end_post = pd.to_datetime(task.run_state.data_end, utc=True)
        lhs = data_end_post + task.config.predict_size * pd.Timedelta(hours=1)
        rhs = untruncated_data_end + PREDICT_SIZE * pd.Timedelta(hours=1)
        assert lhs == rhs, (
            f"Absolute-end invariant violated: data_end({data_end_post}) + "
            f"predict_size({task.config.predict_size})h = {lhs} != "
            f"untruncated_data_end({untruncated_data_end}) + {PREDICT_SIZE}h = {rhs}"
        )

    def test_truncate_absolute_end_invariant_mid_hour_last_slot(self, tmp_path):
        """Invariant holds when the last native slot ends at :30 (not :45).

        This pins the floor-not-ceil choice: a partially observed hour still
        aggregates to a value at its floor, so using ceil would over-extend
        cov_end by one hour.
        """
        # Build a series that ends at :30 instead of :45 (drop last native slot).
        n_slots = (N_DAYS_CLEAN + 1) * N_HOURS_PER_DAY * SLOTS_PER_HOUR - 1
        idx = pd.date_range("2026-05-20", periods=n_slots, freq=CADENCE, tz="UTC")
        rng = np.random.default_rng(42)
        vals = BASE_MW + rng.normal(0, 300, n_slots)
        corrupt_slot = (
            N_DAYS_CLEAN * N_HOURS_PER_DAY * SLOTS_PER_HOUR
            + CORRUPT_HOUR * SLOTS_PER_HOUR
            + 1
        )
        vals[corrupt_slot] = BASE_MW - 11_000
        idx.name = "DateTime"
        df = pd.DataFrame({"Actual Load": vals}, index=idx)

        assert df.index[-1].minute == 30, "Last slot must be at :30 for this variant."

        untruncated_data_end = df.index.max().tz_convert("UTC").floor("h")

        task = LazyTask(
            _cfg_detector(cache_home=tmp_path, target_corruption_policy="truncate"),
            dataframe=df,
        )
        task.prepare_data()

        data_end_post = pd.to_datetime(task.run_state.data_end, utc=True)
        lhs = data_end_post + task.config.predict_size * pd.Timedelta(hours=1)
        rhs = untruncated_data_end + PREDICT_SIZE * pd.Timedelta(hours=1)
        assert lhs == rhs, f"Mid-hour invariant violated: {lhs} != {rhs}"

    def test_truncate_absolute_end_invariant_early_end_train_default(self, tmp_path):
        """Invariant holds when end_train_default is explicitly set earlier than data end.

        end_train_default affects the cutoff for the anchor-zone check but must
        not perturb the bump arithmetic (review finding 9).
        """
        df = _make_native_df(inject_corruption=True)
        untruncated_data_end = df.index.max().tz_convert("UTC").floor("h")

        # Set end_train_default 3 days before the data end.
        early_cutoff = "2026-05-31T00:00"
        task = LazyTask(
            _cfg_detector(
                cache_home=tmp_path,
                target_corruption_policy="truncate",
                end_train_default=early_cutoff,
            ),
            dataframe=df,
        )
        task.prepare_data()

        assert task.config.predict_size == PREDICT_SIZE + _EXPECTED_BUMP, (
            f"end_train_default must not perturb bump; expected "
            f"{PREDICT_SIZE + _EXPECTED_BUMP}, got {task.config.predict_size}"
        )

        data_end_post = pd.to_datetime(task.run_state.data_end, utc=True)
        lhs = data_end_post + task.config.predict_size * pd.Timedelta(hours=1)
        rhs = untruncated_data_end + PREDICT_SIZE * pd.Timedelta(hours=1)
        assert lhs == rhs, (
            f"Absolute-end invariant violated with early end_train_default: "
            f"{lhs} != {rhs}"
        )

    def test_truncate_predict_size_bump_idempotent(self, tmp_path):
        """Calling prepare_data twice on the same corrupt data must not double-bump."""
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

    def test_predict_size_reset_on_recovery(self, tmp_path, caplog):
        """After truncation bumps predict_size, feeding clean data resets it."""
        df_corrupt = _make_native_df(inject_corruption=True)
        df_clean = _make_native_df(inject_corruption=False)

        cfg = _cfg_detector(cache_home=tmp_path, target_corruption_policy="truncate")
        task = LazyTask(cfg, dataframe=df_corrupt)
        task.prepare_data()
        assert task.config.predict_size == PREDICT_SIZE + _EXPECTED_BUMP

        # Feed clean data via demo_data parameter to mirror how a caller would
        # reuse the same task object with fresh data.
        with caplog.at_level(logging.WARNING):
            task.prepare_data(demo_data=df_clean)

        assert task.config.predict_size == PREDICT_SIZE, (
            f"predict_size must reset to {PREDICT_SIZE} after clean run, "
            f"got {task.config.predict_size}"
        )
        assert any(
            "resetting predict_size" in m for m in caplog.messages
        ), "A WARNING about resetting predict_size must be logged on recovery."


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
    def _make_heal_cfg(self, tmp_path):
        return _cfg_detector(
            cache_home=tmp_path,
            target_corruption_policy="heal",
            target_max_heal_hours=24,
            target_anchor_zone_hours=0,
        )

    def test_heal_sets_tc_force_flag_after_prepare_data(self, tmp_path):
        """After prepare_data with heal policy, task._tc_force_weighted_interp must be True."""
        df = _make_native_df(inject_corruption=True)
        task = LazyTask(self._make_heal_cfg(tmp_path), dataframe=df)
        task.prepare_data()

        assert task._tc_force_weighted_interp is True, (
            "prepare_data must set _tc_force_weighted_interp=True for heal policy "
            "when the detector is configured and fires."
        )

    def test_heal_forces_weighted_interp_info_log(self, tmp_path, caplog):
        """impute() must temporarily use 'weighted_interp', emit an INFO log,
        and restore the original imputation_method afterwards."""
        df = _make_native_df(inject_corruption=True)
        cfg = self._make_heal_cfg(tmp_path)
        original_method = cfg.imputation_method
        task = LazyTask(cfg, dataframe=df)

        with caplog.at_level(logging.INFO):
            task.prepare_data()

        # After prepare_data: config.imputation_method is UNCHANGED.
        assert cfg.imputation_method == original_method, (
            "prepare_data must NOT permanently change config.imputation_method; "
            f"expected '{original_method}', got '{cfg.imputation_method}'"
        )
        # _tc_force_weighted_interp is set by prepare_data.
        assert task._tc_force_weighted_interp is True

        # Call impute(): the INFO log is emitted during this call.
        caplog.clear()
        with caplog.at_level(logging.INFO):
            task.impute()

        assert any(
            "target_corruption[heal]" in m and "weighted_interp" in m
            for m in caplog.messages
        ), "impute() must emit an INFO log mentioning 'target_corruption[heal]' and 'weighted_interp'."

        # After impute(): config.imputation_method is restored to original.
        assert cfg.imputation_method == original_method, (
            f"impute() must restore imputation_method to '{original_method}' in "
            f"its finally block; got '{cfg.imputation_method}'"
        )

        # Flagged cells are no longer NaN (interpolated).
        report = task.target_corruption_report
        first_flagged = report.first_flagged_hour
        if first_flagged is not None:
            hour_data = task.df_pipeline.loc[
                first_flagged : first_flagged + pd.Timedelta(minutes=59), "Actual Load"
            ]
            assert (
                not hour_data.isna().any()
            ), "After impute(), flagged hourly cells must no longer be NaN."

        # weight_func is not None (healed slots have zero weight).
        assert (
            task.weight_func is not None
        ), "impute() must produce a non-None weight_func for the heal path."

    def test_heal_nans_flagged_cells_before_impute(self, tmp_path):
        """After prepare_data (BEFORE impute), df_pipeline must contain NaN at
        the flagged hourly timestamps (contract B)."""
        df = _make_native_df(inject_corruption=True)
        task = LazyTask(self._make_heal_cfg(tmp_path), dataframe=df)
        task.prepare_data()

        report = task.target_corruption_report
        assert report.fired
        assert report.action == "heal"
        assert report.first_flagged_hour is not None

        # The policy NaNs whole hours at native cadence; hourly aggregation of
        # an all-NaN hour yields NaN.  Check the first flagged hour in
        # df_pipeline (which is already aggregated to hourly by prepare_data).
        first_flagged = report.first_flagged_hour
        flagged_slot = task.df_pipeline.loc[
            first_flagged : first_flagged + pd.Timedelta(minutes=59), "Actual Load"
        ]
        assert flagged_slot.isna().any(), (
            "Before impute(), df_pipeline must contain NaN at the flagged hour "
            f"({first_flagged}). Got: {flagged_slot.tolist()}"
        )

    def test_heal_no_config_leakage_across_tasks(self, tmp_path):
        """Two tasks sharing the same config object must not see imputation_method
        changed by the first task's prepare_data+impute cycle (contract A.2)."""
        df_corrupt = _make_native_df(inject_corruption=True)
        df_clean = _make_native_df(inject_corruption=False)

        cfg = self._make_heal_cfg(tmp_path)
        original_method = cfg.imputation_method

        task1 = LazyTask(cfg, dataframe=df_corrupt)
        task1.prepare_data()
        task1.impute()

        assert cfg.imputation_method == original_method, (
            f"task1.impute() must not leave config.imputation_method changed; "
            f"expected '{original_method}', got '{cfg.imputation_method}'"
        )

        # Second task on clean data: no heal fires, method must still be original.
        task2 = LazyTask(cfg, dataframe=df_clean)
        task2.prepare_data()
        task2.impute()

        assert cfg.imputation_method == original_method, (
            f"task2.impute() on clean data must not change config.imputation_method; "
            f"expected '{original_method}', got '{cfg.imputation_method}'"
        )

# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for RunState and the derived-field extraction from ConfigMulti.

Covers:
- Fresh RunState has all fields as None.
- After prepare_data() the 6 window fields + targets are populated on run_state.
- After _setup_training_window() the 2 training-window fields are populated.
- Derived fields are NOT declared params on config (_PARAM_NAMES, get_params()).
- Derived fields are NOT accessible on config after pipeline runs (no mirror shim).
- set_params() raises ValueError for derived fields.
- Clamp semantics: user end_train_default beyond data extent → end_train_ts clamps
  to data end; an explicitly earlier cutoff is honoured unchanged.
- config.targets is unchanged by the pipeline (user input preserved).
- No DeprecationWarning is emitted by the pipeline.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.multitask import MultiTask
from spotforecast2_safe.multitask.run_state import RunState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_weeks: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 24 * 7 * n_weeks
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    return pd.DataFrame({"load": rng.normal(100, 10, n)}, index=idx)


def _make_task(tmp_path, df=None, **cfg_kwargs):
    """Create a MultiTask with minimal config for unit tests."""
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        cache_home=tmp_path,
    )
    defaults.update(cfg_kwargs)
    cfg = ConfigMulti(**defaults)
    if df is None:
        df = _make_df()
    return MultiTask(cfg, dataframe=df)


# ---------------------------------------------------------------------------
# RunState: fresh instance
# ---------------------------------------------------------------------------


class TestRunStateFreshInstance:
    """A new RunState must have all fields set to None."""

    def test_start_download_is_none(self):
        assert RunState().start_download is None

    def test_end_download_is_none(self):
        assert RunState().end_download is None

    def test_data_start_is_none(self):
        assert RunState().data_start is None

    def test_data_end_is_none(self):
        assert RunState().data_end is None

    def test_cov_start_is_none(self):
        assert RunState().cov_start is None

    def test_cov_end_is_none(self):
        assert RunState().cov_end is None

    def test_end_train_ts_is_none(self):
        assert RunState().end_train_ts is None

    def test_start_train_ts_is_none(self):
        assert RunState().start_train_ts is None

    def test_targets_is_none(self):
        assert RunState().targets is None

    def test_dataclass_equality(self):
        assert RunState() == RunState()


# ---------------------------------------------------------------------------
# RunState: populated after prepare_data
# ---------------------------------------------------------------------------


class TestRunStateAfterPrepareData:
    """After prepare_data() the 6 window fields + targets must be populated."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.task = _make_task(tmp_path)
        self.task.prepare_data()

    def test_start_download_populated(self):
        assert self.task.run_state.start_download is not None

    def test_end_download_populated(self):
        assert self.task.run_state.end_download is not None

    def test_data_start_populated(self):
        assert isinstance(self.task.run_state.data_start, pd.Timestamp)

    def test_data_end_populated(self):
        assert isinstance(self.task.run_state.data_end, pd.Timestamp)

    def test_cov_start_populated(self):
        assert isinstance(self.task.run_state.cov_start, pd.Timestamp)

    def test_cov_end_populated(self):
        assert isinstance(self.task.run_state.cov_end, pd.Timestamp)

    def test_cov_end_after_data_end(self):
        """cov_end must extend beyond data_end by predict_size hours."""
        assert self.task.run_state.cov_end > self.task.run_state.data_end

    def test_targets_populated(self):
        assert self.task.run_state.targets is not None
        assert len(self.task.run_state.targets) > 0

    def test_targets_are_list_of_strings(self):
        assert all(isinstance(t, str) for t in self.task.run_state.targets)

    def test_end_train_ts_still_none_before_window_setup(self):
        """end_train_ts is only set by _setup_training_window, not prepare_data."""
        assert self.task.run_state.end_train_ts is None

    def test_start_train_ts_still_none_before_window_setup(self):
        assert self.task.run_state.start_train_ts is None


# ---------------------------------------------------------------------------
# RunState: populated after _setup_training_window
# ---------------------------------------------------------------------------


class TestRunStateAfterTrainingWindow:
    """After _setup_training_window() the 2 window timestamps must be set."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.task = _make_task(tmp_path)
        self.task.prepare_data()
        self.task._setup_training_window()

    def test_end_train_ts_is_timestamp(self):
        assert isinstance(self.task.run_state.end_train_ts, pd.Timestamp)

    def test_start_train_ts_is_timestamp(self):
        assert isinstance(self.task.run_state.start_train_ts, pd.Timestamp)

    def test_start_train_ts_before_end_train_ts(self):
        assert self.task.run_state.start_train_ts < self.task.run_state.end_train_ts

    def test_start_train_ts_not_before_data_start(self):
        """start_train_ts is floored at the earliest data row."""
        assert self.task.run_state.start_train_ts >= self.task.run_state.data_start


# ---------------------------------------------------------------------------
# Config no longer has derived fields as params
# ---------------------------------------------------------------------------


class TestDerivedFieldsRemovedFromConfig:
    """The 8 derived fields must NOT appear in _PARAM_NAMES or get_params()."""

    DERIVED = [
        "start_download",
        "end_download",
        "data_start",
        "data_end",
        "cov_start",
        "cov_end",
        "end_train_ts",
        "start_train_ts",
    ]

    @pytest.mark.parametrize("field", DERIVED)
    def test_not_in_param_names(self, field):
        assert field not in ConfigMulti._PARAM_NAMES, (
            f"'{field}' is still in ConfigMulti._PARAM_NAMES; "
            "derived fields must move to RunState."
        )

    @pytest.mark.parametrize("field", DERIVED)
    def test_not_in_get_params(self, field):
        p = ConfigMulti().get_params()
        assert field not in p, (
            f"'{field}' leaked into get_params(); "
            "derived fields must not appear in get_params()."
        )

    @pytest.mark.parametrize("field", DERIVED)
    def test_set_params_raises(self, field):
        cfg = ConfigMulti()
        with pytest.raises(ValueError, match=field):
            cfg.set_params(**{field: None})


# ---------------------------------------------------------------------------
# No config mirror: derived values are NOT accessible via config
# ---------------------------------------------------------------------------


class TestNoConfigMirror:
    """After pipeline runs, derived fields must NOT be set on config (shim removed)."""

    # All 8 derived fields — including the two set only by _setup_training_window.
    DERIVED = [
        "start_download",
        "end_download",
        "data_start",
        "data_end",
        "cov_start",
        "cov_end",
        "end_train_ts",
        "start_train_ts",
    ]

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.task = _make_task(tmp_path)
        self.task.prepare_data()
        # Also run _setup_training_window so end_train_ts / start_train_ts are
        # populated on run_state — the assertion checks they are still absent
        # from config even after both pipeline phases complete.
        self.task._setup_training_window()

    @pytest.mark.parametrize("field", DERIVED)
    def test_derived_field_not_on_config(self, field):
        """After prepare_data() + _setup_training_window(), config must NOT carry any derived field."""
        assert not hasattr(self.task.config, field), (
            f"config.{field} is set after the pipeline ran; "
            "the mirror shim has been removed — derived fields live only on run_state."
        )

    def test_run_state_carries_derived_values(self):
        """run_state must carry all populated derived values."""
        assert self.task.run_state.data_start is not None
        assert self.task.run_state.data_end is not None
        assert self.task.run_state.cov_end is not None
        assert self.task.run_state.start_download is not None
        assert self.task.run_state.end_train_ts is not None
        assert self.task.run_state.start_train_ts is not None

    def test_targets_not_on_config_after_pipeline(self):
        """targets is not mirrored onto config — config.targets keeps user input (None)."""
        # config.targets must remain as the user passed it (None here)
        assert self.task.config.targets is None
        # run_state.targets holds the resolved list
        assert self.task.run_state.targets == ["load"]


# ---------------------------------------------------------------------------
# No DeprecationWarning emitted by the pipeline
# ---------------------------------------------------------------------------


class TestNoDeprecationWarning:
    """The pipeline must not emit any DeprecationWarning after shim removal."""

    def test_no_deprecation_warning_on_prepare_data(self, tmp_path):
        task = _make_task(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.prepare_data()
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0, (
            f"prepare_data() emitted {len(dep_warnings)} DeprecationWarning(s); "
            "the mirror shim has been removed so no such warnings should be emitted."
        )

    def test_no_deprecation_warning_on_setup_training_window(self, tmp_path):
        """_setup_training_window must not emit DeprecationWarning either."""
        task = _make_task(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.prepare_data()
            task._setup_training_window()
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0


# ---------------------------------------------------------------------------
# Clamp semantics for end_train_ts
# ---------------------------------------------------------------------------


class TestClampSemantics:
    """end_train_default clamping moved from prepare_data to _setup_training_window."""

    def test_stale_end_train_default_clamped_to_data_end(self, tmp_path):
        """When end_train_default is far in the future, end_train_ts == data_end."""
        task = _make_task(tmp_path, end_train_default="2099-12-31 00:00+00:00")
        task.prepare_data()
        task._setup_training_window()
        # The clamp should prevent end_train_ts from exceeding data_end.
        assert task.run_state.end_train_ts <= task.run_state.data_end

    def test_earlier_explicit_cutoff_honoured(self, tmp_path):
        """An explicit end_train_default earlier than the data is honoured."""
        # Data covers 2023-01; set cutoff to early 2023-01
        task = _make_task(tmp_path, end_train_default="2023-01-07 00:00+00:00")
        task.prepare_data()
        task._setup_training_window()
        expected = pd.Timestamp("2023-01-07 00:00+00:00")
        assert task.run_state.end_train_ts == expected

    def test_config_end_train_default_not_mutated(self, tmp_path):
        """prepare_data must never mutate config.end_train_default."""
        original = "2099-12-31 00:00+00:00"
        task = _make_task(tmp_path, end_train_default=original)
        task.prepare_data()
        assert task.config.end_train_default == original


# ---------------------------------------------------------------------------
# config.targets unchanged by pipeline
# ---------------------------------------------------------------------------


class TestConfigTargetsPreserved:
    """config.targets must hold only what the user passed; pipeline does not modify it."""

    def test_none_targets_preserved(self, tmp_path):
        """When user passes no targets, config.targets remains None."""
        task = _make_task(tmp_path, targets=None)
        task.prepare_data()
        assert task.config.targets is None

    def test_explicit_targets_preserved(self, tmp_path):
        """When user passes explicit targets, config.targets is unchanged."""
        task = _make_task(tmp_path, targets=["load"])
        task.prepare_data()
        assert task.config.targets == ["load"]

    def test_run_state_targets_has_resolved_list(self, tmp_path):
        """The resolved list (after column reconciliation) lives on run_state."""
        task = _make_task(tmp_path, targets=None)
        task.prepare_data()
        assert task.run_state.targets is not None
        assert len(task.run_state.targets) > 0

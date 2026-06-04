# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for single-target behaviour in the multitask pipeline.

Adapted from spotforecast2/tests/test_multitask_single_target.py; the
make_plot monkeypatch from the sibling package is replaced with an assertion
that the no-op hooks run without raising and return None.

Covers:
- _aggregate_and_show returns the per-target package unchanged for a single target
- _aggregate_and_show stores the result in agg_results
- Single-target path does NOT call agg_predictor
- Multi-target path still invokes agg_predictor (regression guard)
- show=True with single target does NOT raise
- show=True produces identical future_pred to show=False (no-op hooks)
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.multitask import LazyTask

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _minimal_cfg(tmp_path, **kwargs) -> ConfigMulti:
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        cache_home=tmp_path,
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


def _synth_df(n_weeks: int = 4, n_cols: int = 1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 24 * 7 * n_weeks
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    cols = {chr(ord("a") + i): rng.normal(100, 10, n) for i in range(n_cols)}
    return pd.DataFrame(cols, index=idx)


@pytest.fixture()
def single_target_lazy(tmp_path) -> LazyTask:
    """LazyTask with a single target set directly (bypasses prepare_data)."""
    task = LazyTask(
        _minimal_cfg(tmp_path),
        dataframe=_synth_df(n_cols=1),
    )
    task.config.targets = ["only_target"]
    return task


# ---------------------------------------------------------------------------
# _aggregate_and_show: single-target pass-through
# ---------------------------------------------------------------------------


def test_aggregate_and_show_passes_through_single_target(single_target_lazy):
    """Single-target config must return the per-target package unchanged."""
    pkg: Dict[str, Any] = {"future_pred": pd.Series([1.0, 2.0, 3.0]), "marker": "ok"}
    results = {"only_target": pkg}
    out = single_target_lazy._aggregate_and_show(
        results, task_name="single lazy", show=False
    )
    assert out is pkg


def test_aggregate_and_show_stores_in_agg_results(single_target_lazy):
    """The result must be registered in agg_results keyed by task_name."""
    pkg: Dict[str, Any] = {"future_pred": pd.Series([1.0])}
    single_target_lazy._aggregate_and_show(
        {"only_target": pkg}, task_name="my_task", show=False
    )
    assert single_target_lazy.agg_results["my_task"] is pkg


def test_aggregate_and_show_no_extra_figure_single_target(single_target_lazy):
    """show=True must not call _show_prediction_figure_agg for a single target.

    In spotforecast2-safe the hook is a no-op, but we verify it is not reached
    via the single-target fast-path by monkeypatching it to an error-raiser.
    """
    calls = {"agg_fig": 0}

    def raise_if_called(*_args, **_kwargs):
        calls["agg_fig"] += 1
        raise AssertionError("agg figure hook must not be called for single target")

    single_target_lazy._show_prediction_figure_agg = raise_if_called
    pkg: Dict[str, Any] = {"future_pred": pd.Series([1.0])}
    single_target_lazy._aggregate_and_show(
        {"only_target": pkg}, task_name="t", show=True
    )
    assert calls["agg_fig"] == 0


# ---------------------------------------------------------------------------
# _aggregate_and_show: multi-target path still invokes agg_predictor
# ---------------------------------------------------------------------------


def test_multi_target_aggregation_invokes_agg_predictor(tmp_path, monkeypatch):
    """Regression guard: multi-target configurations must call agg_predictor."""
    task = LazyTask(
        _minimal_cfg(tmp_path, agg_weights=[0.5, 0.5]),
        dataframe=_synth_df(n_cols=2),
    )
    task.config.targets = ["a", "b"]

    seen: Dict[str, Any] = {}

    def fake_agg_predictor(results, targets, weights):
        seen["targets"] = list(targets)
        seen["weights"] = list(weights)
        return {"future_pred": pd.Series([99.0])}

    monkeypatch.setattr(task, "agg_predictor", fake_agg_predictor)
    task._aggregate_and_show(
        {
            "a": {"future_pred": pd.Series([1.0])},
            "b": {"future_pred": pd.Series([2.0])},
        },
        task_name="multi",
        show=False,
    )
    assert seen["targets"] == ["a", "b"]
    assert seen["weights"] == [0.5, 0.5]


# ---------------------------------------------------------------------------
# show=True no-op hooks (spotforecast2-safe)
# ---------------------------------------------------------------------------


def test_show_true_does_not_raise_single_target(single_target_lazy):
    """show=True with a single target must not raise."""
    pkg: Dict[str, Any] = {"future_pred": pd.Series([1.0])}
    # Must not raise
    single_target_lazy._aggregate_and_show(
        {"only_target": pkg}, task_name="t", show=True
    )


def test_show_true_produces_same_result_as_false(tmp_path):
    """Identical predictions with show=True and show=False (hooks are no-ops)."""
    rng = np.random.default_rng(42)
    n = 24 * 28
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    df = pd.DataFrame({"a": rng.normal(100, 10, n)}, index=idx)

    def _run(show_flag: bool) -> pd.Series:
        task = LazyTask(
            _minimal_cfg(tmp_path),
            dataframe=df.copy(),
        )
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        result = task.run(show=show_flag)
        return result["future_pred"]

    r_false = _run(False)
    r_true = _run(True)
    pd.testing.assert_series_equal(r_false, r_true, check_exact=True)


# ---------------------------------------------------------------------------
# Integration: single-column DataFrame end-to-end
# ---------------------------------------------------------------------------


def test_single_target_end_to_end(tmp_path):
    """A single-column DataFrame must run the full pipeline without error."""
    rng = np.random.default_rng(7)
    n = 24 * 28
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    df = pd.DataFrame({"load": rng.normal(1000, 50, n)}, index=idx)

    task = LazyTask(
        _minimal_cfg(tmp_path),
        dataframe=df,
    )
    task.prepare_data().detect_outliers().impute().build_exogenous_features()
    result = task.run()
    assert len(result["future_pred"]) == 6
    assert result["future_pred"].notna().all()


def test_single_target_aggregation_uses_target_package_directly(tmp_path):
    """After a single-target run, agg_results["lazy"] must be the per-target pkg."""
    rng = np.random.default_rng(7)
    n = 24 * 28
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    df = pd.DataFrame({"load": rng.normal(1000, 50, n)}, index=idx)

    task = LazyTask(_minimal_cfg(tmp_path), dataframe=df)
    task.prepare_data().detect_outliers().impute().build_exogenous_features()
    result = task.run()

    # Single target: agg_results keyed by task_name string used in _run_strategy
    task_key = "task 1: Lazy Fitting"
    assert task_key in task.agg_results
    agg = task.agg_results[task_key]
    pd.testing.assert_series_equal(
        agg["future_pred"], result["future_pred"], check_exact=True
    )

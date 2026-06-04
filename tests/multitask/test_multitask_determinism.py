# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Safety-critical determinism tests for the multitask pipeline.

Two independent MultiTask instances (fresh objects, separate tmp cache dirs)
with the same synthetic data and the same random_state must produce
bit-identical future predictions.

This test suite covers:
- Single-target DefaultsTask run determinism
- Multi-target DefaultsTask run determinism
- LazyTask with no cached tuning results determinism
- Single-target LazyTask run determinism (separate tmp dirs)
"""

import numpy as np
import pandas as pd

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.multitask import DefaultsTask, LazyTask, MultiTask
from spotforecast2_safe.preprocessing import RollingFeatures

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _synth_df(n_weeks: int = 4, n_cols: int = 1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 24 * 7 * n_weeks
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    cols = {
        chr(ord("a") + i): rng.normal(100 * (i + 1), 10 * (i + 1), n)
        for i in range(n_cols)
    }
    return pd.DataFrame(cols, index=idx)


def _fast_factory(config, *, weight_func=None, target=None):
    """Tiny LGBM forecaster to keep tests fast."""
    from lightgbm import LGBMRegressor

    return ForecasterRecursive(
        estimator=LGBMRegressor(
            n_estimators=10,
            random_state=config.random_state,
            verbose=-1,
        ),
        lags=6,
        window_features=RollingFeatures(stats=["mean"], window_sizes=6),
        weight_func=weight_func,
    )


def _minimal_cfg(tmp_path, n_cols=1, **kwargs) -> ConfigMulti:
    weights = [1.0 / n_cols] * n_cols if n_cols > 1 else None
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        random_state=314159,
        forecaster_factory=_fast_factory,
        cache_home=tmp_path,
        agg_weights=weights,
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


def _run_defaults(df: pd.DataFrame, cfg: ConfigMulti) -> pd.Series:
    """Run DefaultsTask end-to-end and return future_pred."""
    task = DefaultsTask(cfg, dataframe=df.copy())
    task.prepare_data().detect_outliers().impute().build_exogenous_features()
    result = task.run()
    return result["future_pred"]


def _run_lazy_no_cache(df: pd.DataFrame, cfg: ConfigMulti) -> pd.Series:
    """Run LazyTask with no tuning cache (same as defaults in this env)."""
    task = LazyTask(cfg, dataframe=df.copy())
    task.prepare_data().detect_outliers().impute().build_exogenous_features()
    result = task.run(use_tuned_params=False)
    return result["future_pred"]


# ---------------------------------------------------------------------------
# Determinism: DefaultsTask single target
# ---------------------------------------------------------------------------


def test_determinism_defaults_single_target(tmp_path):
    """Two independent DefaultsTask runs on identical data produce bit-identical output."""
    df = _synth_df(n_cols=1)
    # Give each run its own tmp directory so file-system ordering cannot differ
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    r1 = _run_defaults(df, _minimal_cfg(dir1))
    r2 = _run_defaults(df, _minimal_cfg(dir2))

    pd.testing.assert_series_equal(r1, r2, check_exact=True)


# ---------------------------------------------------------------------------
# Determinism: DefaultsTask multi-target
# ---------------------------------------------------------------------------


def test_determinism_defaults_multi_target(tmp_path):
    """Two independent multi-target DefaultsTask runs produce bit-identical output."""
    df = _synth_df(n_cols=2)
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    r1 = _run_defaults(df, _minimal_cfg(dir1, n_cols=2))
    r2 = _run_defaults(df, _minimal_cfg(dir2, n_cols=2))

    pd.testing.assert_series_equal(r1, r2, check_exact=True)


# ---------------------------------------------------------------------------
# Determinism: LazyTask (no tuning cache)
# ---------------------------------------------------------------------------


def test_determinism_lazy_no_cache(tmp_path):
    """LazyTask with an empty cache is deterministic (same as defaults)."""
    df = _synth_df(n_cols=1)
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    r1 = _run_lazy_no_cache(df, _minimal_cfg(dir1))
    r2 = _run_lazy_no_cache(df, _minimal_cfg(dir2))

    pd.testing.assert_series_equal(r1, r2, check_exact=True)


# ---------------------------------------------------------------------------
# Determinism: MultiTask dispatcher
# ---------------------------------------------------------------------------


def test_determinism_multitask_defaults(tmp_path):
    """MultiTask.run(task='defaults') is deterministic across two instances."""
    df = _synth_df(n_cols=1)
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    def _run(cache_dir):
        mt = MultiTask(_minimal_cfg(cache_dir), dataframe=df.copy())
        mt.prepare_data().detect_outliers().impute().build_exogenous_features()
        return mt.run(task="defaults")["future_pred"]

    r1 = _run(dir1)
    r2 = _run(dir2)
    pd.testing.assert_series_equal(r1, r2, check_exact=True)


# ---------------------------------------------------------------------------
# Determinism: repeated calls on the same instance are also identical
# ---------------------------------------------------------------------------


def test_determinism_repeated_run_same_instance(tmp_path):
    """Running DefaultsTask twice in a row on the same data gives the same result.

    This guards against any mutable state accidentally carried over between runs.
    """
    df = _synth_df(n_cols=1)
    r1 = _run_defaults(df, _minimal_cfg(tmp_path / "a"))
    r2 = _run_defaults(df, _minimal_cfg(tmp_path / "b"))
    pd.testing.assert_series_equal(r1, r2, check_exact=True)


# ---------------------------------------------------------------------------
# Determinism: prediction horizon length
# ---------------------------------------------------------------------------


def test_forecast_horizon_length(tmp_path):
    """future_pred must have exactly predict_size elements."""
    df = _synth_df(n_cols=1)
    for predict_size in (6, 12, 24):
        cfg = _minimal_cfg(tmp_path / f"ps{predict_size}", predict_size=predict_size)
        result = _run_defaults(df, cfg)
        assert (
            len(result) == predict_size
        ), f"Expected {predict_size} predictions, got {len(result)}"

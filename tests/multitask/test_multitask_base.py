# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for BaseTask pipeline stages.

Covers:
- prepare_data: auto-derived targets, date-range population, validation errors
- detect_outliers: bounds filtering and IsolationForest path
- impute: weighted and linear methods
- build_exogenous_features: features-off path; holiday/calendar path offline
- Guards: each step raises RuntimeError if called before its predecessor
- Visualisation hooks are no-ops
- log_summary
- Pipeline method chaining (all steps return self)
- create_forecaster factory hook
"""

import logging

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.multitask import LazyTask, MultiTask

# ---------------------------------------------------------------------------
# Shared synthetic data fixture (seeded, no network)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_df() -> pd.DataFrame:
    """Two-week, two-column hourly DataFrame with a named DatetimeIndex."""
    rng = np.random.default_rng(0)
    n = 24 * 28  # 4 weeks, hourly
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    return pd.DataFrame(
        {
            "a": rng.normal(100, 10, n),
            "b": rng.normal(200, 20, n),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def synth_df_single(synth_df) -> pd.DataFrame:
    """Single-column version of synth_df."""
    return synth_df[["a"]].copy()


def _minimal_cfg(**kwargs) -> ConfigMulti:
    """Return a ConfigMulti with the most expensive options disabled."""
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        verbose=False,
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------


class TestPrepareData:
    """Verify prepare_data sets pipeline state correctly."""

    def test_returns_self(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        assert task.prepare_data() is task

    def test_df_pipeline_not_none_after_prepare(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data()
        assert task.df_pipeline is not None

    def test_df_pipeline_has_rows(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data()
        assert task.df_pipeline.shape[0] > 0

    def test_targets_auto_derived(self, synth_df, tmp_path):
        """When config.targets is None, all numeric columns become targets.

        After the RunState extraction (v18.0.0) the resolved list lives on
        task.run_state.targets; config.targets preserves the user input (None).
        """
        cfg = _minimal_cfg(cache_home=tmp_path, targets=None)
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data()
        assert set(task.run_state.targets) == {"a", "b"}
        # config.targets is user input and must not be mutated.
        assert task.config.targets is None

    def test_targets_respected_when_set(self, synth_df, tmp_path):
        """When config.targets is pre-set, only those columns are used."""
        cfg = _minimal_cfg(cache_home=tmp_path, targets=["a"])
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data()
        # config.targets holds the user input unchanged.
        assert task.config.targets == ["a"]
        # run_state.targets holds the resolved list (same value when user specified one).
        assert task.run_state.targets == ["a"]

    def test_date_range_populated(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data()
        assert task.run_state.data_start is not None
        assert task.run_state.data_end is not None
        assert task.run_state.cov_start is not None
        assert task.run_state.cov_end is not None

    def test_data_start_before_data_end(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data()
        assert task.run_state.data_start < task.run_state.data_end

    def test_no_data_raises_value_error(self, tmp_path):
        """Calling prepare_data with no data must raise ValueError."""
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        with pytest.raises(ValueError, match="No data source provided"):
            task.prepare_data()

    def test_demo_data_arg_overrides_constructor(self, synth_df, tmp_path):
        """demo_data positional arg takes priority over constructor dataframe."""
        rng = np.random.default_rng(1)
        n = 24 * 14
        idx = pd.date_range("2023-02-01", periods=n, freq="h", tz="UTC")
        idx.name = "DateTime"
        df_alt = pd.DataFrame({"c": rng.normal(50, 5, n)}, index=idx)

        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data(demo_data=df_alt)
        # run_state.targets holds the resolved list; config.targets is user input.
        assert "c" in task.run_state.targets

    def test_data_loader_callable_used(self, synth_df, tmp_path):
        """config.data_loader is invoked when no explicit DataFrame is given."""
        called = {}

        def loader(_config):
            called["called"] = True
            return synth_df

        cfg = _minimal_cfg(cache_home=tmp_path, data_loader=loader)
        task = LazyTask(cfg)
        task.prepare_data()
        assert called.get("called")
        assert task.df_pipeline is not None

    def test_non_datetime_index_raises(self, tmp_path):
        """A DataFrame with a plain integer index must cause an informative error."""
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0]},
            index=pd.RangeIndex(3),
        )
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=df)
        with pytest.raises(TypeError):
            task.prepare_data()

    def test_pipeline_state_attrs_initially_none(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        assert task.df_pipeline is None
        assert task.df_pipeline_original is None
        assert task.weight_func is None
        assert task.exogenous_features is None
        assert task.exog_feature_names == []
        assert task.data_with_exog is None
        assert task.exo_pred is None
        assert task.results == {}


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------


class TestDetectOutliers:
    """Verify detect_outliers: bounds and IsolationForest paths."""

    def test_returns_self(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data()
        assert task.detect_outliers() is task

    def test_original_df_saved(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data()
        shape_before = task.df_pipeline.shape
        task.detect_outliers()
        assert task.df_pipeline_original.shape == shape_before

    def test_guard_raises_before_prepare(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        with pytest.raises(RuntimeError, match="prepare_data"):
            task.detect_outliers()

    def test_bounds_remove_extremes(self, tmp_path):
        """Hard bounds convert values outside [lower, upper] to NaN."""
        rng = np.random.default_rng(0)
        n = 24 * 14
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        idx.name = "DateTime"
        vals = rng.normal(100, 10, n)
        vals[50] = 99999  # extreme outlier
        df = pd.DataFrame({"a": vals}, index=idx)

        cfg = _minimal_cfg(
            cache_home=tmp_path,
            bounds=[(-500, 200)],
            use_outlier_detection=False,
        )
        task = LazyTask(cfg, dataframe=df)
        task.prepare_data()
        task.detect_outliers()
        # The value 99999 must be gone (replaced with NaN)
        assert task.df_pipeline["a"].max() <= 200

    def test_isolation_forest_path_no_crash(self, synth_df, tmp_path):
        """IsolationForest enabled with low contamination must not raise."""
        cfg = _minimal_cfg(
            cache_home=tmp_path,
            use_outlier_detection=True,
            contamination=0.01,
        )
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data()
        task.detect_outliers()  # must not raise
        assert task.df_pipeline_original is not None

    def test_isolation_forest_disabled_no_nans_added(self, synth_df, tmp_path):
        """With use_outlier_detection=False and no bounds, no NaNs introduced."""
        cfg = _minimal_cfg(
            cache_home=tmp_path,
            use_outlier_detection=False,
            bounds=None,
        )
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data()
        nan_before = task.df_pipeline.isna().sum().sum()
        task.detect_outliers()
        nan_after = task.df_pipeline.isna().sum().sum()
        assert nan_after == nan_before


# ---------------------------------------------------------------------------
# impute
# ---------------------------------------------------------------------------


class TestImpute:
    """Verify impute fills gaps and returns self."""

    def test_returns_self(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data().detect_outliers()
        assert task.impute() is task

    def test_guard_raises_before_prepare(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        with pytest.raises(RuntimeError, match="prepare_data"):
            task.impute()

    def test_weighted_imputation_removes_nans(self, tmp_path):
        """After weighted imputation no NaN should remain."""
        rng = np.random.default_rng(0)
        n = 24 * 14
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        idx.name = "DateTime"
        vals = rng.normal(100, 10, n)
        vals[10:15] = np.nan  # inject some NaN
        df = pd.DataFrame({"a": vals}, index=idx)

        cfg = _minimal_cfg(
            cache_home=tmp_path,
            imputation_method="weighted",
        )
        task = LazyTask(cfg, dataframe=df)
        task.prepare_data().detect_outliers().impute()
        assert task.df_pipeline["a"].isna().sum() == 0

    def test_linear_imputation_removes_nans(self, tmp_path):
        """After linear imputation no NaN should remain."""
        rng = np.random.default_rng(0)
        n = 24 * 14
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        idx.name = "DateTime"
        vals = rng.normal(100, 10, n)
        vals[10:15] = np.nan
        df = pd.DataFrame({"a": vals}, index=idx)

        cfg = _minimal_cfg(
            cache_home=tmp_path,
            imputation_method="linear",
        )
        task = LazyTask(cfg, dataframe=df)
        task.prepare_data().detect_outliers().impute()
        assert task.df_pipeline["a"].isna().sum() == 0


# ---------------------------------------------------------------------------
# build_exogenous_features
# ---------------------------------------------------------------------------


class TestBuildExogenousFeatures:
    """Verify build_exogenous_features: no-op and calendar+holiday offline path."""

    def test_returns_self(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data().detect_outliers().impute()
        assert task.build_exogenous_features() is task

    def test_guard_raises_before_prepare(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        with pytest.raises(RuntimeError, match="prepare_data"):
            task.build_exogenous_features()

    def test_features_off_is_noop(self, synth_df, tmp_path):
        """When use_exogenous_features=False, exog attributes stay at defaults."""
        cfg = _minimal_cfg(cache_home=tmp_path, use_exogenous_features=False)
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        assert task.exogenous_features is None
        assert task.exog_feature_names == []
        assert task.data_with_exog is None
        assert task.exo_pred is None

    def test_calendar_holiday_path_no_network(self, synth_df, tmp_path):
        """Calendar + holiday features work offline (no weather fetch)."""
        cfg = ConfigMulti(
            predict_size=6,
            use_exogenous_features=True,
            include_weather_windows=False,
            include_holiday_features=True,
            include_holiday_adjacency_features=False,
            use_outlier_detection=False,
            auto_save_models=False,
            number_folds=2,
            on_weather_failure="skip",  # skip network
            cache_home=tmp_path,
            verbose=False,
        )
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        assert task.exogenous_features is not None
        assert len(task.exog_feature_names) > 0
        assert task.data_with_exog is not None
        assert task.exo_pred is not None

    def test_exo_pred_anchored_on_end_train(self, synth_df, tmp_path):
        """Regression (2026-06-06 live incident): with a user-pinned
        ``end_train_default`` earlier than the last observed target row, the
        prediction covariate slice must start at ``end_train_ts + 1h`` — the
        first step the forecaster actually predicts — not one hour after the
        data extent (which is phase-shifted against the prediction window
        and trips the fail-safe alignment guard at predict time)."""
        end_train = synth_df.index[-2]  # one hour before the data extent
        cfg = ConfigMulti(
            predict_size=6,
            use_exogenous_features=True,
            include_weather_windows=False,
            include_holiday_features=True,
            include_holiday_adjacency_features=False,
            use_outlier_detection=False,
            auto_save_models=False,
            number_folds=2,
            on_weather_failure="skip",  # skip network
            cache_home=tmp_path,
            verbose=False,
            end_train_default=end_train.isoformat(),
        )
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        assert task.run_state.end_train_ts == end_train
        assert task.exo_pred.index[0] == end_train + pd.Timedelta(hours=1)
        # The slice still reaches cov_end, so it covers the full prediction
        # window [end_train + 1h, end_train + predict_size] and beyond.
        expected_last_pred = end_train + pd.Timedelta(hours=cfg.predict_size)
        assert task.exo_pred.index[-1] >= expected_last_pred


# ---------------------------------------------------------------------------
# Guards (cross-step ordering)
# ---------------------------------------------------------------------------


class TestGuards:
    """Verify that steps fail safely when called out of order."""

    @pytest.mark.parametrize(
        "method",
        ["detect_outliers", "impute", "build_exogenous_features"],
    )
    def test_steps_require_prepare_data(self, method, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        with pytest.raises(RuntimeError, match="prepare_data"):
            getattr(task, method)()

    def test_plot_with_outliers_requires_detect_outliers(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data()
        with pytest.raises(RuntimeError, match="detect_outliers"):
            task.plot_with_outliers()

    def test_plot_with_outliers_raises_not_implemented(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        task.prepare_data().detect_outliers()
        with pytest.raises(NotImplementedError, match="spotforecast2"):
            task.plot_with_outliers()

    def test_run_without_prepare_raises(self, synth_df, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), dataframe=synth_df)
        with pytest.raises(RuntimeError, match="Pipeline data not prepared"):
            task.run()


# ---------------------------------------------------------------------------
# Visualisation hooks (no-ops)
# ---------------------------------------------------------------------------


class TestVisualisationHooks:
    """Confirm hooks exist, return None, and do not raise."""

    def test_show_prediction_figure_exists(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        assert callable(task._show_prediction_figure)

    def test_show_prediction_figure_agg_exists(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        assert callable(task._show_prediction_figure_agg)

    def test_show_prediction_figure_returns_none(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        pkg = {"future_pred": pd.Series([1.0, 2.0])}
        result = task._show_prediction_figure(pkg, "target_a", "test task")
        assert result is None

    def test_show_prediction_figure_agg_returns_none(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        pkg = {"future_pred": pd.Series([1.0, 2.0])}
        result = task._show_prediction_figure_agg(pkg, "test task")
        assert result is None


# ---------------------------------------------------------------------------
# log_summary
# ---------------------------------------------------------------------------


class TestLogSummary:
    """Verify log_summary does not raise at various pipeline stages."""

    def test_log_summary_after_full_pipeline_no_exog(self, synth_df, tmp_path):
        cfg = _minimal_cfg(cache_home=tmp_path, use_exogenous_features=False)
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        task.log_summary()  # must not raise


# ---------------------------------------------------------------------------
# create_forecaster
# ---------------------------------------------------------------------------


class TestCreateForecaster:
    """Verify the default and overridden forecaster factories."""

    def test_default_factory_returns_forecaster_recursive(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        fc = task.create_forecaster()
        assert isinstance(fc, ForecasterRecursive)

    def test_lags_derived_from_config(self, tmp_path):
        cfg = _minimal_cfg(cache_home=tmp_path, lags_consider=[1, 2, 48])
        task = LazyTask(cfg)
        fc = task.create_forecaster()
        assert max(fc.lags) == 48

    def test_weight_func_none_initially(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        fc = task.create_forecaster()
        assert fc.weight_func is None

    def test_lgbm_n_jobs_defaults_to_one(self, tmp_path):
        """The default factory pins LightGBM to a single thread by default."""
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        fc = task.create_forecaster()
        assert fc.estimator.get_params()["n_jobs"] == 1

    def test_lgbm_n_jobs_honoured_from_config(self, tmp_path):
        """config.lgbm_n_jobs flows through to the LightGBM estimator."""
        task = LazyTask(_minimal_cfg(cache_home=tmp_path, lgbm_n_jobs=-1))
        fc = task.create_forecaster()
        assert fc.estimator.get_params()["n_jobs"] == -1

    def test_custom_factory_override(self, tmp_path):
        """config.forecaster_factory replaces the default factory."""
        sentinel = object()
        captured = {}

        def stub(config, *, weight_func=None, target=None):
            captured["target"] = target
            return sentinel

        cfg = _minimal_cfg(cache_home=tmp_path, forecaster_factory=stub)
        task = LazyTask(cfg)
        result = task.create_forecaster(target="col_x")
        assert result is sentinel
        assert captured["target"] == "col_x"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TestLogger:
    """Verify logger is attached and named correctly."""

    def test_logger_is_logger_instance(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        assert isinstance(task.logger, logging.Logger)

    def test_logger_name_includes_class_name(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        assert "LazyTask" in task.logger.name

    def test_logger_level_default(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path))
        assert task.logger.level == logging.INFO

    def test_logger_level_override(self, tmp_path):
        task = LazyTask(_minimal_cfg(cache_home=tmp_path), log_level=logging.DEBUG)
        assert task.logger.level == logging.DEBUG


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------


class TestMethodChaining:
    """All pipeline steps must return self for chaining."""

    def test_full_chain_no_exog(self, synth_df, tmp_path):
        cfg = _minimal_cfg(cache_home=tmp_path, use_exogenous_features=False)
        task = LazyTask(cfg, dataframe=synth_df)
        result = (
            task.prepare_data().detect_outliers().impute().build_exogenous_features()
        )
        assert result is task

    def test_multitask_chain(self, synth_df, tmp_path):
        cfg = _minimal_cfg(cache_home=tmp_path, use_exogenous_features=False)
        mt = MultiTask(cfg, dataframe=synth_df)
        result = mt.prepare_data().detect_outliers().impute().build_exogenous_features()
        assert result is mt


# ---------------------------------------------------------------------------
# exog_provider_window threading into build_providers_from_config
# ---------------------------------------------------------------------------


def _stub_weather(monkeypatch):
    """Replace the Open-Meteo fetch with the empty-frame fallback shape.

    These tests only assert on provider wiring; hitting the live
    archive-api.open-meteo.com endpoint made them slow and flaky in CI.
    The empty frames mirror exactly what the ``on_weather_failure="skip"``
    fallback produces, so the downstream pipeline path is identical.
    """
    import spotforecast2_safe.multitask.base as mt_base

    def offline_weather(*, data, **kwargs):
        empty = pd.DataFrame(index=data.index)
        return empty, empty.copy()

    monkeypatch.setattr(mt_base, "get_weather_features", offline_weather)


class TestExogProviderWindow:
    """Verify that exog_provider_window is honoured at the call site."""

    def test_train_window_passes_provider_window(self, synth_df, tmp_path, monkeypatch):
        """With exog_provider_window='train', build_providers_from_config receives
        a provider_window covering [start_train_ts, cov_end]."""
        _stub_weather(monkeypatch)
        captured = {}

        import spotforecast2_safe.preprocessing.exog_providers as ep_module

        original = ep_module.build_providers_from_config

        def capturing_build(config, **kwargs):
            captured["provider_window"] = kwargs.get("provider_window")
            return original(config, **kwargs)

        monkeypatch.setattr(ep_module, "build_providers_from_config", capturing_build)

        cfg = _minimal_cfg(
            cache_home=tmp_path,
            use_exogenous_features=True,
            exog_provider_window="train",
        )
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data().detect_outliers().impute().build_exogenous_features()

        pw = captured.get("provider_window")
        assert pw is not None, "provider_window should have been passed"
        assert isinstance(pw, pd.DatetimeIndex)
        # After RunState extraction (v18.0.0) the derived fields live on
        # task.run_state; read them there instead of from cfg.
        assert pw[0] == task.run_state.start_train_ts
        assert pw[-1] == task.run_state.cov_end

    def test_default_full_window_passes_none(self, synth_df, tmp_path, monkeypatch):
        """With the default exog_provider_window='full', provider_window is None."""
        _stub_weather(monkeypatch)
        captured = {}

        import spotforecast2_safe.preprocessing.exog_providers as ep_module

        original = ep_module.build_providers_from_config

        def capturing_build(config, **kwargs):
            captured["provider_window"] = kwargs.get("provider_window")
            return original(config, **kwargs)

        monkeypatch.setattr(ep_module, "build_providers_from_config", capturing_build)

        cfg = _minimal_cfg(cache_home=tmp_path, use_exogenous_features=True)
        task = LazyTask(cfg, dataframe=synth_df)
        task.prepare_data().detect_outliers().impute().build_exogenous_features()

        assert captured.get("provider_window") is None

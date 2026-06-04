# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Base class for multi-target forecasting pipeline tasks.

Provides BaseTask, which contains all shared data-preparation,
outlier-detection, imputation, exogenous-feature engineering, forecaster
creation, prediction-packaging, and aggregation logic.  Task-specific
subclasses (LazyTask, DefaultsTask) inherit from BaseTask and implement the
run method.  PredictTask also inherits from BaseTask but skips training
entirely, loading saved models instead.

Plotting is not available in ``spotforecast2-safe``.  The visualisation
hook methods ``_show_prediction_figure`` and ``_show_prediction_figure_agg``
are no-ops here.  Override them in a subclass or use the ``spotforecast2``
sibling package for interactive figures.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol

import pandas as pd
from astral import LocationInfo
from joblib import dump as _joblib_dump
from joblib import load as _joblib_load
from sklearn.model_selection import TimeSeriesSplit as _SklearnTimeSeriesSplit

from spotforecast2_safe.calendar import (
    get_calendar_features,
    get_day_night_features,
    get_holiday_adjacency_features,
    get_holiday_features,
)
from spotforecast2_safe.configurator.config_multi import (  # noqa: F401  (re-exported for subclasses)
    ConfigMulti,
)
from spotforecast2_safe.data.fetch_data import get_cache_home
from spotforecast2_safe.manager.features import (
    apply_cyclical_encoding,
    create_interaction_features,
    get_target_data,
    merge_data_and_covariates,
    select_exogenous_features,
    select_top_poly_features,
)
from spotforecast2_safe.manager.predictor import build_prediction_package
from spotforecast2_safe.multitask.factories import default_lgbm_forecaster_factory
from spotforecast2_safe.preprocessing.curate_data import (
    agg_and_resample_data,
    basic_ts_checks,
    get_start_end,
    reset_index,
)
from spotforecast2_safe.preprocessing.imputation import apply_imputation
from spotforecast2_safe.preprocessing.outlier import (
    get_outliers,
    manual_outlier_removal,
)
from spotforecast2_safe.processing.agg_predict import agg_predict
from spotforecast2_safe.splitter.split_ts_cv import TimeSeriesFold
from spotforecast2_safe.weather import WeatherFetchError, get_weather_features

logger = logging.getLogger(__name__)


class PipelineConfig(Protocol):
    """Structural protocol describing the config surface ``BaseTask`` reads.

    ``ConfigMulti`` and ``ConfigEntsoe`` both satisfy this protocol; any
    future config class only needs to expose the same field names with
    compatible types.  The protocol is the contract that lets a caller pass
    ``MultiTask(config=ConfigEntsoe(...))`` (or any other ``PipelineConfig``)
    without subclassing or rebuilding kwargs.
    """

    # Targets and aggregation
    targets: Optional[List[str]]
    agg_weights: Optional[List[float]]
    bounds: Optional[List[tuple]]
    # Forecast horizon and training window
    predict_size: int
    train_size: pd.Timedelta
    delta_val: pd.Timedelta
    end_train_default: str
    end_train_ts: Optional[pd.Timestamp]
    start_train_ts: Optional[pd.Timestamp]
    # Outlier detection / imputation
    use_outlier_detection: bool
    contamination: float
    imputation_method: str
    window_size: int
    imputation_window_size: Optional[int]
    # Exogenous features
    use_exogenous_features: bool
    include_weather_windows: bool
    include_holiday_features: bool
    include_holiday_adjacency_features: bool
    poly_features_degree: int
    max_poly_features: int
    latitude: float
    longitude: float
    timezone: str
    state: str
    country_code: str
    lags_consider: List[int]
    # Data ranges (derived after data loading)
    data_start: Optional[pd.Timestamp]
    data_end: Optional[pd.Timestamp]
    cov_start: Optional[pd.Timestamp]
    cov_end: Optional[pd.Timestamp]
    start_download: Optional[str]
    end_download: Optional[str]
    # Tuning trial budgets
    n_trials_optuna: int
    n_trials_spotoptim: int
    n_initial_spotoptim: int
    # Cross-validation
    number_folds: int
    # Optional CV block width; ``None`` => fall back to ``predict_size``.
    cv_block_size: Optional[int]
    # Persistence policy and active-dataset identifier
    auto_save_models: bool
    data_frame_name: str
    # Weather-fetch failure policy
    on_weather_failure: Literal["raise", "skip"]
    # Misc
    random_state: int
    verbose: bool
    cache_home: Optional[Any]
    # Optional callables
    forecaster_factory: Optional[Any]
    data_loader: Optional[Any]
    test_data_loader: Optional[Any]

    def set_params(
        self,
        params: Optional[Dict[str, object]] = None,
        **kwargs: object,
    ) -> "PipelineConfig":
        """Update one or more config fields in place and return ``self``."""


def agg_predictor(
    results: Dict[str, Dict[str, Any]],
    targets: List[str],
    weights: List[float],
) -> Dict[str, Any]:
    """Aggregate per-target prediction packages into a weighted forecast.

    Combines future predictions, training predictions, and training actuals
    from per-target prediction packages into an aggregated package compatible
    with downstream consumers.  This is a module-level convenience function;
    the same logic is available as ``BaseTask.agg_predictor``.

    Args:
        results: Mapping of target name to prediction package (as returned
            by ``build_prediction_package``).
        targets: Ordered list of target names to aggregate.
        weights: Per-target aggregation weights aligned with ``targets``.

    Returns:
        Aggregated prediction package with keys ``train_actual``,
        ``train_pred``, ``future_pred``, ``future_actual``,
        ``metrics_train``, ``metrics_future``, ``metrics_future_one_day``,
        ``validation_passed``, and (when present in all sources)
        ``test_actual``.
    """
    future_preds_df = pd.DataFrame({t: results[t]["future_pred"] for t in targets})
    train_preds_df = pd.DataFrame({t: results[t]["train_pred"] for t in targets})
    train_actuals_df = pd.DataFrame({t: results[t]["train_actual"] for t in targets})

    agg_future_pred = agg_predict(future_preds_df, weights=weights)
    agg_train_pred = agg_predict(train_preds_df, weights=weights)
    agg_train_actual = agg_predict(train_actuals_df, weights=weights)

    test_series = {
        t: results[t]["test_actual"]
        for t in targets
        if "test_actual" in results[t] and len(results[t]["test_actual"]) > 0
    }
    agg_test_actual = None
    if test_series:
        test_actuals_df = pd.DataFrame(test_series).reindex(
            sorted(set().union(*[s.index for s in test_series.values()]))
        )
        test_weights = [weights[i] for i, t in enumerate(targets) if t in test_series]
        agg_test_actual = agg_predict(test_actuals_df, weights=test_weights)

    agg_pkg: Dict[str, Any] = {
        "train_actual": agg_train_actual,
        "train_pred": agg_train_pred,
        "future_actual": pd.Series(dtype="float64"),
        "future_pred": agg_future_pred,
        "metrics_train": {},
        "metrics_future": {},
        "metrics_future_one_day": {},
        "validation_passed": True,
    }
    if agg_test_actual is not None:
        agg_pkg["test_actual"] = agg_test_actual
    return agg_pkg


class BaseTask:
    """Shared base for all multi-target forecasting pipeline tasks.

    ``BaseTask`` encapsulates the data-preparation pipeline (steps 1-7)
    and all helper methods shared across the task modes (lazy, defaults,
    predict, clean).  Subclasses implement the run method with task-specific
    training or prediction logic.

    The constructor takes a single ``config`` object satisfying the
    ``PipelineConfig`` protocol — typically a ``ConfigMulti``.  All pipeline
    parameters (forecast horizon, training window, outlier policy, weather/
    holiday hooks, cross-validation fold count, persistence policy, ...) live
    on that object.  Only genuinely call-time state (the dataframes, the cache
    directory override, the logging level) is passed as separate kwargs.
    Extra ``**overrides`` are forwarded to ``config.set_params`` and mutate
    the passed-in config in place.

    Plotting is not available in ``spotforecast2-safe``.  The
    ``_show_prediction_figure`` and ``_show_prediction_figure_agg`` hook
    methods are no-ops; override them in a subclass or use the
    ``spotforecast2`` sibling package for interactive visualisation.

    Args:
        config: A ``PipelineConfig``-conforming object owning every pipeline
            parameter.  ``ConfigMulti`` satisfies the protocol.
        dataframe: Pre-loaded input DataFrame with training data.  Must
            contain a datetime column matching ``config.index_name`` plus
            at least one numeric target column.
        data_test: Pre-loaded test DataFrame (ground truth for the
            forecast horizon).  Optional.
        cache_home: Cache directory override.  When not ``None``, replaces
            ``config.cache_home`` for this task instance.
        log_level: Logging level for the pipeline logger.
        **overrides: Forwarded to ``config.set_params(**overrides)`` — a
            convenience for one-line tweaks without building a fresh config.
            Mutates the caller's config object.

    Attributes:
        config (PipelineConfig): Centralised pipeline configuration.
        df_pipeline (pd.DataFrame): Pipeline DataFrame after preparation.
        df_test (pd.DataFrame): Test DataFrame (ground truth).
        weight_func: Sample-weight function from imputation.
        exogenous_features (pd.DataFrame): Combined exogenous feature matrix.
        exog_feature_names (List[str]): Selected exogenous feature names.
        data_with_exog (pd.DataFrame): Merged target + exogenous data.
        exo_pred (pd.DataFrame): Exogenous covariates for the forecast horizon.
        results (Dict[str, Dict]): Per-task mapping of target name to
            prediction package.
        agg_results (Dict): Mapping of task name to aggregated prediction
            package.
    """

    _task_name: str = "lazy"

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        dataframe: Optional[pd.DataFrame] = None,
        data_test: Optional[pd.DataFrame] = None,
        cache_home: Optional[Path] = None,
        log_level: int = logging.INFO,
        **overrides: Any,
    ) -> None:
        # Task identifier (overridden by subclasses via _task_name)
        self.TASK = self._task_name

        if config is None:
            config = ConfigMulti()
        # Apply caller-supplied overrides in place on the config object.
        if overrides:
            config.set_params(**overrides)
        # ``cache_home`` is the one call-time override that wins over the
        # config value when explicitly supplied.
        if cache_home is not None:
            config.cache_home = cache_home
        # Propagate the task identifier so config-aware helpers know the mode.
        config.task = self.TASK
        self.config = config

        # Call-time data and per-instance state
        self._dataframe = dataframe
        self.data_test = data_test
        # Cached resolved cache directory; helpers read ``self.config.cache_home``.
        self.cache_home = config.cache_home

        # Logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Pipeline state (populated by methods)
        self.df_pipeline: Optional[pd.DataFrame] = None
        self.df_pipeline_original: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.weight_func: Optional[Any] = None
        self.exogenous_features: Optional[pd.DataFrame] = None
        self.weather_aligned: Optional[pd.DataFrame] = None
        self.exog_feature_names: List[str] = []
        self.data_with_exog: Optional[pd.DataFrame] = None
        self.exo_pred: Optional[pd.DataFrame] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.agg_results: Dict[str, Any] = {}

        self._attach_file_handler()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _attach_file_handler(self) -> None:
        """Attach a FileHandler to self.logger writing to get_cache_home()/logging/."""
        log_dir = get_cache_home(self.config.cache_home) / "logging"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.config.data_frame_name}.log"
        # Loggers are singletons — avoid adding duplicate FileHandlers
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file:
                return
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Step 1 — Data Preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        demo_data: Optional[pd.DataFrame] = None,
        df_test: Optional[pd.DataFrame] = None,
    ) -> "BaseTask":
        """Load, resample, validate, and configure the pipeline data.

        Uses the following precedence for the training data:

        1. ``demo_data`` argument (if provided).
        2. ``self._dataframe`` set via the constructor.

        Similarly for test data:

        1. ``df_test`` argument (if provided).
        2. ``self.data_test`` set via the constructor.
        3. ``self.config.test_data_loader(self.config)`` if set.

        Args:
            demo_data: Pre-loaded input DataFrame.  When ``None``, the
                constructor ``dataframe`` is used.
            df_test: Pre-loaded test DataFrame.  When ``None``, the
                constructor ``data_test`` is used, then
                ``config.test_data_loader``.

        Returns:
            ``self`` (for method chaining).

        Raises:
            ValueError: If no data source is available (no ``demo_data``,
                no constructor ``dataframe``).

        Examples:
            ```{python}
            import tempfile
            import pandas as pd
            import numpy as np
            from spotforecast2_safe.multitask import MultiTask
            from spotforecast2_safe.configurator.config_multi import ConfigMulti

            rng = np.random.default_rng(0)
            idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
            df = pd.DataFrame({"a": rng.normal(100, 10, len(idx))}, index=idx)
            df.index.name = "DateTime"

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(
                    predict_size=6,
                    use_exogenous_features=False,
                    use_outlier_detection=False,
                    cache_home=tmp,
                )
                mt = MultiTask(cfg, dataframe=df)
                mt.prepare_data()
                print(f"Pipeline shape: {mt.df_pipeline.shape}")
                print(f"Targets: {mt.config.targets}")
            ```
        """
        if demo_data is None:
            demo_data = self._dataframe
        if demo_data is None and self.config.data_loader is not None:
            demo_data = self.config.data_loader(self.config)
        if demo_data is None:
            raise ValueError(
                "No data source provided. Pass a DataFrame via the "
                "'dataframe' constructor argument, the 'demo_data' parameter "
                "of prepare_data(), or set 'config.data_loader' to a callable "
                "returning a DataFrame."
            )

        if df_test is None:
            df_test = self.data_test
        if (
            df_test is None
            and getattr(self.config, "test_data_loader", None) is not None
        ):
            df_test = self.config.test_data_loader(self.config)

        demo_data = reset_index(demo_data, index_name=self.config.index_name)
        if df_test is not None:
            df_test = reset_index(df_test, index_name=self.config.index_name)
        self.df_test = df_test

        first_ts = pd.Timestamp(demo_data[self.config.index_name].iloc[0])
        last_ts = pd.Timestamp(demo_data[self.config.index_name].iloc[-1])
        self.config.start_download = first_ts.strftime("%Y%m%d%H%M")
        self.config.end_download = last_ts.strftime("%Y%m%d%H%M")
        # Honour an explicit user-set ``end_train_default`` when it is earlier
        # than the last data row; only overwrite when the requested cutoff
        # exceeds the data extent (e.g. demo flows that have no anchor).
        user_end_train = pd.to_datetime(
            getattr(self.config, "end_train_default", None), utc=True, errors="coerce"
        )
        last_ts_utc = (
            last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        )
        if user_end_train is pd.NaT or user_end_train > last_ts_utc:
            self.config.end_train_default = last_ts.isoformat()

        all_targets = [c for c in demo_data.columns if c != self.config.index_name]
        if self.config.targets is None:
            self.config.targets = all_targets

        df_pipeline = demo_data.set_index(self.config.index_name)
        df_pipeline = agg_and_resample_data(df_pipeline, verbose=self.config.verbose)
        basic_ts_checks(df_pipeline, verbose=self.config.verbose)

        self.config.targets = [
            c for c in self.config.targets if c in df_pipeline.columns
        ]

        (
            self.config.data_start,
            self.config.data_end,
            self.config.cov_start,
            self.config.cov_end,
        ) = get_start_end(
            data=df_pipeline,
            forecast_horizon=self.config.predict_size,
            verbose=self.config.verbose,
        )

        self.df_pipeline = df_pipeline
        self.logger.info("Pipeline data shape after preparation: %s", df_pipeline.shape)
        return self

    # ------------------------------------------------------------------
    # Step 2 — Outlier Detection and Removal
    # ------------------------------------------------------------------

    def detect_outliers(self) -> "BaseTask":
        """Apply hard-bound filtering and IsolationForest outlier detection.

        Hard bounds from ``config.bounds`` are applied to the pipeline data
        (out-of-bound values are removed and later filled by ``impute()``).
        IsolationForest detection (``config.use_outlier_detection``) is
        advisory: detected outliers are logged per column but not removed.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If method ``prepare_data`` has not been called.
        """
        if self.df_pipeline is None:
            raise RuntimeError("Call prepare_data() before detect_outliers().")

        self.df_pipeline_original = self.df_pipeline.copy()

        if self.config.bounds is not None:
            for i, (lower, upper) in enumerate(self.config.bounds):
                col = self.df_pipeline.columns[i]
                self.df_pipeline, _ = manual_outlier_removal(
                    data=self.df_pipeline,
                    column=col,
                    lower_threshold=lower,
                    upper_threshold=upper,
                    verbose=self.config.verbose,
                )

        if self.config.use_outlier_detection:
            outliers = get_outliers(
                data=self.df_pipeline,
                contamination=self.config.contamination,
            )
            for col, outlier_vals in outliers.items():
                self.logger.info(
                    "%s: %d outliers automatically detected", col, len(outlier_vals)
                )

        self.logger.info("Outlier detection complete.")
        return self

    # ------------------------------------------------------------------
    # Step 2b — Plot outliers (not available in spotforecast2-safe)
    # ------------------------------------------------------------------

    def plot_with_outliers(self) -> None:
        """Visualise original vs. cleaned data with outlier markers.

        Raises:
            RuntimeError: If method ``detect_outliers`` has not been called.
            NotImplementedError: Always — plotting is not available in
                ``spotforecast2-safe``.  Use the ``spotforecast2`` package for
                visualisation.
        """
        if self.df_pipeline_original is None:
            raise RuntimeError("Call detect_outliers() before plot_with_outliers().")

        raise NotImplementedError(
            "Plotting is not available in spotforecast2-safe (no plotly/matplotlib). "
            "Use the spotforecast2 package for visualisation."
        )

    # ------------------------------------------------------------------
    # Step 3 — Imputation
    # ------------------------------------------------------------------

    def impute(self) -> "BaseTask":
        """Fill missing values using the configured imputation strategy.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If method ``prepare_data`` has not been called.
        """
        if self.df_pipeline is None:
            raise RuntimeError("Call prepare_data() before impute().")

        self.df_pipeline, self.weight_func = apply_imputation(
            df_pipeline=self.df_pipeline,
            config=self.config,
            logger=self.logger,
        )
        self.logger.info(
            "Imputation complete (method=%s).", self.config.imputation_method
        )
        return self

    # ------------------------------------------------------------------
    # Steps 4-7 — Exogenous Features
    # ------------------------------------------------------------------

    def build_exogenous_features(self) -> "BaseTask":
        """Build, combine, encode, and merge exogenous feature covariates.

        This is step 4-7 of the pipeline (run after ``prepare_data``,
        ``detect_outliers``, and ``impute``).  It assembles the full
        exogenous-covariate matrix that the forecaster consumes, then merges
        it onto the target data.  The orchestration proceeds in order:

        * 4a — Weather, via ``get_weather_features`` (Open-Meteo).  The
          response is parquet-cached only when ``config.cache_home`` is set.
          Fetch failures are handled per ``config.on_weather_failure``:
          ``"raise"`` re-raises ``WeatherFetchError``; ``"skip"`` logs a
          warning and continues with an empty weather frame (fail-safe).
        * 4b — Calendar features, via ``get_calendar_features``.
        * 4c — Day/night (solar) features, via ``get_day_night_features``
          (computed with ``astral`` from ``config.latitude`` /
          ``config.longitude``).
        * 4d — Holiday features, via ``get_holiday_features`` for
          ``config.country_code`` / ``config.state``.
        * 5 — The four frames are concatenated along the columns and any
          residual gaps are back- then forward-filled.  Provider-based
          exogenous columns are then appended via
          ``build_providers_from_config`` (requires ``spotforecast2-safe``
          >= 15.7.0).  The active providers are governed by the config flags
          ``include_covid_infection_rate``,
          ``include_entsoe_forecast_load``,
          ``include_entsoe_renewable_forecast``,
          ``include_entsoe_net_load``, and
          ``include_entsoe_day_ahead_price``.  Cyclical (sine/cosine)
          encoding is then applied via ``apply_cyclical_encoding``, and
          degree-``config.poly_features_degree`` interaction terms are added
          via ``create_interaction_features``.  When the degree is at least
          2, the polynomial columns are ranked by mutual information with the
          primary target and capped to ``config.max_poly_features`` via
          ``select_top_poly_features``.
        * 6 — The training feature set is chosen via
          ``select_exogenous_features``, with provider columns appended
          (order-preserving, de-duplicated).
        * 7 — Targets and covariates are merged via
          ``merge_data_and_covariates`` into ``self.data_with_exog`` and the
          forecast-horizon covariates ``self.exo_pred``.

        When ``config.use_exogenous_features`` is ``False`` the method is a
        no-op and returns ``self`` immediately, leaving the pipeline
        target-only.

        Attributes:
            weather_aligned (pd.DataFrame): Weather frame aligned to the
                pipeline index, reused by the interaction and selection steps.
            exogenous_features (pd.DataFrame): Full combined, encoded, and
                capped exogenous feature matrix.
            exog_feature_names (List[str]): Names of the exogenous features
                selected for training (including provider columns).
            data_with_exog (pd.DataFrame): Target data merged with the
                selected exogenous covariates.
            exo_pred (pd.DataFrame): Exogenous covariates spanning the
                forecast horizon, supplied to the forecaster at predict time.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If ``prepare_data`` has not been called.
            WeatherFetchError: If the Open-Meteo fetch fails and
                ``config.on_weather_failure == "raise"``.

        Examples:
            With exogenous features disabled the method is a no-op, so the
            example below runs without any network access and leaves the
            pipeline target-only.

            ```{python}
            import tempfile
            import pandas as pd
            import numpy as np
            from spotforecast2_safe.multitask import MultiTask
            from spotforecast2_safe.configurator.config_multi import ConfigMulti

            rng = np.random.default_rng(0)
            idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
            df = pd.DataFrame({"a": rng.normal(100, 10, len(idx))}, index=idx)
            df.index.name = "DateTime"

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(
                    predict_size=6,
                    use_exogenous_features=False,
                    use_outlier_detection=False,
                    cache_home=tmp,
                )
                mt = MultiTask(cfg, dataframe=df)
                mt.prepare_data().detect_outliers().impute().build_exogenous_features()
                print(f"Exogenous features used: {mt.config.use_exogenous_features}")
                print(f"Selected exog feature names: {mt.exog_feature_names}")
            ```
        """
        if self.df_pipeline is None:
            raise RuntimeError("Call prepare_data() before build_exogenous_features().")

        if not self.config.use_exogenous_features:
            self.logger.info("Exogenous features disabled — target-only pipeline.")
            return self

        self.logger.info("Building exogenous features...")

        # 4a. Weather (with opt-in fail-safe handling for Open-Meteo failures)
        try:
            weather_features, self.weather_aligned = get_weather_features(
                data=self.df_pipeline,
                start=self.config.data_start,
                cov_end=self.config.cov_end,
                forecast_horizon=self.config.predict_size,
                latitude=self.config.latitude,
                longitude=self.config.longitude,
                timezone=self.config.timezone,
                freq="h",
                cache_home=self.config.cache_home,
                verbose=self.config.verbose,
            )
        except WeatherFetchError as exc:
            if self.config.on_weather_failure == "raise":
                raise
            self.logger.warning(
                "Open-Meteo fetch failed (%s); continuing without weather features "
                "because config.on_weather_failure='skip'.",
                exc,
            )
            weather_features = pd.DataFrame(index=self.df_pipeline.index)
            self.weather_aligned = pd.DataFrame(index=self.df_pipeline.index)
        self.logger.info("  Weather features: %s", weather_features.shape)

        # 4b. Calendar
        calendar_features = get_calendar_features(
            start=self.config.data_start,
            cov_end=self.config.cov_end,
            freq="h",
            timezone=self.config.timezone,
        )
        self.logger.info("  Calendar features: %s", calendar_features.shape)

        # 4c. Day/night
        location = LocationInfo(
            latitude=self.config.latitude,
            longitude=self.config.longitude,
            timezone=self.config.timezone,
        )
        sun_light_features = get_day_night_features(
            start=self.config.data_start,
            cov_end=self.config.cov_end,
            location=location,
            freq="h",
            timezone=self.config.timezone,
        )
        self.logger.info("  Day/night features: %s", sun_light_features.shape)

        # 4d. Holidays
        holiday_features = get_holiday_features(
            data=self.df_pipeline,
            start=self.config.data_start,
            cov_end=self.config.cov_end,
            forecast_horizon=self.config.predict_size,
            tz=self.config.timezone,
            freq="h",
            country_code=self.config.country_code,
            state=self.config.state,
        )
        self.logger.info("  Holiday features: %s", holiday_features.shape)

        # 4e. Holiday-adjacency features (Brückentag, before/after holiday)
        concat_frames = [
            calendar_features,
            sun_light_features,
            weather_features,
            holiday_features,
        ]
        if self.config.include_holiday_adjacency_features:
            holiday_adjacency_features = get_holiday_adjacency_features(
                data=self.df_pipeline,
                start=self.config.data_start,
                cov_end=self.config.cov_end,
                forecast_horizon=self.config.predict_size,
                tz=self.config.timezone,
                freq="h",
                country_code=self.config.country_code,
                state=self.config.state,
            )
            self.logger.info(
                "  Holiday adjacency features: %s", holiday_adjacency_features.shape
            )
            concat_frames.append(holiday_adjacency_features)

        # Step 5 — Combine
        self.exogenous_features = pd.concat(
            concat_frames,
            axis=1,
        )

        missing_exog = self.exogenous_features.isnull().sum().sum()
        if missing_exog != 0:
            self.logger.warning(
                "Exogenous features contain %d missing values — backfilling.",
                missing_exog,
            )
            self.exogenous_features = self.exogenous_features.bfill().ffill()

        # Provider-based exogenous features (requires sf2-safe >= 15.7.0):
        # append the columns produced by the ConfigEntsoe/ConfigMulti provider
        # flags — covid_infection_rate and the ENTSO-E day-ahead
        # forecast/renewable/net-load/price. The registry is imported lazily so
        # older sf2-safe installs (without it) stay importable; a provider that
        # cannot cover the full exogenous index is re-raised or skipped per
        # config.on_exog_provider_failure (fail-safe).
        provider_feature_names: list = []
        try:
            from spotforecast2_safe.preprocessing.exog_providers import (
                ExogProviderError,
                build_providers_from_config,
            )
        except ImportError:
            build_providers_from_config = None  # type: ignore[assignment]
        if build_providers_from_config is not None:
            on_fail = getattr(self.config, "on_exog_provider_failure", "raise")
            provider_frames = []
            for provider in build_providers_from_config(self.config):
                try:
                    cols = provider.build(self.exogenous_features.index)
                except ExogProviderError as exc:
                    if on_fail == "raise":
                        raise
                    self.logger.warning(
                        "Skipping exog provider %r: %s", provider.name, exc
                    )
                    continue
                provider_frames.append(cols)
                provider_feature_names.extend(cols.columns.tolist())
            if provider_frames:
                self.exogenous_features = pd.concat(
                    [self.exogenous_features, *provider_frames], axis=1
                )
                self.logger.info(
                    "  Provider features: +%d columns (%s)",
                    len(provider_feature_names),
                    ", ".join(provider_feature_names),
                )

        self.exogenous_features = apply_cyclical_encoding(
            data=self.exogenous_features,
            drop_original=False,
        )
        self.exogenous_features = create_interaction_features(
            exogenous_features=self.exogenous_features,
            weather_aligned=self.weather_aligned,
            degree=self.config.poly_features_degree,
        )

        # Cap polynomial interactions to the top max_poly_features ranked by
        # mutual information with the primary target, then drop the rest.
        if self.config.poly_features_degree >= 2:
            poly_cols = [
                c for c in self.exogenous_features.columns if c.startswith("poly_")
            ]
            max_poly = self.config.max_poly_features
            if poly_cols and max_poly and 0 < max_poly < len(poly_cols):
                keep = select_top_poly_features(
                    self.exogenous_features[poly_cols],
                    self.df_pipeline[self.config.targets[0]],
                    max_poly_features=max_poly,
                    random_state=self.config.random_state,
                    n_jobs=getattr(self.config, "poly_mi_n_jobs", -1),
                    mi_sample_size=getattr(self.config, "poly_mi_sample_size", 4000),
                )
                drop = [c for c in poly_cols if c not in keep]
                self.exogenous_features = self.exogenous_features.drop(columns=drop)
                self.logger.info(
                    "Capped polynomial features: kept %d of %d (dropped %d).",
                    len(keep),
                    len(poly_cols),
                    len(drop),
                )

        self.logger.info(
            "Combined exogenous features shape: %s", self.exogenous_features.shape
        )

        # Step 6 — Select
        self.exog_feature_names = select_exogenous_features(
            exogenous_features=self.exogenous_features,
            weather_aligned=self.weather_aligned,
            include_weather_windows=self.config.include_weather_windows,
            include_holiday_features=self.config.include_holiday_features,
            include_holiday_adjacency_features=self.config.include_holiday_adjacency_features,
            poly_features_degree=self.config.poly_features_degree,
        )
        # ``select_exogenous_features`` matches calendar/weather/holiday/poly
        # columns by name pattern; provider columns carry their own names, so
        # append them to the selection here (order-preserving, de-duplicated).
        if provider_feature_names:
            self.exog_feature_names = list(
                dict.fromkeys(self.exog_feature_names + provider_feature_names)
            )
        self.logger.info(
            "Selected %d exogenous features for training.", len(self.exog_feature_names)
        )

        # Step 7 — Merge
        self.data_with_exog, _, self.exo_pred = merge_data_and_covariates(
            data=self.df_pipeline,
            exogenous_features=self.exogenous_features,
            target_columns=self.config.targets,
            exog_features=self.exog_feature_names,
            start=self.config.data_start,
            end=self.config.data_end,
            cov_end=self.config.cov_end,
            forecast_horizon=self.config.predict_size,
            cast_dtype="float32",
        )
        self.logger.info("Merged data shape: %s", self.data_with_exog.shape)
        self.logger.info(
            "Exogenous prediction covariates shape: %s", self.exo_pred.shape
        )

        return self

    # ------------------------------------------------------------------
    # Training-window setup
    # ------------------------------------------------------------------

    def _setup_training_window(self) -> None:
        """Derive ``config.end_train_ts`` and ``config.start_train_ts``."""
        self.config.end_train_ts = pd.to_datetime(
            self.config.end_train_default, utc=True
        )
        self.config.start_train_ts = self.config.end_train_ts - self.config.train_size
        self.config.start_train_ts = max(
            self.config.start_train_ts, self.df_pipeline.index.min()
        )
        self.logger.info(
            "Training window: %s to %s",
            self.config.start_train_ts,
            self.config.end_train_ts,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def cv_ts(self, y_train: pd.Series) -> TimeSeriesFold:
        """Build a ``TimeSeriesFold`` for cross-validation.

        Constructs the cross-validation splitter used by all tuning tasks.
        Internally uses ``sklearn.model_selection.TimeSeriesSplit`` to
        compute split boundaries that respect temporal ordering and avoid
        data leakage between folds.

        The validation boundary is determined by ``config.end_train_ts`` minus
        ``config.delta_val``.  When ``config.train_size`` is set, the sklearn
        splitter uses a sliding fixed-size training window
        (``max_train_size``); otherwise an expanding window is used.

        Args:
            y_train: Training time series for the current target.  Used both
                to determine the validation boundary and as the sequence passed
                to ``TimeSeriesSplit.split`` to derive ``initial_train_size``.

        Returns:
            A configured ``TimeSeriesFold`` instance ready to be passed to
            a model-selection function.
        """
        end_cv = self.config.end_train_ts - self.config.delta_val
        n_train_cv = len(y_train.loc[:end_cv])

        # CV block width: decoupled from the live forecast horizon.  Defaults to
        # ``predict_size`` so existing configs are unaffected; an optional
        # ``cv_block_size`` field (None => fall back) widens or narrows the
        # validation fold without changing what gets forecast in production.
        cv_block = (
            getattr(self.config, "cv_block_size", None) or self.config.predict_size
        )

        # Fixed sliding window when a training-size limit is configured;
        # expanding window otherwise (sklearn default).
        max_train_size: Optional[int] = (
            n_train_cv if self.config.train_size is not None else None
        )

        skl_cv = _SklearnTimeSeriesSplit(
            n_splits=self.config.number_folds,
            max_train_size=max_train_size,
            test_size=cv_block,
            gap=0,
        )

        # The first fold's training-set size determines the initial training
        # window passed to TimeSeriesFold, ensuring both splitters agree on
        # where training ends and validation begins.
        splits = list(skl_cv.split(y_train))
        initial_train_size = len(splits[0][0]) if splits else n_train_cv

        return TimeSeriesFold(
            steps=cv_block,
            refit=False,
            initial_train_size=initial_train_size,
            fixed_train_size=(self.config.train_size is not None),
            gap=0,
            allow_incomplete_fold=True,
        )

    def create_forecaster(self, target: Optional[str] = None) -> Any:
        """Create a fresh forecaster for the given target.

        Delegates to ``config.forecaster_factory`` when set; otherwise falls
        back to ``default_lgbm_forecaster_factory``.  This factory hook lets
        callers swap the estimator without subclassing ``BaseTask``.

        Args:
            target: Optional target column name.  Forwarded to the factory
                so that custom factories can specialise per target.

        Returns:
            A new, unfitted forecaster instance.

        Examples:
            ```{python}
            import tempfile
            from pathlib import Path
            from spotforecast2_safe.multitask import LazyTask
            from spotforecast2_safe.configurator.config_multi import ConfigMulti

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(
                    predict_size=6,
                    use_exogenous_features=False,
                    cache_home=Path(tmp),
                )
                task = LazyTask(cfg)
                forecaster = task.create_forecaster()
            print(f"Type: {type(forecaster).__name__}")
            print(f"Lags: {forecaster.lags}")
            ```
        """
        factory = self.config.forecaster_factory or default_lgbm_forecaster_factory
        return factory(self.config, weight_func=self.weight_func, target=target)

    # ------------------------------------------------------------------
    # Tuning-result persistence
    # ------------------------------------------------------------------

    def save_tuning_results(
        self,
        target: str,
        task_name: str,
        best_params: Dict[str, Any],
        best_lags: Any,
    ) -> Path:
        """Save tuning results (best parameters and lags) to a JSON file.

        The file is stored under ``<cache_home>/tuning_results/`` with a
        datetime-stamped filename so that loaders can determine freshness.

        Filename format::

            <data_frame_name>_<target>_<task_name>_<YYYYMMDD_HHMMSS>.json

        Args:
            target: Name of the forecast target column.
            task_name: Tuning algorithm identifier (e.g. ``"optuna"``,
                ``"spotoptim"``).
            best_params: Best hyperparameters discovered during tuning.
            best_lags: Best lag configuration (int, list, or nested list).

        Returns:
            Path to the saved JSON file.

        Examples:
            ```{python}
            import tempfile
            from pathlib import Path
            from spotforecast2_safe.multitask import LazyTask
            from spotforecast2_safe.configurator.config_multi import ConfigMulti

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(data_frame_name="demo10", cache_home=Path(tmp))
                task = LazyTask(cfg)
                path = task.save_tuning_results(
                    target="target_0",
                    task_name="optuna",
                    best_params={"n_estimators": 100, "learning_rate": 0.05},
                    best_lags=[1, 2, 24],
                )
                print(path.name[:10])
            ```
        """
        tuning_dir = get_cache_home(self.config.cache_home) / "tuning_results"
        tuning_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{self.config.data_frame_name}_{target}_{task_name}_{timestamp}.json"
        )
        filepath = tuning_dir / filename

        # Convert lags to a JSON-safe type
        lags_serializable: Any = best_lags
        if hasattr(best_lags, "tolist"):
            lags_serializable = best_lags.tolist()

        payload = {
            "data_frame_name": self.config.data_frame_name,
            "target": target,
            "task_name": task_name,
            "timestamp": timestamp,
            "best_params": best_params,
            "best_lags": lags_serializable,
        }

        with open(filepath, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

        self.logger.info("  Saved tuning results: %s", filepath)
        return filepath

    def load_tuning_results(
        self,
        target: str,
        task_name: Optional[str] = None,
        max_age_days: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent tuning results for a target from cache.

        Scans ``<cache_home>/tuning_results/`` for files matching the
        current ``data_frame_name`` and ``target``.  Optionally filters by
        ``task_name`` and discards results older than ``max_age_days``.

        Args:
            target: Name of the forecast target column.
            task_name: If given, only consider results from this tuning
                algorithm (e.g. ``"optuna"`` or ``"spotoptim"``).
                ``None`` accepts any algorithm.
            max_age_days: Maximum age in days.  Results older than this
                are ignored.  ``None`` accepts any age.

        Returns:
            A dictionary with keys ``best_params``, ``best_lags``,
            ``task_name``, ``target``, ``data_frame_name``, and
            ``timestamp``; or ``None`` if no matching file was found.

        Examples:
            ```{python}
            import tempfile
            from pathlib import Path
            from spotforecast2_safe.multitask import LazyTask
            from spotforecast2_safe.configurator.config_multi import ConfigMulti

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(data_frame_name="demo10", cache_home=Path(tmp))
                task = LazyTask(cfg)
                task.save_tuning_results(
                    target="target_0",
                    task_name="optuna",
                    best_params={"n_estimators": 100},
                    best_lags=24,
                )
                result = task.load_tuning_results(target="target_0")
                print(result["best_params"])
            ```
        """
        tuning_dir = get_cache_home(self.config.cache_home) / "tuning_results"
        if not tuning_dir.exists():
            return None

        prefix = f"{self.config.data_frame_name}_{target}_"
        candidates: List[Path] = sorted(
            (
                p
                for p in tuning_dir.glob(f"{prefix}*.json")
                if task_name is None or f"_{task_name}_" in p.name
            ),
            reverse=True,
        )

        now = datetime.now(timezone.utc)
        for candidate in candidates:
            try:
                with open(candidate) as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue

            if max_age_days is not None:
                ts = datetime.strptime(data["timestamp"], "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                age_days = (now - ts).total_seconds() / 86400
                if age_days > max_age_days:
                    continue

            self.logger.info("  Loaded tuning results from: %s", candidate)
            return data

        return None

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    # Maps each task key to the task_name string used in filenames.
    # The "optuna" and "spotoptim" entries are kept here as filename-parsing
    # identifiers only so that sibling-produced caches remain loadable.
    _TASK_MODEL_NAMES: Dict[str, str] = {
        "lazy": "task 1: Lazy Fitting",
        "defaults": "task 2: Defaults",
        "optuna": "task 3: Optuna Tuned",
        "spotoptim": "task 4: SpotOptim Tuned",
    }

    def save_models(
        self,
        task_name: str,
        forecasters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Save fitted forecaster models to the cache directory.

        Each model is serialised with ``joblib`` (compress=3) into
        ``<cache_home>/models/<data_frame_name>/`` using a datetime-stamped
        filename so that multiple snapshots can coexist.

        Filename format::

            <data_frame_name>_<target>_<task_name>_<YYYYMMDD_HHMMSS>.joblib

        If ``forecasters`` is ``None`` the method collects fitted models
        from ``self.results[task_name]``, where each prediction package is
        expected to contain a ``"forecaster"`` key.

        Args:
            task_name: Task identifier (``"lazy"``, ``"defaults"``).
                The names ``"optuna"`` and ``"spotoptim"`` are also accepted
                so that model caches produced by the ``spotforecast2``
                sibling package can be saved and loaded; no tuning is
                performed in this package.
            forecasters: Optional mapping ``{target: fitted_forecaster}``.
                When ``None``, models are taken from the prediction
                packages stored in ``self.results``.

        Returns:
            Mapping ``{target: Path}`` of saved model file paths.

        Raises:
            ValueError: If ``task_name`` is not one of ``"lazy"``,
                ``"defaults"``, ``"optuna"``, ``"spotoptim"``.
            RuntimeError: If no fitted models are available for the
                requested task.
        """
        if task_name not in self._TASK_MODEL_NAMES:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Choose from {list(self._TASK_MODEL_NAMES)}"
            )

        # Resolve forecaster objects
        if forecasters is None:
            task_results = self.results.get(task_name)
            if not task_results:
                raise RuntimeError(
                    f"No results for task '{task_name}'. "
                    "Run the task before calling save_models()."
                )
            forecasters = {}
            for target, pkg in task_results.items():
                if "forecaster" not in pkg:
                    raise RuntimeError(
                        f"Prediction package for target '{target}' does not "
                        "contain a 'forecaster' key."
                    )
                forecasters[target] = pkg["forecaster"]

        model_dir = (
            get_cache_home(self.config.cache_home)
            / "models"
            / self.config.data_frame_name
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        saved: Dict[str, Path] = {}

        for target, forecaster in forecasters.items():
            filename = (
                f"{self.config.data_frame_name}_{target}_{task_name}_{timestamp}.joblib"
            )
            filepath = model_dir / filename
            _joblib_dump(forecaster, filepath, compress=3)
            saved[target] = filepath
            self.logger.info("  Saved model: %s", filepath)

        return saved

    def load_models(
        self,
        task_name: Optional[str] = None,
        target: Optional[str] = None,
        max_age_days: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load the most recent fitted models from the cache directory.

        Scans ``<cache_home>/models/<data_frame_name>/`` for ``.joblib``
        files matching the current ``data_frame_name``.  Optionally
        filters by ``task_name``, ``target``, and ``max_age_days``.

        Args:
            task_name: If given, only load models from this task
                (``"lazy"``, ``"defaults"``, ``"optuna"``, or
                ``"spotoptim"``).  ``None`` accepts any task.
            target: If given, only load the model for this target
                column.  ``None`` loads the most recent model for
                every target found.
            max_age_days: Maximum age in days.  Models older than
                this are ignored.  ``None`` accepts any age.

        Returns:
            Mapping ``{target: forecaster}`` of loaded model objects.
            Empty dict if no matching models were found.
        """
        model_dir = (
            get_cache_home(self.config.cache_home)
            / "models"
            / self.config.data_frame_name
        )
        if not model_dir.exists():
            return {}

        prefix = f"{self.config.data_frame_name}_"
        candidates: List[Path] = sorted(
            model_dir.glob(f"{prefix}*.joblib"),
            reverse=True,
        )

        if task_name is not None:
            candidates = [p for p in candidates if f"_{task_name}_" in p.name]

        if target is not None:
            candidates = [p for p in candidates if f"_{target}_" in p.name]

        if max_age_days is not None:
            now = datetime.now(timezone.utc)
            filtered = []
            for p in candidates:
                # Extract timestamp from filename:
                # <data_frame_name>_<target>_<task_name>_<YYYYMMDD_HHMMSS>.joblib
                stem = p.stem  # without .joblib
                ts_str = stem[-15:]  # YYYYMMDD_HHMMSS
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                    age_days = (now - ts).total_seconds() / 86400
                    if age_days <= max_age_days:
                        filtered.append(p)
                except ValueError:
                    continue
            candidates = filtered

        # For each target, keep only the most recent file.
        # Files are already sorted newest-first.
        loaded: Dict[str, Any] = {}
        for p in candidates:
            # Parse target from filename:
            # <data_frame_name>_<target>_<task_name>_<YYYYMMDD_HHMMSS>.joblib
            stem = p.stem
            # Remove the prefix (data_frame_name + "_")
            rest = stem[len(prefix) :]
            # Remove the timestamp suffix ("_YYYYMMDD_HHMMSS")
            rest_no_ts = rest[:-16]  # strip "_YYYYMMDD_HHMMSS"
            # rest_no_ts = "<target>_<task_name>"
            # Split on known task names to extract target
            parsed_target = None
            for tname in self._TASK_MODEL_NAMES:
                suffix = f"_{tname}"
                if rest_no_ts.endswith(suffix):
                    parsed_target = rest_no_ts[: -len(suffix)]
                    break
            if parsed_target is None:
                # Unknown task — use everything before the last underscore
                # group as the target.
                parts = rest_no_ts.rsplit("_", 1)
                parsed_target = parts[0] if len(parts) > 1 else rest_no_ts

            if parsed_target in loaded:
                continue  # already have a newer file for this target

            try:
                loaded[parsed_target] = _joblib_load(p)
                self.logger.info("  Loaded model from: %s", p)
            except (OSError, EOFError) as exc:
                self.logger.warning("  Failed to load model %s: %s", p, exc)
                continue

        return loaded

    def _train_and_predict_target(
        self,
        target: str,
        task_name: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit, save, and build prediction package for one target."""
        forecaster.fit(y=y_train, exog=exog_train)
        pkg = build_prediction_package(
            forecaster=forecaster,
            target=target,
            y_train=y_train,
            predict_size=self.config.predict_size,
            exog_train=exog_train,
            exog_future=exog_future,
            df_test=self.df_test,
            index_name=self.config.index_name,
        )
        pkg["forecaster"] = forecaster
        return pkg

    def _get_target_data(self, target: str) -> tuple:
        """Extract ``(y_train, exog_train, exog_future)`` for one target."""
        return get_target_data(
            target=target,
            df_pipeline=self.df_pipeline,
            config=self.config,
            data_with_exog=self.data_with_exog,
            exog_feature_names=(
                self.exog_feature_names if self.exog_feature_names else None
            ),
            exo_pred=self.exo_pred,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def agg_predictor(
        self,
        results: Dict[str, Dict[str, Any]],
        targets: List[str],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Aggregate per-target prediction packages into a weighted forecast.

        Delegates to the module-level ``agg_predictor`` function.
        Available as an instance method so that subclasses can override the
        aggregation strategy when needed.

        Args:
            results: Mapping of target name to prediction package (as
                returned by ``build_prediction_package``).
            targets: Ordered list of target names to include.
            weights: Per-target aggregation weights aligned with ``targets``.

        Returns:
            Aggregated prediction package dict.
        """
        return agg_predictor(results, targets, weights)

    # ------------------------------------------------------------------
    # Visualisation hooks (no-ops in spotforecast2-safe)
    # ------------------------------------------------------------------

    def _show_prediction_figure(
        self,
        pred_pkg: Dict[str, Any],
        target: str,
        task_name: str,
    ) -> None:
        """No-op visualisation hook for one target.

        In ``spotforecast2-safe`` this method is a no-op: it logs a debug
        message and returns immediately.  The ``spotforecast2`` sibling
        package overrides this hook to display an interactive Plotly figure.

        Args:
            pred_pkg: Prediction package for this target.
            target: Target column name.
            task_name: Human-readable task label for the figure title.
        """
        logger.debug(
            "Rendering unavailable in spotforecast2-safe (target=%r, task=%r). "
            "Use the spotforecast2 package for interactive visualisation.",
            target,
            task_name,
        )

    def _show_prediction_figure_agg(
        self,
        agg_pkg: Dict[str, Any],
        task_name: str,
    ) -> None:
        """No-op visualisation hook for the aggregated forecast.

        In ``spotforecast2-safe`` this method is a no-op: it logs a debug
        message and returns immediately.  The ``spotforecast2`` sibling
        package overrides this hook to display an interactive Plotly figure.

        Args:
            agg_pkg: Aggregated prediction package.
            task_name: Human-readable task label for the figure title.
        """
        logger.debug(
            "Aggregated rendering unavailable in spotforecast2-safe (task=%r). "
            "Use the spotforecast2 package for interactive visualisation.",
            task_name,
        )

    def _aggregate_and_show(
        self,
        results: Dict[str, Dict[str, Any]],
        task_name: str,
        show: bool = False,
    ) -> Dict[str, Any]:
        """Aggregate results and optionally invoke the display hook.

        For multi-target configurations, aggregates via ``agg_predictor``
        using either the configured ``agg_weights`` or equal weights as a
        fallback.  For single-target configurations, aggregation is a no-op:
        the per-target prediction package is returned directly and ``show``
        has no effect (the aggregated-figure hook is never invoked).  The
        result is stored in ``agg_results`` keyed by ``task_name``.
        """
        if len(self.config.targets) == 1:
            target = self.config.targets[0]
            agg_pkg = results[target]
            self.agg_results[task_name] = agg_pkg
            return agg_pkg

        if self.config.agg_weights is not None:
            active_weights = self.config.agg_weights[: len(self.config.targets)]
        else:
            n = len(self.config.targets)
            active_weights = [1.0 / n] * n
            self.logger.info(
                "No agg_weights configured — using equal weights (1/%d each).", n
            )

        agg_pkg = self.agg_predictor(
            results=results,
            targets=self.config.targets,
            weights=active_weights,
        )
        self.agg_results[task_name] = agg_pkg
        if show:
            self._show_prediction_figure_agg(agg_pkg, task_name)
        return agg_pkg

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def _run_strategy(
        self,
        strategy: Any,
        task_name: str,
        results_key: str,
        show: bool = False,
        log_prefix: str = "",
    ) -> Dict[str, Any]:
        """Run a ``TrainingStrategy`` over every configured target.

        Centralised per-target loop used by ``execute_lazy`` /
        ``execute_defaults``.  The strategy is responsible only for the
        per-target prepare step (parameter application); the final
        fit, optional model save, aggregation, and display all stay in this
        method so the previously observable ordering is preserved.

        Args:
            strategy: A ``TrainingStrategy`` instance.
            task_name: Human-readable label used in log lines, figure titles,
                and ``agg_results`` keys.
            results_key: Key under which per-target packages are stored in
                ``self.results``.
            show: If ``True``, invoke the per-target and aggregated display
                hooks (no-ops in this package).
            log_prefix: Optional log-line prefix (e.g. ``"[task 1] "``).

        Returns:
            Aggregated prediction package.
        """
        self._ensure_pipeline_ready()
        results: Dict[str, Dict[str, Any]] = {}

        for target in self.config.targets:
            if log_prefix:
                self.logger.info("%sTarget '%s': %s ...", log_prefix, target, task_name)
            else:
                self.logger.info("Target '%s': %s ...", target, task_name)

            y_train, exog_train, exog_future = self._get_target_data(target)
            forecaster = self.create_forecaster(target=target)
            prepared = strategy.prepare_forecaster(
                self,
                target=target,
                forecaster=forecaster,
                y_train=y_train,
                exog_train=exog_train,
            )
            results[target] = self._train_and_predict_target(
                target=target,
                task_name=task_name,
                forecaster=prepared,
                y_train=y_train,
                exog_train=exog_train,
                exog_future=exog_future,
            )
            if show:
                self._show_prediction_figure(results[target], target, task_name)

        self.results[results_key] = results
        if getattr(self.config, "auto_save_models", True):
            self.save_models(task_name=results_key)
        return self._aggregate_and_show(results, task_name, show=show)

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _ensure_pipeline_ready(self) -> None:
        """Raise if data has not been prepared and training window not set."""
        if self.df_pipeline is None:
            raise RuntimeError("Pipeline data not prepared. Call prepare_data() first.")
        if self.config.end_train_ts is None:
            self._setup_training_window()

    # ------------------------------------------------------------------
    # Pipeline summary
    # ------------------------------------------------------------------

    def log_summary(self) -> None:
        """Log a summary of the current pipeline configuration."""
        self.logger.info("=" * 60)
        self.logger.info("DATA PROCESSING PIPELINE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(
            "  Outlier detection:   %s",
            (
                f"ON (IsolationForest, contamination={self.config.contamination})"
                if self.config.use_outlier_detection
                else "OFF"
            ),
        )
        self.logger.info("  Imputation method:   %s", self.config.imputation_method)
        self.logger.info(
            "  Exogenous features:  %s",
            (
                f"ON ({len(self.exog_feature_names)} features selected)"
                if self.config.use_exogenous_features
                else "OFF"
            ),
        )
        self.logger.info(
            "  Weight function:     %s",
            "YES" if self.weight_func is not None else "NO",
        )
        if self.data_with_exog is not None:
            self.logger.info("  Merged data shape:   %s", self.data_with_exog.shape)
            self.logger.info("  Prediction exog:     %s", self.exo_pred.shape)
        self.logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Abstract run
    # ------------------------------------------------------------------

    def run(  # noqa: PLR0913
        self,
        show: bool = False,
        task: Optional[str] = None,
        task_name: Optional[str] = None,
        use_tuned_params: bool = True,
        max_age_days: Optional[float] = None,
        search_space: Optional[Any] = None,
        dry_run: bool = False,
        cache_home: Optional[Path] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the task-specific training / prediction pipeline.

        Subclasses must override this method.

        Args:
            show: If ``True``, invoke the visualisation hooks (no-ops in
                this package; meaningful only in ``spotforecast2``).
            task: Task mode override (used by ``MultiTask``).
            task_name: Restrict model loading to a specific source task
                (used by ``PredictTask``).
            use_tuned_params: Load cached tuning results when available
                (used by ``LazyTask``).
            max_age_days: Maximum age in days for cached results
                (used by ``LazyTask`` and ``PredictTask``).  Freshness is
                judged against the wall-clock timestamp embedded in the
                cache filename, so the check is machine-local.
            search_space: Hyperparameter search-space definition
                (accepted for API compatibility; not used in this package).
            dry_run: Report what would be deleted without removing anything
                (used by ``CleanTask``).
            cache_home: Override the cache directory (used by ``CleanTask``).
            **kwargs: Additional task-specific arguments.

        Returns:
            Aggregated prediction package for the task.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run(). "
            "Use LazyTask, DefaultsTask, PredictTask, or CleanTask."
        )

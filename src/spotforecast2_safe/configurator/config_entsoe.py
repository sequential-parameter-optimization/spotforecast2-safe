# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration for ENTSO-E task pipeline."""

from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from spotforecast2_safe.configurator._base_config import (
    apply_set_params,
    build_get_params,
    default_periods,
    validate_config,
)
from spotforecast2_safe.data import Period


class ConfigEntsoe:
    """Configuration for the ENTSO-E forecasting pipeline.

    Single-target counterpart to ``ConfigMulti``.  Used by the ENTSO-E CLI
    (``spotforecast2.tasks.task_entsoe``) and any other single-target pipeline
    routed through ``spotforecast2.multitask.runner.run(config_cls=ConfigEntsoe)``.

    ``country_code`` is the canonical ISO 3166-1 alpha-2 country-code
    attribute used by both API queries and the multitask ``PipelineConfig``
    protocol.

    Args:
        country_code (str): ISO 3166-1 alpha-2 country code (e.g. ``"DE"``).
        periods (Optional[List[Period]]): Cyclical feature encodings.
        lags_consider (Optional[List[int]]): Lag values for autoregressive features.
        train_size (Optional[pd.Timedelta]): Training window.
        end_train_default (str): Default end-of-training timestamp (ISO).
        delta_val (Optional[pd.Timedelta]): Validation window.
        predict_size (int): Prediction horizon in hours.
        cv_block_size (int | None): Cross-validation test-block width in
            hours.  Defaults to ``None``, meaning the CV uses
            ``predict_size``.  Set to a fixed value (e.g. ``24``) to
            decouple the cross-validation horizon from a render-dependent
            live ``predict_size``.
        refit_size (int): Refit cadence in days.
        random_state (int): Random seed.
        n_hyperparameters_trials (int): Hyperparameter-tuning trial budget.
        data_filename (str): Path to the merged interim CSV.
        targets (Optional[List[str]]): Active target column names.  ``None``
            until set after data loading.  For ENTSO-E this is typically
            ``["Actual Load"]``.
        use_outlier_detection (bool): Apply IsolationForest-based outlier
            removal.  Defaults to ``True``.
        contamination (float): IsolationForest contamination fraction.
        imputation_method (str): Gap-filling strategy.
        window_size (int): Rolling window for weighted imputation. Also the
            LightGBM rolling-mean feature window in the ENTSO-E factories.
        imputation_window_size (Optional[int]): Width of the gap-penalty zone
            (in hours) around each imputed value for the ``"weighted"``
            strategy. When ``None`` (default), falls back to ``window_size``,
            so existing behaviour is unchanged. Set this to decouple the
            imputation penalty zone from the rolling-feature window.
        use_exogenous_features (bool): Build weather/calendar/holiday features.
        latitude (float): Location latitude.
        longitude (float): Location longitude.
        timezone (str): IANA timezone string.
        state (str): Subdivision code for regional holidays.
        include_weather_windows (bool): Weather-window feature toggle.
        include_holiday_features (bool): Holiday feature toggle.
        include_holiday_adjacency_features (bool): Brückentag and
            before/after-holiday indicator toggle.  Defaults to ``False``.
        poly_features_degree (int): Polynomial-interaction degree passed to
            the feature builder. ``1`` (default) generates no interactions;
            ``2`` adds pairwise bilinear terms; ``3+`` higher order.
        max_poly_features (int): Cap on polynomial interaction columns. When
            more than this many ``poly_*`` columns are generated, only the
            top ``max_poly_features`` ranked by mutual information with the
            target are kept (``<= 0`` disables the cap). Defaults to ``10``.
        poly_mi_n_jobs (Optional[int]): Parallel jobs for the
            mutual-information ranking that enforces ``max_poly_features``.
            ``-1`` (default) uses all cores; ``None`` runs single-threaded.
            Parallelism does not change the selection.
        poly_mi_sample_size (Optional[int]): Row cap for that ranking; longer
            series are scored on a reproducible random subsample of this size
            (seeded by ``random_state``), which can change which borderline
            columns make the top K. ``None`` scores every row (the pre-15.8
            behaviour). Defaults to ``4000``.
        include_covid_infection_rate (bool): Append the bundled German national
            COVID-19 7-day incidence (RKI) as an exogenous level regressor.
            Defaults to ``False``.
        include_entsoe_forecast_load (bool): Append the ENTSO-E day-ahead
            Forecasted Load as a near-oracle exogenous prior. Defaults to
            ``False``.
        include_entsoe_renewable_forecast (bool): Append the ENTSO-E day-ahead
            wind and solar generation forecast. Defaults to ``False``.
        include_entsoe_net_load (bool): Append the ENTSO-E day-ahead net load
            (Forecasted Load minus wind/solar forecast). Defaults to ``False``.
        include_entsoe_day_ahead_price (bool): Append the ENTSO-E day-ahead
            spot price (DE/LU). Defaults to ``False``.
        index_name (str): Datetime column name when the DataFrame index is
            reset.  ENTSO-E CSVs use ``"Time (UTC)"``; defaults to that.
        bounds (Optional[List[tuple]]): Per-column outlier bounds.  For
            single-target ENTSO-E this is typically ``None`` or a single
            ``[(lower, upper)]`` entry.
        verbose (bool): Verbose pipeline output.
        cache_home (Optional[Any]): Cache directory override.
        n_trials_optuna (int): Optuna Bayesian-search trial budget.
        n_trials_spotoptim (int): SpotOptim surrogate-search trial budget.
        n_initial_spotoptim (int): SpotOptim initial random evaluations.
        n_jobs_spotoptim (Optional[int]): Worker count for SpotOptim's parallel
            (steady-state) evaluation. ``None`` (default) runs sequentially;
            ``-1`` uses all CPU cores; a positive integer pins the worker count.
            Parallel tuning is faster but, being steady-state, changes the search
            trajectory, so the tuned result is not bit-identical to a sequential
            run even with a fixed ``random_state``.
        warm_start_lags (bool): Seed the SpotOptim search with ``lags_consider``.
        task (str): Active prediction task name.
        agg_weights (Optional[List[float]]): Per-target aggregation weights.
            For single-target use this is typically ``[1.0]`` or ``None``.
        forecaster_factory (Optional[Any]): Callable
            ``factory(config, *, weight_func, target) -> forecaster``
            consumed by ``BaseTask.create_forecaster``.  ``None`` falls back
            to the default LightGBM factory.
        data_loader (Optional[Any]): Callable ``data_loader(config)`` returning
            a pandas DataFrame.  Invoked by ``BaseTask.prepare_data`` when no
            DataFrame is supplied — the ENTSO-E pipeline hook for
            ``download_new_data`` / ``merge_build_manual``.
        test_data_loader (Optional[Any]): Callable ``test_data_loader(config)``
            returning a pandas DataFrame with ground-truth values for the
            prediction horizon.  Invoked by ``BaseTask.prepare_data`` when no
            test DataFrame is supplied; the returned frame populates
            ``test_actual`` and ``metrics_future`` in the prediction package.
        auto_save_models (bool): Whether ``BaseTask._run_strategy`` should
            persist fitted forecasters to ``<cache_home>/models/`` after every
            training run.  Defaults to ``True``.
        data_frame_name (str): Identifier for the active dataset.  Used by
            ``BaseTask`` to name cache subdirectories, model files, and the
            per-dataset log file.  Defaults to ``"default"``.
        number_folds (int): Number of folds used by ``BaseTask.cv_ts`` when
            building the ``TimeSeriesSplit`` cross-validation splitter for
            tuning tasks.  Defaults to ``10``.
        on_weather_failure (Literal["raise", "skip"]): Policy for handling
            Open-Meteo fetch failures inside
            ``BaseTask.build_exogenous_features``.  ``"raise"`` (default)
            aborts the pipeline with a ``WeatherFetchError`` and preserves
            the safety-critical fail-safe semantics.  ``"skip"`` logs a
            warning and continues with empty weather features so the rest
            of the pipeline can run without the Open-Meteo dependency.
        on_exog_provider_failure (Literal["raise", "skip"]): Policy for an
            exogenous-provider failure inside ``ExogBuilder.build``. ``"raise"``
            (default) propagates the ``ExogProviderError`` (fail-safe);
            ``"skip"`` logs a warning and omits that provider's columns.
        exog_max_gap_hours (int): Maximum length, in hours, of a contiguous run
            of missing exogenous-provider values healed before the provider is
            rejected. Interior gaps are time-interpolated; leading/trailing edge
            gaps are back-/forward-filled. ``0`` (default) keeps the strict
            fail-safe (any gap raises). Healed runs are logged with count and
            span. Only already-published day-ahead vintages are involved, so
            healing is leakage-clean (CR-3).
        exog_max_tail_gap_hours (int): Extended healing budget, in hours,
            applied exclusively to the trailing-edge NaN run (the run
            containing the last index timestamp). The effective tail budget is
            ``max(exog_max_gap_hours, exog_max_tail_gap_hours)``. The canonical
            use case is the ENTSO-E day-ahead publication frontier: the last
            published vintage is zero-order-held forward to the forecast horizon
            without touching interior gaps (CR-3-clean). When
            ``exog_max_tail_gap_hours <= exog_max_gap_hours`` the parameter is
            inert (the interior budget already covers the tail) and a warning is
            logged. Defaults to ``0``.
        exog_provider_window (Literal["full", "train"]): Span the exogenous
            providers are validated against. ``"full"`` (default) requires
            coverage of the entire ``data_start``→``cov_end`` request, matching
            prior behaviour. ``"train"`` validates only the consumed window
            ``[start_train_ts, cov_end]``, tolerating missing values before the
            training window. Honoured by the MultiTask pipeline; the
            forecaster-wrapper path currently always validates the full span.
        retrain_max_age (pd.Timedelta): Maximum age of a previously trained
            model before retraining is required.  Consumed by
            ``spotforecast2_safe.manager.trainer.should_retrain`` to gate
            scheduled retraining workflows.  Defaults to ``Timedelta(days=7)``.

    Attributes:
        country_code (str): ISO country code used for API queries and
            holiday feature generation.
        auto_save_models (bool): Whether to auto-persist fitted forecasters
            after each training run.
        data_frame_name (str): Active-dataset identifier used for cache and
            log-file naming.
        number_folds (int): Cross-validation fold count for tuning tasks.
        on_weather_failure (Literal["raise", "skip"]): Open-Meteo fetch-failure
            policy: ``"raise"`` aborts, ``"skip"`` continues without weather.

    Examples:
        ```{python}
        import pandas as pd

        from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe

        # Use default configuration
        config = ConfigEntsoe()
        print(config.country_code)
        print(config.predict_size)
        print(config.random_state)

        # Create custom configuration
        custom_config = ConfigEntsoe(
            country_code="FR",
            predict_size=48,
            cv_block_size=24,
            random_state=42,
        )
        print(custom_config.country_code)
        print(custom_config.predict_size)
        print(custom_config.cv_block_size)

        # Verify training window
        assert config.train_size == pd.Timedelta(days=3 * 365)

        # Check default periods
        print(len(config.periods))
        print(config.periods[0].name)
        ```
    """

    _PARAM_NAMES = (
        "country_code",
        "periods",
        "lags_consider",
        "train_size",
        "end_train_default",
        "delta_val",
        "predict_size",
        "cv_block_size",
        "refit_size",
        "random_state",
        "n_hyperparameters_trials",
        "data_filename",
        "targets",
        "use_outlier_detection",
        "contamination",
        "imputation_method",
        "window_size",
        "imputation_window_size",
        "use_exogenous_features",
        "latitude",
        "longitude",
        "timezone",
        "state",
        "include_weather_windows",
        "include_holiday_features",
        "include_holiday_adjacency_features",
        "poly_features_degree",
        "max_poly_features",
        "poly_mi_n_jobs",
        "poly_mi_sample_size",
        "include_covid_infection_rate",
        "include_entsoe_forecast_load",
        "include_entsoe_renewable_forecast",
        "include_entsoe_net_load",
        "include_entsoe_day_ahead_price",
        "index_name",
        "bounds",
        "verbose",
        "cache_home",
        "n_trials_optuna",
        "n_trials_spotoptim",
        "n_initial_spotoptim",
        "n_jobs_spotoptim",
        "warm_start_lags",
        "task",
        "agg_weights",
        "forecaster_factory",
        "data_loader",
        "test_data_loader",
        "auto_save_models",
        "data_frame_name",
        "number_folds",
        "on_weather_failure",
        "on_exog_provider_failure",
        "exog_max_gap_hours",
        "exog_max_tail_gap_hours",
        "exog_provider_window",
        "retrain_max_age",
        "target_qc_range_mw",
        "target_qc_step_mw",
        "target_qc_window_days",
        "target_corruption_policy",
        "target_max_heal_hours",
        "target_anchor_zone_hours",
        "target_qc_deviation_mw",
        "target_qc_deviation_ref",
        "target_qc_deviation_slots",
    )

    def __init__(
        self,
        country_code: str = "DE",
        periods: Optional[List[Period]] = None,
        lags_consider: Optional[List[int]] = None,
        train_size: Optional[pd.Timedelta] = None,
        end_train_default: str = "2025-12-31 00:00+00:00",
        delta_val: Optional[pd.Timedelta] = None,
        predict_size: int = 24,
        cv_block_size: Optional[int] = None,
        refit_size: int = 7,
        random_state: int = 314159,
        n_hyperparameters_trials: int = 20,
        data_filename: str = "interim/energy_load.csv",
        targets: Optional[List[str]] = None,
        # Outlier detection
        use_outlier_detection: bool = True,
        contamination: float = 0.01,
        # Imputation
        imputation_method: str = "weighted",
        window_size: int = 72,
        imputation_window_size: Optional[int] = None,
        # Exogenous features
        use_exogenous_features: bool = True,
        latitude: float = 51.5136,
        longitude: float = 7.4653,
        timezone: str = "UTC",
        state: str = "NW",
        # Feature selection toggles
        include_weather_windows: bool = False,
        include_holiday_features: bool = False,
        include_holiday_adjacency_features: bool = False,
        poly_features_degree: int = 1,
        max_poly_features: int = 10,
        poly_mi_n_jobs: Optional[int] = -1,
        poly_mi_sample_size: Optional[int] = 4000,
        # Provider-based exogenous toggles (preprocessing.exog_providers)
        include_covid_infection_rate: bool = False,
        include_entsoe_forecast_load: bool = False,
        include_entsoe_renewable_forecast: bool = False,
        include_entsoe_net_load: bool = False,
        include_entsoe_day_ahead_price: bool = False,
        # Data source and index
        index_name: str = "Time (UTC)",
        # Per-column outlier bounds
        bounds: Optional[List[tuple]] = None,
        # Verbosity and caching
        verbose: bool = False,
        cache_home: Optional[Any] = None,
        # Hyperparameter tuning trial budgets
        n_trials_optuna: int = 15,
        n_trials_spotoptim: int = 10,
        n_initial_spotoptim: int = 5,
        # SpotOptim parallel-evaluation worker count (None=serial, -1=all cores);
        # consumed by spotforecast2.multitask.strategies.SpotOptimStrategy
        n_jobs_spotoptim: Optional[int] = None,
        # Seed the SpotOptim search with ``lags_consider`` (consumed by
        # spotforecast2.multitask.strategies.SpotOptimStrategy)
        warm_start_lags: bool = False,
        # Active task
        task: str = "lazy",
        # Aggregation weights (single-target uses [1.0] or None)
        agg_weights: Optional[List[float]] = None,
        # Forecaster factory hook (consumed by spotforecast2.multitask.base)
        forecaster_factory: Optional[Any] = None,
        # Data-loader hook (consumed by spotforecast2.multitask.base.prepare_data)
        data_loader: Optional[Any] = None,
        # Test-data-loader hook (consumed by spotforecast2.multitask.base.prepare_data)
        test_data_loader: Optional[Any] = None,
        # Persistence policy and active-dataset name (consumed by spotforecast2.multitask.base)
        auto_save_models: bool = True,
        data_frame_name: str = "default",
        # Cross-validation fold count (consumed by spotforecast2.multitask.base.cv_ts)
        number_folds: int = 10,
        # Weather-fetch failure policy (consumed by spotforecast2.multitask.base.build_exogenous_features)
        on_weather_failure: Literal["raise", "skip"] = "raise",
        # Exog-provider failure policy (consumed by preprocessing.exog_builder.ExogBuilder)
        on_exog_provider_failure: Literal["raise", "skip"] = "raise",
        # Gap-healing budget for exog providers (0 = strict fail-safe)
        exog_max_gap_hours: int = 0,
        # Extended trailing-edge healing budget (0 = same as exog_max_gap_hours)
        exog_max_tail_gap_hours: int = 0,
        # Validation window for exog providers ("full" or "train")
        exog_provider_window: Literal["full", "train"] = "full",
        # Retraining cadence (consumed by spotforecast2_safe.manager.trainer.should_retrain)
        retrain_max_age: Optional[pd.Timedelta] = None,
        # Target-side corruption detector knobs.
        # Detector active only when target_qc_window_days AND at least one of
        # target_qc_range_mw / target_qc_step_mw / target_qc_deviation_mw are
        # set.  Defaults are all None / off, so the pipeline is byte-identical
        # to the pre-feature baseline.
        # Recommended episode policy: "truncate" (auto-extends predict_size).
        # "heal" under the default anchor_zone_hours=168 with a <=7-day QC window
        # never engages (refusal by design — lowering the zone is a deliberate
        # operator decision).
        # The deviation rule (dropout-only, vs a published reference column
        # such as "Forecasted Load") catches corruption that stays below the
        # dynamics thresholds; when enabling it, scope `targets` to the
        # actuals so heal/truncate leave the reference column intact.
        target_qc_range_mw: Optional[float] = None,
        target_qc_step_mw: Optional[float] = None,
        target_qc_window_days: Optional[int] = None,
        target_corruption_policy: str = "abort",
        target_max_heal_hours: int = 0,
        target_anchor_zone_hours: int = 168,
        target_qc_deviation_mw: Optional[float] = None,
        target_qc_deviation_ref: Optional[str] = None,
        target_qc_deviation_slots: int = 2,
    ):
        """Initialize ConfigEntsoe with specified or default parameters."""
        self.country_code = country_code

        self.periods = periods if periods is not None else default_periods()
        self.lags_consider = (
            lags_consider if lags_consider is not None else list(range(1, 24))
        )
        self.train_size = (
            train_size if train_size is not None else pd.Timedelta(days=3 * 365)
        )
        self.end_train_default = end_train_default
        self.delta_val = (
            delta_val if delta_val is not None else pd.Timedelta(hours=24 * 7 * 10)
        )
        self.predict_size = predict_size
        # Cross-validation test-block width (hours).  ``None`` defers to
        # ``predict_size``; the actual CV-split logic lives in the sibling
        # ``spotforecast2`` package (``BaseTask.cv_ts``).
        self.cv_block_size = cv_block_size
        self.refit_size = refit_size
        self.random_state = random_state
        self.n_hyperparameters_trials = n_hyperparameters_trials
        self.data_filename = data_filename
        self.targets = targets
        # Outlier detection
        self.use_outlier_detection = use_outlier_detection
        self.contamination = contamination
        # Imputation
        self.imputation_method = imputation_method
        self.window_size = window_size
        self.imputation_window_size = imputation_window_size
        # Exogenous features
        self.use_exogenous_features = use_exogenous_features
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.state = state
        # Feature selection toggles
        self.include_weather_windows = include_weather_windows
        self.include_holiday_features = include_holiday_features
        self.include_holiday_adjacency_features = include_holiday_adjacency_features
        self.poly_features_degree = poly_features_degree
        self.max_poly_features = max_poly_features
        self.poly_mi_n_jobs = poly_mi_n_jobs
        self.poly_mi_sample_size = poly_mi_sample_size
        # Provider-based exogenous toggles, each gated by a registry flag in
        # ``spotforecast2_safe.preprocessing.exog_providers``.
        self.include_covid_infection_rate = include_covid_infection_rate
        self.include_entsoe_forecast_load = include_entsoe_forecast_load
        self.include_entsoe_renewable_forecast = include_entsoe_renewable_forecast
        self.include_entsoe_net_load = include_entsoe_net_load
        self.include_entsoe_day_ahead_price = include_entsoe_day_ahead_price
        # Data source and index
        self.index_name = index_name
        # Per-column outlier bounds
        self.bounds = bounds
        # Verbosity and caching
        self.verbose = verbose
        self.cache_home = cache_home
        # Hyperparameter tuning trial budgets
        self.n_trials_optuna = n_trials_optuna
        self.n_trials_spotoptim = n_trials_spotoptim
        self.n_initial_spotoptim = n_initial_spotoptim
        self.n_jobs_spotoptim = n_jobs_spotoptim
        # When True, ``SpotOptimStrategy`` injects ``lags_consider`` as a
        # candidate lag set and seeds the optimizer's first evaluation with
        # it (via SpotOptim's ``x0``).  Pure data here; the behaviour lives
        # in the sibling ``spotforecast2`` package.
        self.warm_start_lags = warm_start_lags
        # Active task
        self.task = task
        # Aggregation weights
        self.agg_weights = agg_weights
        # Optional callable ``factory(config, *, weight_func, target) -> forecaster``.
        self.forecaster_factory = forecaster_factory
        # Optional callable ``data_loader(config) -> pd.DataFrame`` invoked
        # by ``BaseTask.prepare_data`` when no DataFrame is supplied.
        self.data_loader = data_loader
        # Optional callable ``test_data_loader(config) -> pd.DataFrame`` invoked
        # by ``BaseTask.prepare_data`` when no test DataFrame is supplied.
        # Returned frame populates ``test_actual`` and ``metrics_future`` in
        # the prediction package; mirrors ``data_loader`` for the test slice.
        self.test_data_loader = test_data_loader
        # Whether ``BaseTask._run_strategy`` should persist fitted models to
        # the cache directory after every training run.  Defaults to ``True``
        # so that saved models are immediately available for ``PredictTask``.
        self.auto_save_models = auto_save_models
        # Identifier for the active dataset, used by ``BaseTask`` for
        # cache-subdirectory naming, model-file naming, and per-dataset
        # log-file routing.
        self.data_frame_name = data_frame_name
        # Number of TimeSeriesSplit folds used by ``BaseTask.cv_ts`` when
        # building cross-validation splitters for tuning tasks.
        self.number_folds = number_folds
        # Policy for Open-Meteo fetch failures consumed by
        # ``BaseTask.build_exogenous_features``: ``"raise"`` aborts the
        # pipeline (default, preserves fail-safe semantics); ``"skip"``
        # logs a warning and continues without weather features.
        self.on_weather_failure = on_weather_failure
        # Policy for exog-provider failures consumed by
        # ``ExogBuilder``: ``"raise"`` aborts (default, fail-safe); ``"skip"``
        # logs a warning and omits the failing provider's columns.
        self.on_exog_provider_failure = on_exog_provider_failure
        # Maximum contiguous gap in hours that providers will heal (0 = strict).
        self.exog_max_gap_hours = exog_max_gap_hours
        # Extended trailing-edge healing budget (0 = same as exog_max_gap_hours).
        self.exog_max_tail_gap_hours = exog_max_tail_gap_hours
        # Validation window for providers: "full" (default) or "train".
        self.exog_provider_window = exog_provider_window
        # Maximum age of a previously trained model before retraining is
        # required.  Consumed by
        # ``spotforecast2_safe.manager.trainer.should_retrain``.
        self.retrain_max_age = (
            retrain_max_age if retrain_max_age is not None else pd.Timedelta(days=7)
        )
        # Target-side corruption detector and policy knobs.
        self.target_qc_range_mw = target_qc_range_mw
        self.target_qc_step_mw = target_qc_step_mw
        self.target_qc_window_days = target_qc_window_days
        self.target_corruption_policy = target_corruption_policy
        self.target_max_heal_hours = target_max_heal_hours
        self.target_anchor_zone_hours = target_anchor_zone_hours
        self.target_qc_deviation_mw = target_qc_deviation_mw
        self.target_qc_deviation_ref = target_qc_deviation_ref
        self.target_qc_deviation_slots = target_qc_deviation_slots
        validate_config(self)

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        """
        Get parameters for this configuration object.

        Args:
            deep: If True, will return the parameters for this configuration and
                contained sub-objects that are estimators.

        Returns:
            params: Dictionary of parameter names mapped to their values.

        Examples:
            ```{python}
            from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe

            config = ConfigEntsoe(country_code="FR")
            p = config.get_params()
            print(p["country_code"])
            print(p["predict_size"])
            assert p["country_code"] == "FR"
            assert p["predict_size"] == 24
            assert p["cv_block_size"] is None
            ```
        """
        return build_get_params(self, self._PARAM_NAMES, deep)

    def set_params(
        self, params: Dict[str, object] = None, **kwargs: object
    ) -> "ConfigEntsoe":
        """
        Set the parameters of this configuration object.

        Args:
            params: Optional dictionary of parameter names mapped to their
                new values.
            **kwargs: Additional parameter names mapped to their new values.
                It supports configuring nested 'Period' objects using the
                `periods__<name>__<param>` notation.

        Returns:
            ConfigEntsoe: The configuration instance with updated
                parameters (supports method chaining).

        Examples:
            ```{python}
            from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe

            config = ConfigEntsoe()

            # Flat parameter setting
            config.set_params(country_code="FR", predict_size=48)
            print(config.country_code)
            print(config.predict_size)
            assert config.country_code == "FR"
            assert config.predict_size == 48

            # Deep parameter setting for nested Period objects
            config.set_params(periods__daily__n_periods=24)
            daily_n = next(p.n_periods for p in config.periods if p.name == "daily")
            print(daily_n)
            assert daily_n == 24
            ```
        """
        return apply_set_params(self, params, **kwargs)

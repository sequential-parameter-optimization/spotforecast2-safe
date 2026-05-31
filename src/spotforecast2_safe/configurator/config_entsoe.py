# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration for ENTSO-E task pipeline."""

from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

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
        poly_features_degree (int): Polynomial-interaction degree passed to
            the feature builder. ``1`` (default) generates no interactions;
            ``2`` adds pairwise bilinear terms; ``3+`` higher order.
        max_poly_features (int): Cap on polynomial interaction columns. When
            more than this many ``poly_*`` columns are generated, only the
            top ``max_poly_features`` ranked by mutual information with the
            target are kept (``<= 0`` disables the cap). Defaults to ``10``.
        index_name (str): Datetime column name when the DataFrame index is
            reset.  ENTSO-E CSVs use ``"Time (UTC)"``; defaults to that.
        start_download (Optional[str]): Start of the data download range.
        end_download (Optional[str]): End of the data download range.
        data_start (Optional[pd.Timestamp]): First pipeline-data timestamp.
        data_end (Optional[pd.Timestamp]): Last pipeline-data timestamp.
        cov_start (Optional[pd.Timestamp]): Start of the covariate range.
        cov_end (Optional[pd.Timestamp]): End of the covariate range.
        bounds (Optional[List[tuple]]): Per-column outlier bounds.  For
            single-target ENTSO-E this is typically ``None`` or a single
            ``[(lower, upper)]`` entry.
        verbose (bool): Verbose pipeline output.
        cache_home (Optional[Any]): Cache directory override.
        end_train_ts (Optional[pd.Timestamp]): Derived end-of-training.
        start_train_ts (Optional[pd.Timestamp]): Derived start-of-training.
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
        poly_features_degree: int = 1,
        max_poly_features: int = 10,
        # Data source and index
        index_name: str = "Time (UTC)",
        start_download: Optional[str] = None,
        end_download: Optional[str] = None,
        # Derived date ranges (set after data loading via get_start_end())
        data_start: Optional[pd.Timestamp] = None,
        data_end: Optional[pd.Timestamp] = None,
        cov_start: Optional[pd.Timestamp] = None,
        cov_end: Optional[pd.Timestamp] = None,
        # Per-column outlier bounds
        bounds: Optional[List[tuple]] = None,
        # Verbosity and caching
        verbose: bool = False,
        cache_home: Optional[Any] = None,
        # Derived training window
        end_train_ts: Optional[pd.Timestamp] = None,
        start_train_ts: Optional[pd.Timestamp] = None,
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
        # Retraining cadence (consumed by spotforecast2_safe.manager.trainer.should_retrain)
        retrain_max_age: Optional[pd.Timedelta] = None,
    ):
        """Initialize ConfigEntsoe with specified or default parameters."""
        self.country_code = country_code

        # Default periods use deliberate n_periods choices:
        # - daily: n_periods=12 for 24 hours (2:1 ratio) provides 2-hour resolution,
        #   balancing detail vs overfitting while reducing dimensionality by 50%
        # - weekly/monthly/quarterly: n_periods matches range_size (1:1 ratio)
        # - yearly: n_periods=12 for 365 days (30:1 ratio) provides strong smoothing
        # See docs/PERIOD_CONFIGURATION_RATIONALE.md for detailed analysis
        self.periods = (
            periods
            if periods is not None
            else [
                Period(name="daily", n_periods=12, column="hour", input_range=(1, 24)),
                Period(
                    name="weekly", n_periods=7, column="dayofweek", input_range=(0, 6)
                ),
                Period(
                    name="monthly", n_periods=12, column="month", input_range=(1, 12)
                ),
                Period(
                    name="quarterly", n_periods=4, column="quarter", input_range=(1, 4)
                ),
                Period(
                    name="yearly",
                    n_periods=12,
                    column="dayofyear",
                    input_range=(1, 365),
                ),
            ]
        )
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
        if poly_features_degree < 1:
            raise ValueError(
                f"poly_features_degree must be >= 1, got {poly_features_degree}."
            )
        self.poly_features_degree = poly_features_degree
        self.max_poly_features = max_poly_features
        # Data source and index
        self.index_name = index_name
        self.start_download = start_download
        self.end_download = end_download
        # Derived date ranges (set after data loading via get_start_end())
        self.data_start = data_start
        self.data_end = data_end
        self.cov_start = cov_start
        self.cov_end = cov_end
        # Per-column outlier bounds
        self.bounds = bounds
        # Verbosity and caching
        self.verbose = verbose
        self.cache_home = cache_home
        # Derived training window
        self.end_train_ts = end_train_ts
        self.start_train_ts = start_train_ts
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
        # Maximum age of a previously trained model before retraining is
        # required.  Consumed by
        # ``spotforecast2_safe.manager.trainer.should_retrain``.
        self.retrain_max_age = (
            retrain_max_age
            if retrain_max_age is not None
            else pd.Timedelta(days=7)
        )

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
        params = {
            "country_code": self.country_code,
            "periods": self.periods,
            "lags_consider": self.lags_consider,
            "train_size": self.train_size,
            "end_train_default": self.end_train_default,
            "delta_val": self.delta_val,
            "predict_size": self.predict_size,
            "cv_block_size": self.cv_block_size,
            "refit_size": self.refit_size,
            "random_state": self.random_state,
            "n_hyperparameters_trials": self.n_hyperparameters_trials,
            "data_filename": self.data_filename,
            "targets": self.targets,
            # Outlier detection
            "use_outlier_detection": self.use_outlier_detection,
            "contamination": self.contamination,
            # Imputation
            "imputation_method": self.imputation_method,
            "window_size": self.window_size,
            "imputation_window_size": self.imputation_window_size,
            # Exogenous features
            "use_exogenous_features": self.use_exogenous_features,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "state": self.state,
            # Feature selection toggles
            "include_weather_windows": self.include_weather_windows,
            "include_holiday_features": self.include_holiday_features,
            "poly_features_degree": self.poly_features_degree,
            "max_poly_features": self.max_poly_features,
            # Data source and index
            "index_name": self.index_name,
            "start_download": self.start_download,
            "end_download": self.end_download,
            # Derived date ranges
            "data_start": self.data_start,
            "data_end": self.data_end,
            "cov_start": self.cov_start,
            "cov_end": self.cov_end,
            # Per-column outlier bounds
            "bounds": self.bounds,
            # Verbosity and caching
            "verbose": self.verbose,
            "cache_home": self.cache_home,
            # Derived training window
            "end_train_ts": self.end_train_ts,
            "start_train_ts": self.start_train_ts,
            # Hyperparameter tuning trial budgets
            "n_trials_optuna": self.n_trials_optuna,
            "n_trials_spotoptim": self.n_trials_spotoptim,
            "n_initial_spotoptim": self.n_initial_spotoptim,
            "n_jobs_spotoptim": self.n_jobs_spotoptim,
            "warm_start_lags": self.warm_start_lags,
            # Active task
            "task": self.task,
            # Aggregation weights
            "agg_weights": self.agg_weights,
            # Optional hooks
            "forecaster_factory": self.forecaster_factory,
            "data_loader": self.data_loader,
            "test_data_loader": self.test_data_loader,
            # Persistence policy and active-dataset identifier
            "auto_save_models": self.auto_save_models,
            "data_frame_name": self.data_frame_name,
            # Cross-validation fold count
            "number_folds": self.number_folds,
            # Weather-fetch failure policy
            "on_weather_failure": self.on_weather_failure,
            # Retraining cadence
            "retrain_max_age": self.retrain_max_age,
        }

        # Expose period sub-objects via the '__' notation if deep=True
        if deep and self.periods is not None:
            for period in self.periods:
                prefix = f"periods__{period.name}"
                params[f"{prefix}__n_periods"] = period.n_periods
                params[f"{prefix}__column"] = period.column
                params[f"{prefix}__input_range"] = period.input_range

        return params

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
        # Merge params dict and kwargs
        all_params: Dict[str, object] = {}
        if params is not None:
            all_params.update(params)
        all_params.update(kwargs)

        if not all_params:
            return self

        nested_period_params = {}
        flat_params = {}

        for key, value in all_params.items():
            if key.startswith("periods__"):
                parts = key.split("__")
                if len(parts) == 3:
                    _, p_name, p_param = parts
                    if p_name not in nested_period_params:
                        nested_period_params[p_name] = {}
                    nested_period_params[p_name][p_param] = value
                else:
                    raise ValueError(
                        f"Invalid deep parameter format: {key}. "
                        "Expected format: periods__<name>__<param>"
                    )
            else:
                flat_params[key] = value

        # Set standard parameters first
        for key, value in flat_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for {self.__class__.__name__}. "
                    "Check the list of available parameters with `get_params()`."
                )

        # Apply nested parameters to frozen Period dataclasses
        if nested_period_params and self.periods is not None:
            existing_names = {p.name for p in self.periods}
            for p_name in nested_period_params:
                if p_name not in existing_names:
                    raise ValueError(
                        f"Period with name '{p_name}' not found in configuration."
                    )

            new_periods = []
            for period in self.periods:
                if period.name in nested_period_params:
                    # Period is a frozen dataclass, so we utilize replace() to replicate
                    # an updated version.
                    updated_period = replace(
                        period, **nested_period_params[period.name]
                    )
                    new_periods.append(updated_period)
                else:
                    new_periods.append(period)
            self.periods = new_periods

        return self

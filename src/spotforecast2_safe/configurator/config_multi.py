# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration for multi-input task pipeline."""

from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from spotforecast2_safe.configurator._base_config import (
    apply_set_params,
    build_get_params,
    default_periods,
    validate_config,
)
from spotforecast2_safe.data import Period


class ConfigMulti:
    """Configuration for the multi-input forecasting pipeline.

    This class manages all configuration parameters for the multi-input task,
    including training/prediction intervals, data sources, and feature
    engineering specifications. All parameters can be customized during
    initialization or used with sensible defaults.

    ``country_code`` serves as the single ISO country code used for both
    API queries and holiday feature generation.

    Args:
        country_code (str): ISO 3166-1 alpha-2 country code (e.g. ``"DE"``).
            Used for both API queries and holiday feature generation.
        periods (Optional[List[Period]]): List of Period objects defining cyclical feature encodings.
        lags_consider (Optional[List[int]]): List of lag values to consider for feature selection.
        train_size (Optional[pd.Timedelta]): Time window for training data.
        end_train_default (str): Default end date for training period (ISO format with timezone).
        delta_val (Optional[pd.Timedelta]): Validation window size.
        predict_size (int): Number of hours to predict ahead.
        cv_block_size (int | None): Cross-validation test-block width in hours.
            Defaults to ``None``, meaning the CV uses ``predict_size``. Set to a
            fixed value (e.g. ``24``) to decouple the cross-validation horizon
            from a render-dependent live ``predict_size``.
        refit_size (int): Number of days between model refits.
        random_state (int): Random seed for reproducibility.
        n_hyperparameters_trials (int): Number of trials for hyperparameter optimization.
        data_filename (str): Path to the interim merged data file.
        targets (Optional[List[str]]): List of target column names to train models for.
            When ``None`` (default), no targets are pre-selected; set this attribute
            after loading the dataset (e.g. ``config.targets = df.columns.tolist()``).
            Replaces standalone ``TARGETS`` and ``target_columns`` variables in
            pipeline scripts, providing a single source of truth for the active
            target set.
        use_outlier_detection (bool): If True, apply IsolationForest-based outlier removal.
        contamination (float): Proportion of outliers for IsolationForest (0 < contamination < 0.5).
        imputation_method (str): Gap-filling strategy — ``"weighted"`` (n2n-style rolling weights)
            or ``"linear"`` (linear interpolation).
        window_size (int): Rolling window size in hours for gap detection (weighted imputation).
        use_exogenous_features (bool): If True, build weather/calendar/day-night/holiday features.
        latitude (float): Latitude of the target location in decimal degrees.
        longitude (float): Longitude of the target location in decimal degrees.
        timezone (str): IANA timezone string for the target location (e.g. ``"Europe/Berlin"``).
        state (str): ISO 3166-2 subdivision code for regional holidays (e.g. ``"NW"``).
        include_weather_windows (bool): If True, include rolling weather-window features.
        include_holiday_features (bool): If True, include public-holiday indicator features.
        include_holiday_adjacency_features (bool): If True, include Brückentag and
            before/after-holiday indicators (``is_brueckentag``, ``is_before_holiday``,
            ``is_after_holiday``).  Defaults to ``False``.
        poly_features_degree (int): Polynomial-interaction degree. ``1`` (default)
            generates no interactions; ``2`` adds pairwise bilinear terms; ``3+``
            higher order.
        max_poly_features (int): Cap on polynomial interaction columns; only the
            top ``max_poly_features`` ranked by mutual information with the target
            are kept (``<= 0`` disables). Defaults to ``10``.
        poly_mi_n_jobs (Optional[int]): Parallel jobs for the mutual-information
            ranking that enforces ``max_poly_features``. ``-1`` (default) uses
            all cores; ``None`` runs single-threaded. Parallelism does not
            change the selection.
        poly_mi_sample_size (Optional[int]): Row cap for that ranking; longer
            series are scored on a reproducible random subsample of this size
            (seeded by ``random_state``), which can change which borderline
            columns make the top K. ``None`` scores every row (the pre-15.8
            behaviour). Defaults to ``4000``.
        index_name (str): Name assigned to the datetime column when the index is reset.
            Defaults to ``"DateTime"``.
        start_download (Optional[str]): Start of the download/data range as a string
            (format ``"YYYYMMDDHHMM"``). Derived from the loaded dataset; ``None`` until set.
        end_download (Optional[str]): End of the download/data range as a string
            (format ``"YYYYMMDDHHMM"``). Derived from the loaded dataset; ``None`` until set.
        data_start (Optional[pd.Timestamp]): First timestamp of the pipeline data range.
            Derived from the loaded dataset via ``get_start_end()``; ``None`` until set.
        data_end (Optional[pd.Timestamp]): Last timestamp of the pipeline data range.
            Derived from the loaded dataset via ``get_start_end()``; ``None`` until set.
        cov_start (Optional[pd.Timestamp]): Start of the covariate range (same as ``data_start``).
            Derived from the loaded dataset via ``get_start_end()``; ``None`` until set.
        cov_end (Optional[pd.Timestamp]): End of the covariate range (extends ``data_end`` by
            ``predict_size`` hours). Derived via ``get_start_end()``; ``None`` until set.
        bounds (Optional[List[tuple]]): Per-column outlier bounds as a list of
            ``(lower, upper)`` tuples, one entry per target column. ``None`` until set.
        verbose (bool): If ``True``, enable verbose output for pipeline steps.
            Defaults to ``False``.
        cache_home (Optional[Any]): Path to the cache directory. ``None`` means
            the library default (``~/spotforecast2_cache/``) is used.
        end_train_ts (Optional[pd.Timestamp]): End of the training window.
            Derived from ``end_train_default`` after data loading; ``None`` until set.
        start_train_ts (Optional[pd.Timestamp]): Start of the training window.
            Derived as ``end_train_ts - train_size`` after data loading; ``None`` until set.
        n_trials_optuna (int): Number of Optuna Bayesian-search trials for hyperparameter
            optimization (task 3). Defaults to ``15``.
        n_trials_spotoptim (int): Number of SpotOptim surrogate-search trials (task 4).
            Defaults to ``10``.
        n_initial_spotoptim (int): Number of initial random evaluations for SpotOptim
            (task 4). Defaults to ``5``.
        n_jobs_spotoptim (Optional[int]): Worker count for SpotOptim's parallel
            (steady-state) evaluation. ``None`` (default) runs sequentially;
            ``-1`` uses all CPU cores; a positive integer pins the worker count.
            Parallel tuning is faster but, being steady-state, changes the search
            trajectory, so the tuned result is not bit-identical to a sequential
            run even with a fixed ``random_state``.
        warm_start_lags (bool): When True, the SpotOptim task injects
            ``lags_consider`` as a candidate lag set and seeds the optimizer's
            first evaluation with it. Defaults to ``False``.
        task (str): Active prediction task — one of ``"lazy"``, ``"training"``,
            ``"optuna"``, or ``"spotoptim"``. Defaults to ``"lazy"``.
        agg_weights (Optional[List[float]]): Per-target aggregation weights used
            when combining individual target forecasts into a single weighted sum.
            The list must contain one weight per entry in ``targets`` (in the same
            order). Positive values add the target's contribution; negative values
            invert it. Slice the list to ``agg_weights[:len(targets)]`` when only
            a subset of targets is active. Defaults to ``None`` (no weights
            pre-defined; set after loading the dataset).
        auto_save_models (bool): Whether ``BaseTask._run_strategy`` should
            persist fitted forecasters to ``<cache_home>/models/`` after every
            training run.  Defaults to ``True`` so that saved models are
            immediately available for ``PredictTask`` without an explicit
            ``save_models()`` call.
        data_frame_name (str): Identifier for the active dataset.  Used by
            ``BaseTask`` to name cache subdirectories, model files, and the
            per-dataset log file.  Defaults to ``"default"``.
        on_weather_failure (Literal["raise", "skip"]): Policy for handling
            Open-Meteo fetch failures inside
            ``BaseTask.build_exogenous_features``.  ``"raise"`` (default)
            aborts the pipeline with a ``WeatherFetchError`` and preserves
            the safety-critical fail-safe semantics.  ``"skip"`` logs a
            warning and continues with empty weather features so the rest
            of the pipeline can run without the Open-Meteo dependency.
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

    Attributes:
        country_code (str): ISO country code for API queries and holiday generation.
        periods (List[Period]): Cyclical feature encoding specifications.
        lags_consider (List[int]): Lag values for autoregressive features.
        train_size (pd.Timedelta): Training data window.
        end_train_default (str): Default training end date.
        delta_val (pd.Timedelta): Validation window.
        predict_size (int): Prediction horizon in hours.
        refit_size (int): Refit interval in days.
        random_state (int): Random seed.
        n_hyperparameters_trials (int): Hyperparameter tuning trials.
        targets (Optional[List[str]]): Active target column names. ``None`` until
            explicitly set from the loaded dataset.
        use_outlier_detection (bool): IsolationForest outlier removal toggle.
        contamination (float): IsolationForest contamination fraction.
        imputation_method (str): Gap-filling strategy (``"weighted"`` or ``"linear"``).
        window_size (int): Rolling window size for weighted imputation.
        use_exogenous_features (bool): Exogenous feature construction toggle.
        latitude (float): Location latitude.
        longitude (float): Location longitude.
        timezone (str): IANA timezone string.
        state (str): Subdivision code for regional holidays.
        include_weather_windows (bool): Weather-window feature toggle.
        include_holiday_features (bool): Holiday feature toggle.
        include_holiday_adjacency_features (bool): Brückentag and
            before/after-holiday indicator toggle.  Defaults to ``False``.
        poly_features_degree (int): Polynomial-interaction degree (1 = off).
        max_poly_features (int): Cap on kept ``poly_*`` columns (top-K by MI).
        poly_mi_n_jobs (Optional[int]): Parallel jobs for the MI ranking
            (``-1`` = all cores; selection-invariant).
        poly_mi_sample_size (Optional[int]): Row cap for the MI ranking
            (``None`` = score every row).
        include_covid_infection_rate (bool): Append the bundled RKI German
            national COVID-19 7-day incidence as an exogenous regressor.
        include_entsoe_forecast_load (bool): Append the ENTSO-E day-ahead
            Forecasted Load as a near-oracle exogenous prior.
        include_entsoe_renewable_forecast (bool): Append the ENTSO-E day-ahead
            wind/solar generation forecast.
        include_entsoe_net_load (bool): Append the ENTSO-E day-ahead net load
            (Forecasted Load minus wind/solar forecast).
        include_entsoe_day_ahead_price (bool): Append the ENTSO-E day-ahead
            spot price (DE/LU).
        index_name (str): Datetime column name used when resetting the index.
        start_download (Optional[str]): Start of the data download range.
        end_download (Optional[str]): End of the data download range.
        data_start (Optional[pd.Timestamp]): First timestamp of the pipeline data.
        data_end (Optional[pd.Timestamp]): Last timestamp of the pipeline data.
        cov_start (Optional[pd.Timestamp]): Start of the covariate date range.
        cov_end (Optional[pd.Timestamp]): End of the covariate date range.
        bounds (Optional[List[tuple]]): Per-column outlier bounds ``(lower, upper)``.
        verbose (bool): Verbose output toggle.
        cache_home (Optional[Any]): Path to the cache directory.
        end_train_ts (Optional[pd.Timestamp]): End of the training window.
        start_train_ts (Optional[pd.Timestamp]): Start of the training window.
        n_trials_optuna (int): Number of Optuna hyperparameter-search trials.
        n_trials_spotoptim (int): Number of SpotOptim search trials.
        n_initial_spotoptim (int): Number of initial SpotOptim evaluations.
        n_jobs_spotoptim (Optional[int]): Worker count for SpotOptim's parallel
            (steady-state) evaluation (``None`` = sequential, ``-1`` = all cores).
        warm_start_lags (bool): Seed the SpotOptim search with ``lags_consider``.
        task (str): Active prediction task (``"lazy"``, ``"training"``,
            ``"optuna"``, or ``"spotoptim"``).
        agg_weights (Optional[List[float]]): Per-target aggregation weights.
            One weight per entry in ``targets``; positive values add, negative
            values invert the target's contribution. ``None`` until set.
        auto_save_models (bool): Whether to auto-persist fitted forecasters
            after each training run.
        data_frame_name (str): Active-dataset identifier used for cache and
            log-file naming.
        number_folds (int): Cross-validation fold count for tuning tasks.
        on_weather_failure (Literal["raise", "skip"]): Open-Meteo fetch-failure
            policy: ``"raise"`` aborts, ``"skip"`` continues without weather.
        on_exog_provider_failure (Literal["raise", "skip"]): Exog-provider
            failure policy in ``ExogBuilder.build``: ``"raise"`` (default)
            propagates the ``ExogProviderError``; ``"skip"`` logs and omits the
            failing provider's columns.
        exog_max_gap_hours (int): Maximum contiguous gap in hours that providers
            will heal before raising (0 = strict fail-safe).
        exog_provider_window (Literal["full", "train"]): Validation window for
            exog providers: ``"full"`` (default) or ``"train"``.

    Notes:
        The default period configurations use specific `n_periods` to balance resolution and smoothing:
        - **Daily**: `n_periods=12` (24h) provides ~2h resolution, smoothing hourly noise and halving dimensionality.
        - **Weekly**: `n_periods` typically matches range (1:1) to distinguish day-of-week patterns.
        - **Yearly**: `n_periods=12` (365d) provides ~1 month resolution, capturing broad seasonal trends without overfitting.

        See `docs/PERIOD_CONFIGURATION_RATIONALE.md` for a detailed analysis.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.configurator.config_multi import ConfigMulti
        config = ConfigMulti()
        print(f"country_code: {config.country_code}")
        print(f"Predict size: {config.predict_size}")
        print(f"Random state: {config.random_state}")
        print(f"Targets (default): {config.targets}")
        print(f"agg_weights (default): {config.agg_weights}")
        print(f"index_name: {config.index_name}")
        print(f"start_download: {config.start_download}")
        print(f"end_download: {config.end_download}")
        print(f"data_start: {config.data_start}")
        print(f"data_end: {config.data_end}")
        print(f"cov_start: {config.cov_start}")
        print(f"cov_end: {config.cov_end}")
        print(f"bounds: {config.bounds}")

        # Set targets and derived ranges after loading data
        config.targets = ["A", "B", "C"]
        config.start_download = "202401010000"
        config.end_download = "202412312300"
        config.data_start = pd.Timestamp("2022-01-01", tz="UTC")
        config.data_end = pd.Timestamp("2024-12-31", tz="UTC")
        config.cov_start = pd.Timestamp("2022-01-01", tz="UTC")
        config.cov_end = pd.Timestamp("2025-01-01", tz="UTC")
        config.bounds = [(-2500, 4500), (-10, 3000)]
        print(f"Targets (after setting): {config.targets}")
        print(f"start_download: {config.start_download}")
        print(f"data_start: {config.data_start}")
        print(f"bounds: {config.bounds}")

        # Create custom configuration — country_code serves both API and holiday purposes
        custom_config = ConfigMulti(
            country_code='FR',
            predict_size=48,
            random_state=42,
            targets=["A", "B"],
            index_name="DateTime",
        )
        print(f"country_code: {custom_config.country_code}")
        print(f"Predict size: {custom_config.predict_size}")
        print(f"Random state: {custom_config.random_state}")
        print(f"Targets: {custom_config.targets}")

        # Verify training window
        print(f"Training window: {config.train_size == pd.Timedelta(days=3 * 365)}")

        # Check default periods
        print(f"Number of periods: {len(config.periods)}")
        print(f"First period name: {config.periods[0].name}")
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
        "start_download",
        "end_download",
        "data_start",
        "data_end",
        "cov_start",
        "cov_end",
        "bounds",
        "verbose",
        "cache_home",
        "end_train_ts",
        "start_train_ts",
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
        index_name: str = "DateTime",
        start_download: Optional[str] = None,
        end_download: Optional[str] = None,
        # Derived date ranges (set after data loading via get_start_end())
        data_start: Optional[pd.Timestamp] = None,
        data_end: Optional[pd.Timestamp] = None,
        cov_start: Optional[pd.Timestamp] = None,
        cov_end: Optional[pd.Timestamp] = None,
        # Per-column outlier bounds [(lower, upper), ...]
        bounds: Optional[List[tuple]] = None,
        # Verbosity and caching
        verbose: bool = False,
        cache_home: Optional[Any] = None,
        # Derived training window (set after data loading)
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
        # Aggregation weights (one per target, in target order)
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
    ):
        """Initialize ConfigMulti with specified or default parameters."""
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
        # Cross-validation test-block width (hours). ``None`` defers to
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
        self.start_download = start_download
        self.end_download = end_download
        # Derived date ranges (set after data loading via get_start_end())
        self.data_start = data_start
        self.data_end = data_end
        self.cov_start = cov_start
        self.cov_end = cov_end
        # Per-column outlier bounds [(lower, upper), ...]
        self.bounds = bounds
        # Verbosity and caching
        self.verbose = verbose
        self.cache_home = cache_home
        # Derived training window (set after data loading)
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
        # Aggregation weights (one per target, in target order)
        self.agg_weights = agg_weights
        # Optional callable consumed by ``spotforecast2.multitask.base``:
        # ``factory(config, *, weight_func, target) -> forecaster``.  When
        # ``None``, ``BaseTask.create_forecaster`` falls back to
        # ``default_lgbm_forecaster_factory``.
        self.forecaster_factory = forecaster_factory
        # Optional callable consumed by ``BaseTask.prepare_data``:
        # ``data_loader(config) -> pd.DataFrame``.  When set, it is invoked iff
        # no DataFrame is supplied via the constructor or ``prepare_data``.
        # Enables data-acquisition extensions (e.g. the ENTSO-E API download)
        # without subclassing ``BaseTask``.
        self.data_loader = data_loader
        # Optional callable consumed by ``BaseTask.prepare_data``:
        # ``test_data_loader(config) -> pd.DataFrame``.  Mirrors ``data_loader``
        # for the test/ground-truth slice — invoked iff no test DataFrame is
        # supplied via the constructor or ``prepare_data``.  When set, the
        # returned frame is used to populate ``test_actual`` and
        # ``metrics_future`` in the prediction package.
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
        # Policy for exog-provider failures consumed by ``ExogBuilder``:
        # ``"raise"`` aborts (default, fail-safe); ``"skip"`` logs a warning
        # and omits the failing provider's columns.
        self.on_exog_provider_failure = on_exog_provider_failure
        # Maximum contiguous gap in hours that providers will heal (0 = strict).
        self.exog_max_gap_hours = exog_max_gap_hours
        # Extended trailing-edge healing budget (0 = same as exog_max_gap_hours).
        self.exog_max_tail_gap_hours = exog_max_tail_gap_hours
        # Validation window for providers: "full" (default) or "train".
        self.exog_provider_window = exog_provider_window
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
            from spotforecast2_safe.configurator.config_multi import ConfigMulti
            config = ConfigMulti(country_code="FR")
            p = config.get_params()
            print(f"country_code: {p['country_code']}")
            print(f"Predict size: {p['predict_size']}")
            print(f"Random state: {p['random_state']}")
            print(f"index_name: {p['index_name']}")
            print(f"data_start: {p['data_start']}")
            print(f"data_end: {p['data_end']}")
            print(f"cov_start: {p['cov_start']}")
            print(f"cov_end: {p['cov_end']}")
            print(f"bounds: {p['bounds']}")
            print(f"agg_weights: {p['agg_weights']}")
            ```
        """
        return build_get_params(self, self._PARAM_NAMES, deep)

    def set_params(
        self, params: Dict[str, object] = None, **kwargs: object
    ) -> "ConfigMulti":
        """
        Set the parameters of this configuration object.

        Args:
            params: Optional dictionary of parameter names mapped to their
                new values.
            **kwargs: Additional parameter names mapped to their new values.
                It supports configuring nested 'Period' objects using the
                `periods__<name>__<param>` notation.

        Returns:
            ConfigMulti: The configuration instance with updated
                parameters (supports method chaining).

        Examples:
            ```{python}
            from spotforecast2_safe.configurator.config_multi import ConfigMulti
            config = ConfigMulti()
            _ = config.set_params(country_code="FR", predict_size=48)
            print(f"country_code: {config.country_code}")
            print(f"Predict size: {config.predict_size}")
            print(f"Random state: {config.random_state}")

            # Set derived download range after loading data
            _ = config.set_params(start_download="202401010000", end_download="202412312300")
            print(f"start_download: {config.start_download}")

            # Deep parameter setting
            _ = config.set_params(periods__daily__n_periods=24)
            print(next(p.n_periods for p in config.periods if p.name == "daily"))
            ```
        """
        return apply_set_params(self, params, **kwargs)

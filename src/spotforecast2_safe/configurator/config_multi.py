# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration for multi-input task pipeline."""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from spotforecast2_safe.configurator._base_config import (
    apply_set_params,
    build_get_params,
    default_periods,
    validate_config,
)
from spotforecast2_safe.data import Period

# Default seed lag set for the SpotOptim warm start: short-range (1-3 h),
# around-daily (23-25 h), two-day (47-48 h), around-weekly (167-169 h), and
# two-week (336 h) structure of hourly load series.
DEFAULT_WARM_START_LAGS: List[int] = [
    1,
    2,
    3,
    23,
    24,
    25,
    47,
    48,
    167,
    168,
    169,
    336,
]


@dataclass
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
        include_ephemeris_features (bool): If True, include solar-elevation and
            daylight-duration features.  Defaults to ``False``.
        include_day_type_features (bool): If True, include working-day and day-type
            class features (``is_workday``, ``day_type``).  Defaults to ``False``.
        include_school_holiday_features (bool): Append the ``is_school_holiday``
            binary indicator from the bundled OpenHolidays API dataset (ODbL-1.0).
            Coverage 2022-01-01 to 2027-12-31 for all 16 German Bundesländer.
            Only ``country_code="DE"`` is supported.  Defaults to ``False``.
        per_zone_weather (bool): When True, each target is treated as a German
            TSO control zone and receives weather from its own regional cities
            via ``weather.locations.locations_for_zone``.  Mutually exclusive
            with ``use_population_weighted_weather``; requires
            ``use_exogenous_features=True``; not compatible with
            ``poly_features_degree >= 2``.  Default ``False`` → byte-identical
            to the shared-weather baseline.
        zone_weather_locations (Optional[Dict[str, Any]]): Optional override
            mapping from zone key (e.g. ``"load_50hertz"``) to a list of
            ``WeatherLocation`` objects.  ``None`` (default) uses the built-in
            registry partition from ``GERMAN_TSO_ZONE_CITIES``.
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
        bounds (Optional[List[tuple]]): Per-column outlier bounds as a list of
            ``(lower, upper)`` tuples, one entry per target column. ``None`` until set.
        verbose (bool): If ``True``, enable verbose output for pipeline steps.
            Defaults to ``False``.
        cache_home (Optional[Any]): Path to the cache directory. ``None`` means
            the library default (``~/spotforecast2_cache/``) is used.
        n_trials_optuna (int): Number of Optuna Bayesian-search trials for hyperparameter
            optimization (task 3). Defaults to ``15``.
        n_trials_spotoptim (int): Number of SpotOptim surrogate-search trials (task 4).
            Defaults to ``10``.
        n_initial_spotoptim (int): Number of initial random evaluations for SpotOptim
            (task 4). Defaults to ``5``.
        max_time_spotoptim (Optional[float]): Wall-clock budget for the SpotOptim
            search in minutes (task 4). The search stops when either
            ``n_trials_spotoptim`` evaluations or this time limit is reached,
            whichever comes first. ``None`` (the default) disables the limit.
        warm_start_lags (Optional[List[int]]): Lag set the SpotOptim task
            injects as a search-space candidate and uses to seed the
            optimizer's first evaluation. Defaults to
            ``DEFAULT_WARM_START_LAGS``
            (``[1, 2, 3, 23, 24, 25, 47, 48, 167, 168, 169, 336]``).
            ``None`` or an empty list disables the warm start.
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
        include_ephemeris_features (bool): Solar-elevation and daylight-duration
            feature toggle.  Defaults to ``False``.
        include_day_type_features (bool): Working-day / day-type class feature
            toggle.  Defaults to ``False``.
        include_school_holiday_features (bool): Per-Bundesland school-holiday
            indicator toggle.  Defaults to ``False``.
        per_zone_weather (bool): When True, each target is a TSO control zone
            that fetches its own regional weather via
            ``weather.locations.locations_for_zone``.  Mutually exclusive with
            ``use_population_weighted_weather``; requires
            ``use_exogenous_features=True``; not compatible with
            ``poly_features_degree >= 2``.  Default ``False``.
        zone_weather_locations (Optional[Dict[str, Any]]): Override mapping
            from zone key to a list of ``WeatherLocation`` objects.  ``None``
            uses the built-in ``GERMAN_TSO_ZONE_CITIES`` partition.
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
        include_football_match_window (bool): Append the bundled German
            football-match event-window feature (1.0 during configured
            match windows, 0.0 otherwise). Covers German national-team
            matches and tournament finals from UEFA Euro 2016 through
            FIFA World Cup 2026.
        include_energy_saving_window (bool): Append the bundled German
            energy-saving regulatory window feature (1.0 during the
            EnSikuMaV and EU Regulation 2022/1854 periods, 0.0 otherwise).
        index_name (str): Datetime column name used when resetting the index.
        bounds (Optional[List[tuple]]): Per-column outlier bounds ``(lower, upper)``.
        verbose (bool): Verbose output toggle.
        cache_home (Optional[Any]): Path to the cache directory.
        n_trials_optuna (int): Number of Optuna hyperparameter-search trials.
        n_trials_spotoptim (int): Number of SpotOptim search trials.
        n_initial_spotoptim (int): Number of initial SpotOptim evaluations.
        max_time_spotoptim (Optional[float]): Wall-clock budget for the SpotOptim
            search in minutes; ``None`` disables the limit.
        warm_start_lags (Optional[List[int]]): Seed lag set for the SpotOptim
            search; ``None`` or empty disables the warm start.
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
        print(f"bounds: {config.bounds}")

        # Set targets and bounds (user input that stays on the config)
        config.targets = ["A", "B", "C"]
        config.bounds = [(-2500, 4500), (-10, 3000)]
        print(f"Targets (after setting): {config.targets}")
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

    country_code: str = "DE"
    periods: List[Period] = field(default_factory=default_periods)
    lags_consider: List[int] = field(default_factory=lambda: list(range(1, 24)))
    train_size: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(days=3 * 365))
    end_train_default: str = "2025-12-31 00:00+00:00"
    delta_val: pd.Timedelta = field(
        default_factory=lambda: pd.Timedelta(hours=24 * 7 * 10)
    )
    predict_size: int = 24
    # Cross-validation test-block width (hours). ``None`` defers to
    # ``predict_size``; the actual CV-split logic lives in the sibling
    # ``spotforecast2`` package (``BaseTask.cv_ts``).
    cv_block_size: Optional[int] = None
    refit_size: int = 7
    random_state: int = 314159
    n_hyperparameters_trials: int = 20
    data_filename: str = "interim/energy_load.csv"
    targets: Optional[List[str]] = None
    # Outlier detection
    use_outlier_detection: bool = True
    contamination: float = 0.01
    # Imputation
    imputation_method: str = "weighted"
    window_size: int = 72
    imputation_window_size: Optional[int] = None
    # Exogenous features
    use_exogenous_features: bool = True
    latitude: float = 51.5136
    longitude: float = 7.4653
    timezone: str = "UTC"
    state: str = "NW"
    # Feature selection toggles
    include_weather_windows: bool = False
    include_holiday_features: bool = False
    include_holiday_adjacency_features: bool = False
    # Global / population-weighted weather and derived weather features
    # (consumed by spotforecast2.multitask.base.build_exogenous_features via
    # spotforecast2_safe.weather.get_weather_features). All default off so the
    # pipeline stays byte-identical to the single-point baseline.
    # ``use_population_weighted_weather`` fetches the fixed German load-centre
    # registry (spotforecast2_safe.weather.locations) and combines the cities
    # by population weight instead of sampling the single latitude/longitude.
    # ``include_degree_hours`` adds heating/cooling degree-hours (hdh/cdh) and
    # ``include_apparent_temperature`` adds apparent temperature + dew point.
    use_population_weighted_weather: bool = False
    # Per-zone weather: when True, each target is resolved as a TSO control
    # zone and fetches weather from its own regional cities (via
    # spotforecast2_safe.weather.locations.locations_for_zone). Mutually
    # exclusive with use_population_weighted_weather; requires
    # use_exogenous_features=True; not compatible with
    # poly_features_degree>=2 (polynomial interactions are precomputed from
    # the shared weather frame). Default OFF → byte-identical to today.
    per_zone_weather: bool = False
    # Optional override mapping zone key → list of WeatherLocation objects;
    # None uses the built-in registry partition (GERMAN_TSO_ZONE_CITIES).
    # Not a mutable default — the field is None, not [].
    zone_weather_locations: Optional[Dict[str, Any]] = None
    include_degree_hours: bool = False
    include_apparent_temperature: bool = False
    degree_hours_base_heating: float = 15.0
    degree_hours_base_cooling: float = 22.0
    # Ephemeris (continuous solar geometry) and day-type calendar refinements
    # (consumed by spotforecast2.multitask.base.build_exogenous_features). Both
    # default off so the pipeline stays byte-identical to the baseline.
    # ``include_ephemeris_features`` adds solar_elevation + daylight_duration_h
    # + signed sunrise/sunset-relative time; ``include_day_type_features`` adds
    # is_workday + day_type (workday/Saturday/Sunday/holiday class).
    include_ephemeris_features: bool = False
    include_day_type_features: bool = False
    include_school_holiday_features: bool = False
    poly_features_degree: int = 1
    max_poly_features: int = 10
    poly_mi_n_jobs: Optional[int] = -1
    poly_mi_sample_size: Optional[int] = 4000
    # Provider-based exogenous toggles, each gated by a registry flag in
    # ``spotforecast2_safe.preprocessing.exog_providers``.
    include_covid_infection_rate: bool = False
    include_entsoe_forecast_load: bool = False
    include_entsoe_renewable_forecast: bool = False
    include_entsoe_net_load: bool = False
    include_entsoe_day_ahead_price: bool = False
    include_football_match_window: bool = False
    include_energy_saving_window: bool = False
    # Data source and index
    index_name: str = "DateTime"
    # Per-column outlier bounds [(lower, upper), ...]
    bounds: Optional[List[tuple]] = None
    # Verbosity and caching
    verbose: bool = False
    cache_home: Optional[Any] = None
    # Hyperparameter tuning trial budgets
    n_trials_optuna: int = 15
    n_trials_spotoptim: int = 10
    n_initial_spotoptim: int = 5
    # Wall-clock budget for the SpotOptim search in MINUTES (consumed by
    # spotforecast2.multitask.strategies.SpotOptimStrategy). ``None`` means no
    # time limit: the search runs until ``n_trials_spotoptim`` is exhausted.
    max_time_spotoptim: Optional[float] = None
    # Seed lag set for the SpotOptim search (consumed by
    # spotforecast2.multitask.strategies.SpotOptimStrategy). ``None`` or an
    # empty list disables the warm start.
    warm_start_lags: Optional[List[int]] = field(
        default_factory=lambda: list(DEFAULT_WARM_START_LAGS)
    )
    # Active task
    task: str = "lazy"
    # Aggregation weights (one per target, in target order)
    agg_weights: Optional[List[float]] = None
    # Forecaster factory hook (consumed by spotforecast2.multitask.base):
    # ``factory(config, *, weight_func, target) -> forecaster``. When ``None``,
    # ``BaseTask.create_forecaster`` falls back to
    # ``default_lgbm_forecaster_factory``.
    forecaster_factory: Optional[Any] = None
    # Data-loader hook (consumed by ``BaseTask.prepare_data``):
    # ``data_loader(config) -> pd.DataFrame``. Invoked iff no DataFrame is
    # supplied via the constructor or ``prepare_data``.
    data_loader: Optional[Any] = None
    # Test-data-loader hook (consumed by ``BaseTask.prepare_data``): mirrors
    # ``data_loader`` for the test/ground-truth slice.
    test_data_loader: Optional[Any] = None
    # Persistence policy and active-dataset name (consumed by
    # spotforecast2.multitask.base).
    auto_save_models: bool = True
    data_frame_name: str = "default"
    # Cross-validation fold count (consumed by spotforecast2.multitask.base.cv_ts)
    number_folds: int = 10
    # Weather-fetch failure policy (consumed by
    # spotforecast2.multitask.base.build_exogenous_features)
    on_weather_failure: Literal["raise", "skip"] = "raise"
    # Exog-provider failure policy (consumed by
    # preprocessing.exog_builder.ExogBuilder)
    on_exog_provider_failure: Literal["raise", "skip"] = "raise"
    # Gap-healing budget for exog providers (0 = strict fail-safe)
    exog_max_gap_hours: int = 0
    # Extended trailing-edge healing budget (0 = same as exog_max_gap_hours)
    exog_max_tail_gap_hours: int = 0
    # Validation window for exog providers ("full" or "train")
    exog_provider_window: Literal["full", "train"] = "full"
    # Target-side corruption detector knobs. Detector active only when
    # target_qc_window_days AND at least one of target_qc_range_mw /
    # target_qc_step_mw / target_qc_deviation_mw are set. Defaults are all
    # None / off, so the pipeline is byte-identical to the pre-feature baseline.
    # Recommended episode policy: "truncate" (auto-extends predict_size).
    # "heal" under the default anchor_zone_hours=168 with a <=7-day QC window
    # never engages (refusal by design). The deviation rule (dropout-only, vs a
    # published reference column such as "Forecasted Load") catches corruption
    # that stays below the dynamics thresholds; when enabling it, scope
    # ``targets`` to the actuals so heal/truncate leave the reference intact.
    target_qc_range_mw: Optional[float] = None
    target_qc_step_mw: Optional[float] = None
    target_qc_window_days: Optional[int] = None
    target_corruption_policy: str = "abort"
    target_max_heal_hours: int = 0
    target_anchor_zone_hours: int = 168
    target_qc_deviation_mw: Optional[float] = None
    target_qc_deviation_ref: Optional[str] = None
    target_qc_deviation_slots: int = 2

    def __post_init__(self) -> None:
        """Reject clearly-invalid hyperparameter values (fail-safe)."""
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
            print(f"bounds: {p['bounds']}")
            print(f"agg_weights: {p['agg_weights']}")
            ```
        """
        return build_get_params(self, [f.name for f in fields(self)], deep)

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

            # Deep parameter setting
            _ = config.set_params(periods__daily__n_periods=24)
            print(next(p.n_periods for p in config.periods if p.name == "daily"))
            ```
        """
        return apply_set_params(self, params, **kwargs)


# ``_PARAM_NAMES`` is derived from the dataclass fields (declaration order) so it
# can never drift from the actual fields; consumers and tests still read it as a
# class attribute. Set after the class body because ``fields()`` needs the
# finished dataclass.
ConfigMulti._PARAM_NAMES = tuple(f.name for f in fields(ConfigMulti))

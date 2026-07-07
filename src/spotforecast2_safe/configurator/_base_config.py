# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared configuration helpers for the pipeline config classes.

``ConfigEntsoe`` and ``ConfigMulti`` share the same cyclical-feature period
defaults and the same ``get_params`` / ``set_params`` semantics. Those were
previously hand-duplicated in both classes, which let fields drift between the
two copies. These helpers centralise the logic so each config only declares its
``@dataclass`` fields; ``_PARAM_NAMES`` is derived from
``dataclasses.fields()`` (so it can never drift from the fields), and the
``get_params`` / ``set_params`` behaviour and the period defaults live here.
"""

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from spotforecast2_safe.data import Period

_base_logger = logging.getLogger(__name__)


def default_periods() -> List[Period]:
    """Return the default cyclical-feature period specifications.

    The ``n_periods`` choices are deliberate: daily uses ``n_periods=12`` over 24
    hours (2-hour resolution, halving dimensionality); weekly/monthly/quarterly
    match their range size (1:1); yearly uses ``n_periods=12`` over 365 days for
    strong smoothing. See ``docs/PERIOD_CONFIGURATION_RATIONALE.md``.

    Examples:
        ```{python}
        from spotforecast2_safe.configurator._base_config import default_periods

        periods = default_periods()
        print(f"Number of default periods: {len(periods)}")
        names = [p.name for p in periods]
        print(f"Period names: {names}")
        assert len(periods) == 5
        assert names == ["daily", "weekly", "monthly", "quarterly", "yearly"]
        daily = periods[0]
        assert daily.n_periods == 12
        assert daily.column == "hour"
        assert daily.input_range == (1, 24)
        ```
    """
    return [
        Period(name="daily", n_periods=12, column="hour", input_range=(1, 24)),
        Period(name="weekly", n_periods=7, column="dayofweek", input_range=(0, 6)),
        Period(name="monthly", n_periods=12, column="month", input_range=(1, 12)),
        Period(name="quarterly", n_periods=4, column="quarter", input_range=(1, 4)),
        Period(name="yearly", n_periods=12, column="dayofyear", input_range=(1, 365)),
    ]


def build_get_params(
    config: object, param_names: Sequence[str], deep: bool = True
) -> Dict[str, object]:
    """Build the ``get_params`` dictionary for a config object.

    Args:
        config: The configuration instance to read attributes from.
        param_names: Ordered parameter names; the returned dict preserves this
            order, matching the historical hand-written ``get_params`` output.
        deep: If True, expose the nested ``Period`` sub-objects via the
            ``periods__<name>__<param>`` notation.

    Returns:
        Dictionary of parameter names mapped to their current values.

    Examples:
        ```{python}
        from spotforecast2_safe.configurator._base_config import build_get_params
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        config = ConfigMulti(country_code="DE")
        # Retrieve a subset of flat parameters
        params = build_get_params(config, ["predict_size", "random_state"], deep=False)
        print(f"predict_size: {params['predict_size']}")
        print(f"random_state: {params['random_state']}")
        assert params["predict_size"] == 24
        # With deep=True (default) the nested Period sub-objects are also exposed
        params_deep = build_get_params(config, ["predict_size"], deep=True)
        assert "periods__daily__n_periods" in params_deep
        print(f"periods__daily__n_periods: {params_deep['periods__daily__n_periods']}")
        ```
    """
    params: Dict[str, object] = {name: getattr(config, name) for name in param_names}

    periods = getattr(config, "periods", None)
    if deep and periods is not None:
        for period in periods:
            prefix = f"periods__{period.name}"
            params[f"{prefix}__n_periods"] = period.n_periods
            params[f"{prefix}__column"] = period.column
            params[f"{prefix}__input_range"] = period.input_range

    return params


def apply_set_params(
    config: object, params: Optional[Dict[str, object]] = None, **kwargs: object
):
    """Apply flat and nested parameter updates to a config object in place.

    Flat parameters are set via ``setattr`` (rejecting unknown names and
    read-only properties). Nested ``periods__<name>__<param>`` keys update the
    frozen ``Period`` dataclasses via `dataclasses.replace()`.

    Args:
        config: The configuration instance to mutate.
        params: Optional dictionary of parameter names mapped to new values.
        **kwargs: Additional parameter names mapped to new values.

    Returns:
        The same ``config`` instance (supports method chaining).

    Raises:
        ValueError: For an unknown flat parameter, a malformed ``periods__``
            key, or a period name not present in the configuration.

    Examples:
        ```{python}
        from spotforecast2_safe.configurator._base_config import apply_set_params
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        config = ConfigMulti(country_code="DE")
        print(f"predict_size before: {config.predict_size}")

        # Update a flat parameter
        apply_set_params(config, predict_size=48)
        print(f"predict_size after: {config.predict_size}")
        assert config.predict_size == 48

        # Update a nested Period field via the periods__<name>__<param> notation
        apply_set_params(config, **{"periods__daily__n_periods": 24})
        daily = next(p for p in config.periods if p.name == "daily")
        print(f"daily n_periods after: {daily.n_periods}")
        assert daily.n_periods == 24

        # apply_set_params returns the config instance for method chaining
        result = apply_set_params(config, random_state=42)
        assert result is config
        ```
    """
    all_params: Dict[str, object] = {}
    if params is not None:
        all_params.update(params)
    all_params.update(kwargs)

    if not all_params:
        return config

    nested_period_params: Dict[str, Dict[str, object]] = {}
    flat_params: Dict[str, object] = {}

    for key, value in all_params.items():
        if key.startswith("periods__"):
            parts = key.split("__")
            if len(parts) == 3:
                _, p_name, p_param = parts
                nested_period_params.setdefault(p_name, {})[p_param] = value
            else:
                raise ValueError(
                    f"Invalid deep parameter format: {key}. "
                    "Expected format: periods__<name>__<param>"
                )
        else:
            flat_params[key] = value

    # Set standard parameters first
    for key, value in flat_params.items():
        if hasattr(config, key) and not isinstance(
            getattr(type(config), key, None), property
        ):
            setattr(config, key, value)
        else:
            raise ValueError(
                f"Invalid parameter {key} for {config.__class__.__name__}. "
                "Check the list of available parameters with `get_params()`."
            )

    # Apply nested parameters to frozen Period dataclasses
    if nested_period_params and config.periods is not None:
        existing_names = {p.name for p in config.periods}
        for p_name in nested_period_params:
            if p_name not in existing_names:
                raise ValueError(
                    f"Period with name '{p_name}' not found in configuration."
                )

        new_periods = []
        for period in config.periods:
            if period.name in nested_period_params:
                new_periods.append(replace(period, **nested_period_params[period.name]))
            else:
                new_periods.append(period)
        config.periods = new_periods

    validate_config(config)
    return config


def validate_config(config: object) -> None:
    """Reject clearly-invalid hyperparameter values (fail-safe).

    Conservative on purpose: only values that are unambiguously invalid are
    rejected, so a configuration a caller legitimately builds is never refused.
    Invoked by both ``__init__`` and ``set_params`` so the guard cannot be
    bypassed by mutating an instance after construction.

    Raises:
        ValueError: For a non-positive ``predict_size``, a ``contamination``
            outside ``[0, 0.5]``, a ``poly_features_degree`` below 1, a
            ``poly_mi_sample_size`` below 1 (``None`` is allowed), an
            ``on_weather_failure`` / ``on_exog_provider_failure`` that is not
            ``"raise"`` or ``"skip"``, a negative ``exog_max_gap_hours``, a
            negative ``exog_max_tail_gap_hours``, an
            ``exog_provider_window`` that is not ``"full"`` or ``"train"``,
            an ``on_load_lag_failure`` that is not ``"raise"`` or ``"skip"``,
            a ``load_lag_sources`` outside ``{"total", "zones",
            "zone_shares"}``, a negative ``load_lag_max_staleness_hours``, an
            empty/None ``load_lag_hours`` or one containing an entry below
            168 when ``include_load_lag_exog`` is set, or a
            ``zone_weather_columns=True`` combined with ``per_zone_weather``,
            ``use_population_weighted_weather``,
            ``use_exogenous_features=False``, or ``poly_features_degree>=2``.
        TypeError: If ``zone_weather_locations`` is set to something other
            than a ``dict`` while ``zone_weather_columns`` (or
            ``per_zone_weather``) is ``True``.

    Examples:
        ```{python}
        from spotforecast2_safe.configurator._base_config import validate_config
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        config = ConfigMulti(country_code="DE")
        # A freshly constructed config always passes validation
        validate_config(config)
        print("Default config is valid.")

        # Demonstrate that an invalid value is rejected
        config.predict_size = -1
        try:
            validate_config(config)
        except ValueError as exc:
            print(f"Caught expected error: {exc}")

        # Restore and check contamination boundary
        config.predict_size = 24
        config.contamination = 0.9  # out of [0, 0.5]
        try:
            validate_config(config)
        except ValueError as exc:
            print(f"Caught expected error: {exc}")
        ```
    """
    predict_size = getattr(config, "predict_size", None)
    if predict_size is not None and predict_size <= 0:
        raise ValueError(f"predict_size must be positive; got {predict_size}.")

    contamination = getattr(config, "contamination", None)
    if contamination is not None and not 0 <= contamination <= 0.5:
        raise ValueError(
            f"contamination must be between 0 and 0.5; got {contamination}."
        )

    poly_features_degree = getattr(config, "poly_features_degree", None)
    if poly_features_degree is not None and poly_features_degree < 1:
        raise ValueError(
            f"poly_features_degree must be >= 1; got {poly_features_degree}."
        )

    poly_mi_sample_size = getattr(config, "poly_mi_sample_size", None)
    if poly_mi_sample_size is not None and poly_mi_sample_size < 1:
        raise ValueError(
            f"poly_mi_sample_size must be a positive integer or None; "
            f"got {poly_mi_sample_size}."
        )

    on_weather_failure = getattr(config, "on_weather_failure", None)
    if on_weather_failure is not None and on_weather_failure not in ("raise", "skip"):
        raise ValueError(
            f"on_weather_failure must be 'raise' or 'skip'; got {on_weather_failure!r}."
        )

    on_exog_provider_failure = getattr(config, "on_exog_provider_failure", None)
    if on_exog_provider_failure is not None and on_exog_provider_failure not in (
        "raise",
        "skip",
    ):
        raise ValueError(
            "on_exog_provider_failure must be 'raise' or 'skip'; got "
            f"{on_exog_provider_failure!r}."
        )

    exog_max_gap_hours = getattr(config, "exog_max_gap_hours", None)
    if exog_max_gap_hours is not None and exog_max_gap_hours < 0:
        raise ValueError(f"exog_max_gap_hours must be >= 0; got {exog_max_gap_hours}.")

    exog_max_tail_gap_hours = getattr(config, "exog_max_tail_gap_hours", None)
    if exog_max_tail_gap_hours is not None and exog_max_tail_gap_hours < 0:
        raise ValueError(
            f"exog_max_tail_gap_hours must be >= 0; got {exog_max_tail_gap_hours}."
        )
    if (
        exog_max_tail_gap_hours is not None
        and exog_max_gap_hours is not None
        and 0 < exog_max_tail_gap_hours <= exog_max_gap_hours
    ):
        _base_logger.warning(
            "exog_max_tail_gap_hours=%d is inert: tail budget subsumed by "
            "exog_max_gap_hours=%d",
            exog_max_tail_gap_hours,
            exog_max_gap_hours,
        )

    exog_provider_window = getattr(config, "exog_provider_window", None)
    if exog_provider_window is not None and exog_provider_window not in (
        "full",
        "train",
    ):
        raise ValueError(
            "exog_provider_window must be 'full' or 'train'; got "
            f"{exog_provider_window!r}."
        )

    target_qc_range_mw = getattr(config, "target_qc_range_mw", None)
    if target_qc_range_mw is not None and target_qc_range_mw < 0:
        raise ValueError(
            f"target_qc_range_mw must be >= 0 if set; got {target_qc_range_mw}."
        )

    target_qc_step_mw = getattr(config, "target_qc_step_mw", None)
    if target_qc_step_mw is not None and target_qc_step_mw < 0:
        raise ValueError(
            f"target_qc_step_mw must be >= 0 if set; got {target_qc_step_mw}."
        )

    target_qc_window_days = getattr(config, "target_qc_window_days", None)
    if target_qc_window_days is not None and target_qc_window_days < 1:
        raise ValueError(
            f"target_qc_window_days must be >= 1 if set; got {target_qc_window_days}."
        )

    target_corruption_policy = getattr(config, "target_corruption_policy", None)
    if target_corruption_policy is not None and target_corruption_policy not in (
        "abort",
        "heal",
        "truncate",
    ):
        raise ValueError(
            "target_corruption_policy must be one of 'abort', 'heal', 'truncate'; "
            f"got {target_corruption_policy!r}."
        )

    target_max_heal_hours = getattr(config, "target_max_heal_hours", None)
    if target_max_heal_hours is not None and target_max_heal_hours < 0:
        raise ValueError(
            f"target_max_heal_hours must be >= 0; got {target_max_heal_hours}."
        )

    target_anchor_zone_hours = getattr(config, "target_anchor_zone_hours", None)
    if target_anchor_zone_hours is not None and target_anchor_zone_hours < 0:
        raise ValueError(
            f"target_anchor_zone_hours must be >= 0; got {target_anchor_zone_hours}."
        )

    target_qc_deviation_mw = getattr(config, "target_qc_deviation_mw", None)
    if target_qc_deviation_mw is not None and target_qc_deviation_mw < 0:
        raise ValueError(
            f"target_qc_deviation_mw must be >= 0 if set; got {target_qc_deviation_mw}."
        )

    target_qc_deviation_ref = getattr(config, "target_qc_deviation_ref", None)
    if target_qc_deviation_ref is not None and not isinstance(
        target_qc_deviation_ref, str
    ):
        raise ValueError(
            "target_qc_deviation_ref must be a column name (str) if set; "
            f"got {target_qc_deviation_ref!r}."
        )

    target_qc_deviation_slots = getattr(config, "target_qc_deviation_slots", None)
    if target_qc_deviation_slots is not None and target_qc_deviation_slots < 1:
        raise ValueError(
            f"target_qc_deviation_slots must be >= 1; got {target_qc_deviation_slots}."
        )

    per_zone_weather = getattr(config, "per_zone_weather", False)
    if per_zone_weather:
        if getattr(config, "use_population_weighted_weather", False):
            raise ValueError(
                "per_zone_weather and use_population_weighted_weather are mutually "
                "exclusive (zones use regional weather, not the global index)."
            )
        if not getattr(config, "use_exogenous_features", True):
            raise ValueError(
                "per_zone_weather requires exogenous features "
                "(use_exogenous_features must be True)."
            )
        if getattr(config, "poly_features_degree", 1) >= 2:
            raise ValueError(
                "per-zone weather does not support polynomial interaction features "
                "(poly_features_degree>=2), whose products are precomputed from the "
                "shared weather."
            )
        zone_weather_locations = getattr(config, "zone_weather_locations", None)
        if zone_weather_locations is not None and not isinstance(
            zone_weather_locations, dict
        ):
            raise TypeError(
                "zone_weather_locations must be a dict or None; "
                f"got {type(zone_weather_locations).__name__!r}."
            )

    zone_weather_columns = getattr(config, "zone_weather_columns", False)
    if zone_weather_columns:
        if getattr(config, "per_zone_weather", False):
            raise ValueError(
                "zone_weather_columns and per_zone_weather are mutually exclusive "
                "(per_zone_weather overwrites weather per multi-zone target; "
                "zone_weather_columns concatenates all zones for one target)."
            )
        if getattr(config, "use_population_weighted_weather", False):
            raise ValueError(
                "zone_weather_columns and use_population_weighted_weather are "
                "mutually exclusive (zone_weather_columns already supplies "
                "population-weighted weather per zone)."
            )
        if not getattr(config, "use_exogenous_features", True):
            raise ValueError(
                "zone_weather_columns requires exogenous features "
                "(use_exogenous_features must be True)."
            )
        if getattr(config, "poly_features_degree", 1) >= 2:
            raise ValueError(
                "zone_weather_columns does not support polynomial interaction "
                "features (poly_features_degree>=2), whose products are "
                "precomputed from the shared weather."
            )
        zwl = getattr(config, "zone_weather_locations", None)
        if zwl is not None and not isinstance(zwl, dict):
            raise TypeError(
                f"zone_weather_locations must be a dict or None; got {type(zwl).__name__!r}."
            )

    on_load_lag_failure = getattr(config, "on_load_lag_failure", None)
    if on_load_lag_failure is not None and on_load_lag_failure not in (
        "raise",
        "skip",
    ):
        raise ValueError(
            f"on_load_lag_failure must be 'raise' or 'skip'; got {on_load_lag_failure!r}."
        )

    load_lag_sources = getattr(config, "load_lag_sources", None)
    if load_lag_sources is not None and load_lag_sources not in (
        "total",
        "zones",
        "zone_shares",
    ):
        raise ValueError(
            "load_lag_sources must be 'total', 'zones', or 'zone_shares'; "
            f"got {load_lag_sources!r}."
        )

    load_lag_max_staleness_hours = getattr(config, "load_lag_max_staleness_hours", None)
    if load_lag_max_staleness_hours is not None and load_lag_max_staleness_hours < 0:
        raise ValueError(
            "load_lag_max_staleness_hours must be >= 0; "
            f"got {load_lag_max_staleness_hours}."
        )

    if getattr(config, "include_load_lag_exog", False):
        if not getattr(config, "use_exogenous_features", True):
            raise ValueError(
                "include_load_lag_exog requires exogenous features "
                "(use_exogenous_features must be True)."
            )
        hours = getattr(config, "load_lag_hours", None)
        if not hours:
            raise ValueError("load_lag_hours must be a non-empty tuple of ints.")
        if any(int(h) < 168 for h in hours):
            raise ValueError(
                "every load_lag_hours entry must be >= 168 (horizon-availability "
                "safety: the lagged value must already be published at forecast "
                "time)."
            )

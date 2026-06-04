# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared configuration helpers for the pipeline config classes.

``ConfigEntsoe`` and ``ConfigMulti`` share the same cyclical-feature period
defaults and the same ``get_params`` / ``set_params`` semantics. Those were
previously hand-duplicated in both classes, which let fields drift between the
two copies. These helpers centralise the logic so each config only declares its
ordered parameter names (``_PARAM_NAMES``) and its ``__init__``; the
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
    frozen ``Period`` dataclasses via :func:`dataclasses.replace`.

    Args:
        config: The configuration instance to mutate.
        params: Optional dictionary of parameter names mapped to new values.
        **kwargs: Additional parameter names mapped to new values.

    Returns:
        The same ``config`` instance (supports method chaining).

    Raises:
        ValueError: For an unknown flat parameter, a malformed ``periods__``
            key, or a period name not present in the configuration.
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
            negative ``exog_max_tail_gap_hours``, or an
            ``exog_provider_window`` that is not ``"full"`` or ``"train"``.
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
            "on_weather_failure must be 'raise' or 'skip'; got "
            f"{on_weather_failure!r}."
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

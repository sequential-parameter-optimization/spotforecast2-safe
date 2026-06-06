# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests that ``ConfigEntsoe`` satisfies the ``PipelineConfig`` protocol
surface that ``spotforecast2.multitask.base.BaseTask`` reads.

After the RunState extraction (ADR ``adr-multitask-configmulti-merge``,
v18.0.0) the 8 derived window fields (start_download, end_download,
data_start, data_end, cov_start, cov_end, end_train_ts, start_train_ts)
are no longer part of the ``PipelineConfig`` protocol or the config object.
They live on ``task.run_state`` instead.
"""

import pytest

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe

PIPELINE_PROTOCOL_FIELDS = [
    # Targets and aggregation
    ("targets", None),
    ("agg_weights", None),
    ("bounds", None),
    # Forecast horizon and training window (user input)
    ("predict_size", 24),
    ("refit_size", 7),
    # Outlier detection / imputation
    ("use_outlier_detection", True),
    ("contamination", 0.01),
    ("imputation_method", "weighted"),
    ("window_size", 72),
    # Exogenous features
    ("use_exogenous_features", True),
    ("include_weather_windows", False),
    ("include_holiday_features", False),
    ("include_holiday_adjacency_features", False),
    ("poly_features_degree", 1),
    ("max_poly_features", 10),
    ("latitude", 51.5136),
    ("longitude", 7.4653),
    ("timezone", "UTC"),
    ("state", "NW"),
    # Misc
    ("random_state", 314159),
    ("cache_home", None),
    ("index_name", "Time (UTC)"),
    ("verbose", False),
    ("task", "lazy"),
    # Tuning trial budgets
    ("n_trials_optuna", 15),
    ("n_trials_spotoptim", 10),
    ("n_initial_spotoptim", 5),
    # Hooks
    ("forecaster_factory", None),
    ("data_loader", None),
    # Gap healing and provider-window
    ("exog_max_gap_hours", 0),
    ("exog_max_tail_gap_hours", 0),
    ("exog_provider_window", "full"),
]

# Fields that were removed from the config (now live on task.run_state).
DERIVED_FIELDS_REMOVED = [
    "data_start",
    "data_end",
    "cov_start",
    "cov_end",
    "start_download",
    "end_download",
    "start_train_ts",
    "end_train_ts",
]


@pytest.mark.parametrize("field, default", PIPELINE_PROTOCOL_FIELDS)
def test_pipeline_protocol_field_default(field, default):
    """Each field required by the PipelineConfig protocol exists with the
    documented default on a freshly-constructed ConfigEntsoe."""
    config = ConfigEntsoe()
    assert hasattr(config, field), f"ConfigEntsoe missing field {field!r}"
    assert getattr(config, field) == default, (
        f"ConfigEntsoe.{field} default = {getattr(config, field)!r}; "
        f"expected {default!r}"
    )


@pytest.mark.parametrize("field", DERIVED_FIELDS_REMOVED)
def test_derived_field_not_in_param_names(field):
    """Derived fields must NOT appear in ConfigEntsoe._PARAM_NAMES."""
    assert field not in ConfigEntsoe._PARAM_NAMES, (
        f"Derived field '{field}' is still in _PARAM_NAMES; "
        "it should live on task.run_state instead."
    )


@pytest.mark.parametrize("field", DERIVED_FIELDS_REMOVED)
def test_derived_field_not_in_get_params(field):
    """Derived fields must NOT appear in get_params() output."""
    params = ConfigEntsoe().get_params()
    assert field not in params, (
        f"Derived field '{field}' leaked into get_params(); "
        "it should live on task.run_state instead."
    )


def test_country_code_constructor_kwarg_and_attribute():
    """``country_code`` is the single canonical attribute."""
    config = ConfigEntsoe(country_code="FR")
    assert config.country_code == "FR"

    config.country_code = "ES"
    assert config.country_code == "ES"


def test_forecaster_factory_stores_callable_verbatim():
    """The hook fields accept callables and are stored unchanged so the
    multitask machinery can invoke them later."""

    def stub_factory(config, *, weight_func=None, target=None):
        return ("stub", target)

    config = ConfigEntsoe(forecaster_factory=stub_factory)
    assert config.forecaster_factory is stub_factory


def test_data_loader_stores_callable_verbatim():
    def stub_loader(config):
        return ("stub-df",)

    config = ConfigEntsoe(data_loader=stub_loader)
    assert config.data_loader is stub_loader


def test_get_params_includes_all_protocol_fields():
    """``get_params()`` exposes every protocol field so callers
    using ``config.set_params(get_params())`` round-trip without loss."""
    config = ConfigEntsoe()
    params = config.get_params()
    for field, _default in PIPELINE_PROTOCOL_FIELDS:
        assert field in params, f"get_params() missing {field!r}"


def test_set_params_accepts_new_fields():
    """``set_params`` accepts the protocol field names."""
    config = ConfigEntsoe()
    config.set_params(
        targets=["Actual Load"],
        agg_weights=[1.0],
        use_outlier_detection=False,
        n_trials_optuna=42,
        index_name="DateTime",
    )
    assert config.targets == ["Actual Load"]
    assert config.agg_weights == [1.0]
    assert config.use_outlier_detection is False
    assert config.n_trials_optuna == 42
    assert config.index_name == "DateTime"


@pytest.mark.parametrize("field", DERIVED_FIELDS_REMOVED)
def test_set_params_derived_field_raises(field):
    """``set_params`` must raise ``ValueError`` for each removed derived field.

    These fields have moved to ``task.run_state``; accepting them silently
    via ``set_params`` would give the misleading impression that they are
    user-configurable config parameters.
    """
    config = ConfigEntsoe()
    with pytest.raises(ValueError, match=field):
        config.set_params(**{field: object()})

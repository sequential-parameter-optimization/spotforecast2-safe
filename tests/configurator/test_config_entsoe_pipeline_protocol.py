# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests that ``ConfigEntsoe`` satisfies the full ``PipelineConfig`` protocol
surface that ``spotforecast2.multitask.base.BaseTask`` reads (ADR-002 Step 1).

The fields listed here are exactly those required by the multitask
``BaseTask`` to drive single-target forecasting through the unified
``run(config_cls=ConfigEntsoe, ...)`` entry point.
"""

import pytest

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe


PIPELINE_PROTOCOL_FIELDS = [
    # Targets and aggregation
    ("targets", None),
    ("agg_weights", None),
    ("bounds", None),
    # Forecast horizon and training window
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
    ("include_poly_features", False),
    ("latitude", 51.5136),
    ("longitude", 7.4653),
    ("timezone", "UTC"),
    ("state", "NW"),
    # Data ranges (derived after data loading)
    ("data_start", None),
    ("data_end", None),
    ("cov_start", None),
    ("cov_end", None),
    ("start_download", None),
    ("end_download", None),
    ("start_train_ts", None),
    ("end_train_ts", None),
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


def test_country_code_property_aliases_api_country_code():
    """The ``country_code`` property mirrors ``API_COUNTRY_CODE`` in both
    directions (PipelineConfig reads ``country_code``; legacy callers read
    ``API_COUNTRY_CODE``)."""
    config = ConfigEntsoe(api_country_code="FR")
    assert config.country_code == "FR"
    assert config.API_COUNTRY_CODE == "FR"

    config.country_code = "ES"
    assert config.API_COUNTRY_CODE == "ES"

    config.API_COUNTRY_CODE = "IT"
    assert config.country_code == "IT"


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


def test_get_params_includes_all_new_fields():
    """``get_params()`` exposes every new pipeline-protocol field so callers
    using ``config.set_params(get_params())`` round-trip without loss."""
    config = ConfigEntsoe()
    params = config.get_params()
    for field, _default in PIPELINE_PROTOCOL_FIELDS:
        assert field in params, f"get_params() missing {field!r}"


def test_set_params_accepts_new_fields():
    """``set_params`` accepts the new pipeline-protocol field names."""
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

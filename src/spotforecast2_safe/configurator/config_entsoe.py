# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration for the ENTSO-E (single-target) task pipeline."""

from dataclasses import dataclass, field, fields

import pandas as pd

from spotforecast2_safe.configurator.config_multi import ConfigMulti


@dataclass
class ConfigEntsoe(ConfigMulti):
    """Configuration for the ENTSO-E forecasting pipeline.

    Single-target counterpart to `ConfigMulti`, used by the ENTSO-E CLI
    (``spotforecast2.tasks.task_entsoe``) and any single-target pipeline routed
    through ``spotforecast2.multitask.runner.run(config_cls=ConfigEntsoe)``.

    ``ConfigEntsoe`` **inherits every field and method of `ConfigMulti`** — so any
    feature flag added to ``ConfigMulti`` is available here automatically (this
    is what closes the historical feature-flag parity gap structurally, rather
    than via a hand-maintained mirror). It differs from ``ConfigMulti`` in
    exactly two ways:

    - ``index_name`` defaults to ``"Time (UTC)"`` (the ENTSO-E CSV time column)
      instead of ``"DateTime"``.
    - it adds ``retrain_max_age`` — the maximum age of a previously trained model
      before retraining is required (consumed by
      `spotforecast2_safe.manager.trainer.should_retrain`).

    See `ConfigMulti` for the full field reference (training/validation windows,
    feature toggles, exogenous-provider flags, target-corruption knobs, …).

    Args:
        index_name (str): Datetime column name used when resetting the index.
            Defaults to ``"Time (UTC)"``.
        retrain_max_age (pd.Timedelta): Maximum age of a trained model before a
            retrain is forced. Defaults to ``pd.Timedelta(days=7)``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        config = ConfigEntsoe(country_code="DE")
        # ENTSO-E-specific defaults:
        print("index_name:", config.index_name)
        print("retrain_max_age:", config.retrain_max_age)
        assert config.index_name == "Time (UTC)"
        assert config.retrain_max_age == pd.Timedelta(days=7)

        # Inherits the full ConfigMulti surface, incl. the opt-in feature flags:
        assert isinstance(config, ConfigMulti)
        config = ConfigEntsoe(
            include_ephemeris_features=True,
            include_day_type_features=True,
            include_degree_hours=True,
        )
        print("ephemeris:", config.include_ephemeris_features)
        print("predict_size:", config.predict_size)
        ```
    """

    # Default override (re-declaring an inherited field keeps its position and
    # only changes the default).
    index_name: str = "Time (UTC)"
    # ENTSO-E-only field (appended after the inherited ConfigMulti fields).
    retrain_max_age: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(days=7))


# ``_PARAM_NAMES`` is derived from the dataclass fields (declaration order, which
# for a subclass is the base fields followed by the subclass-only fields) so it
# can never drift from the actual fields; consumers and tests read it as a class
# attribute. Set after the class body because ``fields()`` needs the finished
# dataclass.
ConfigEntsoe._PARAM_NAMES = tuple(f.name for f in fields(ConfigEntsoe))

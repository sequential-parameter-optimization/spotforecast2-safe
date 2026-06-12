# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ENTSO-E interim-CSV data loaders.

Config-driven loaders for the merged ENTSO-E interim CSV, suitable for the
``data_loader`` / ``test_data_loader`` hooks on `ConfigEntsoe`. Ported from
``spotforecast2.tasks.task_entsoe`` ahead of that subpackage's removal.
"""

from pathlib import Path

import pandas as pd

from spotforecast2_safe.configurator import ConfigEntsoe
from spotforecast2_safe.data.fetch_data import get_data_home


def entsoe_data_loader(config: ConfigEntsoe) -> pd.DataFrame:
    """Read the merged interim ENTSO-E CSV that ``config.data_filename`` points at.

    Args:
        config: A `ConfigEntsoe` with ``data_filename`` set.  Relative paths
            are resolved against `spotforecast2_safe.data.fetch_data.get_data_home`.

    Returns:
        DataFrame indexed by the ENTSO-E timestamp column (``Time (UTC)``)
        with the load columns as data columns.

    Raises:
        FileNotFoundError: If the merged CSV does not exist.  Run
            ``spotforecast2-entsoe download`` and ``merge`` first.

    Examples:
        ```{python}
        import os
        import tempfile

        import pandas as pd
        from spotforecast2_safe.configurator import ConfigEntsoe
        from spotforecast2_safe.data.entsoe_loader import entsoe_data_loader

        # Build a tiny synthetic interim CSV in a temp directory.
        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "energy_load.csv")
        idx = pd.date_range(
            "2025-01-01", periods=48, freq="h", tz="UTC", name="Time (UTC)"
        )
        pd.DataFrame({"Actual Load": range(48)}, index=idx).to_csv(csv_path)

        # Absolute path bypasses get_data_home; loader returns the full frame.
        config = ConfigEntsoe()
        config.data_filename = csv_path
        df = entsoe_data_loader(config)

        print(df.shape)
        assert df.shape == (48, 1)
        assert df.index.name == "Time (UTC)"
        ```
    """
    path = Path(config.data_filename)
    if not path.is_absolute():
        path = get_data_home() / path
    if not path.exists():
        raise FileNotFoundError(
            f"ENTSO-E merged CSV not found at {path}.  Run "
            "`spotforecast2-entsoe download` and `merge` first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def entsoe_test_data_loader(config: ConfigEntsoe) -> pd.DataFrame:
    """Return the merged ENTSO-E CSV sliced to the forecast horizon.

    The slice spans ``(end_train, end_train + predict_size * 1 h]`` so that
    ``build_prediction_package``'s ``test_actual = ts.reindex(future_pred.index)``
    matches the hourly forecast row-for-row.  ``end_train`` is taken from
    ``config.end_train_default`` (treated as the *inclusive* last training
    timestamp, the same convention the forecaster uses), and the step is
    assumed to be 1 h after the pipeline's hourly resampling.

    For the live ENTSO-E exemplar with ``end_train_default = D-2 23:00 UTC``
    and ``predict_size = 24``, this returns the rows for
    ``[D-1 00:00, D 00:00)`` — i.e., ``y_{-1}``.  For backtests at an arbitrary
    ``end_train_default``, it returns the post-cutoff window the model is
    actually predicting, rather than always "yesterday in wall-clock UTC".

    Args:
        config: A `ConfigEntsoe` with ``data_filename``, ``end_train_default``,
            and ``predict_size`` set; the merged interim CSV must already
            contain data covering the forecast horizon (run
            ``spotforecast2-entsoe download`` first).

    Returns:
        DataFrame indexed by ``Time (UTC)`` with the rows the forecast will be
        scored against.

    Examples:
        ```{python}
        import os
        import tempfile

        import pandas as pd
        from spotforecast2_safe.configurator import ConfigEntsoe
        from spotforecast2_safe.data.entsoe_loader import entsoe_test_data_loader

        # Synthetic interim CSV spanning the forecast window.
        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "energy_load.csv")
        idx = pd.date_range(
            "2025-12-29 00:00", periods=120, freq="h", tz="UTC", name="Time (UTC)"
        )
        pd.DataFrame({"Actual Load": range(120)}, index=idx).to_csv(csv_path)

        config = ConfigEntsoe()
        config.data_filename = csv_path
        config.end_train_default = "2025-12-31 00:00+00:00"
        config.predict_size = 24

        test_df = entsoe_test_data_loader(config)

        # The slice covers exactly predict_size hourly steps after end_train.
        print(test_df.shape)
        assert test_df.shape == (24, 1)
        assert test_df.index[0] == pd.Timestamp("2025-12-31 01:00", tz="UTC")
        ```
    """
    df = entsoe_data_loader(config)
    end_train = pd.Timestamp(config.end_train_default)
    if end_train.tzinfo is None:
        end_train = end_train.tz_localize("UTC")
    step = pd.Timedelta(hours=1)  # post-resample assumption
    start = end_train + step  # first forecast step
    end = start + config.predict_size * step  # exclusive upper bound
    if df.index.tz is None:
        start = start.tz_localize(None)
        end = end.tz_localize(None)
    return df.loc[(df.index >= start) & (df.index < end)]

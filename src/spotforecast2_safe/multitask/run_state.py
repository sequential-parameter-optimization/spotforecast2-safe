# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Task-owned runtime state derived during pipeline execution.

Design rationale (ADR ``adr-multitask-configmulti-merge``, accepted 2026-06-06):

``ConfigMulti`` and ``ConfigEntsoe`` are **user-input objects** — they carry
the parameters a caller passes when constructing a pipeline task.  However,
the pipeline also derives a set of *window-geometry* values that only exist
once data has been loaded (``prepare_data``) or the training window computed
(``_setup_training_window``).  Historically these derived values were written
back onto the config, making the config non-reproducible mid-pipeline and
creating a shared-config cross-contamination footgun.

``RunState`` is the task-owned companion object that holds **only** those
derived values.  A fresh ``RunState`` has every field as ``None``; the
pipeline fills it in two stages:

1. ``prepare_data`` fills: ``start_download``, ``end_download``,
   ``data_start``, ``data_end``, ``cov_start``, ``cov_end``, ``targets``.
2. ``_setup_training_window`` fills: ``end_train_ts``, ``start_train_ts``.

The config object is never mutated by the pipeline.  Callers that need the
derived geometry read it from ``task.run_state``.

Input/derived split in one sentence: if the value exists before any data is
loaded, it belongs on the config; if the pipeline *computes* it from the data,
it belongs here.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(slots=True)
class RunState:
    """Runtime-derived window geometry for one pipeline execution.

    All fields default to ``None`` and are populated progressively by the
    pipeline.  After a successful ``prepare_data`` call the first seven fields
    are set; after ``_setup_training_window`` (called lazily before any
    training step) the two training-window timestamps are also set.

    Attributes:
        start_download: First timestamp of the raw downloaded data as a
            ``"YYYYMMDDHHMM"`` string.  Set by ``prepare_data``.
        end_download: Last timestamp of the raw downloaded data as a
            ``"YYYYMMDDHHMM"`` string.  Set by ``prepare_data``.
        data_start: First timestamp of the pipeline data range, as a
            tz-aware ``pd.Timestamp``.  Set by ``prepare_data`` via
            ``get_start_end()``.
        data_end: Last timestamp of the pipeline data range, as a
            tz-aware ``pd.Timestamp``.  Set by ``prepare_data`` via
            ``get_start_end()``.
        cov_start: Start of the covariate date range (equals ``data_start``).
            Set by ``prepare_data`` via ``get_start_end()``.
        cov_end: End of the covariate date range (extends ``data_end`` by
            ``predict_size`` hours).  Set by ``prepare_data`` via
            ``get_start_end()``.
        end_train_ts: End of the training window, derived from
            ``config.end_train_default`` (clamped to ``data_end`` when the
            user value exceeds the data extent).  Set by
            ``_setup_training_window``.
        start_train_ts: Start of the training window, derived as
            ``end_train_ts - config.train_size`` (floored at the pipeline
            data start).  Set by ``_setup_training_window``.
        targets: Resolved list of target column names after reconciling the
            user-supplied ``config.targets`` value against the columns
            actually present in the pipeline DataFrame.  ``None`` until
            ``prepare_data`` completes.

    Examples:
        ```{python}
        import tempfile
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.multitask.run_state import RunState
        from spotforecast2_safe.multitask import MultiTask
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        # A freshly constructed RunState has every field set to None.
        rs = RunState()
        assert rs.targets is None
        assert rs.data_start is None
        print("Fresh RunState (all None):", rs)

        # The pipeline fills the fields progressively.  prepare_data populates
        # data_start, data_end, cov_start, cov_end, and targets.
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=24 * 14, freq="h", tz="UTC")
        df = pd.DataFrame({"load": rng.normal(100, 10, len(idx))}, index=idx)
        df.index.name = "DateTime"

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(
                predict_size=6,
                use_exogenous_features=False,
                use_outlier_detection=False,
                cache_home=tmp,
            )
            mt = MultiTask(cfg, dataframe=df)
            mt.prepare_data()

            rs = mt.run_state
            assert rs.targets == ["load"]
            assert isinstance(rs.data_start, pd.Timestamp)
            assert isinstance(rs.data_end, pd.Timestamp)
            print(f"targets: {rs.targets}")
            print(f"data_start: {rs.data_start}")
            print(f"data_end:   {rs.data_end}")
            print(f"cov_end extends data_end by predict_size hours: "
                  f"{rs.cov_end > rs.data_end}")
        ```
    """

    start_download: Optional[str] = None
    end_download: Optional[str] = None
    data_start: Optional[pd.Timestamp] = None
    data_end: Optional[pd.Timestamp] = None
    cov_start: Optional[pd.Timestamp] = None
    cov_end: Optional[pd.Timestamp] = None
    end_train_ts: Optional[pd.Timestamp] = None
    start_train_ts: Optional[pd.Timestamp] = None
    targets: Optional[List[str]] = None

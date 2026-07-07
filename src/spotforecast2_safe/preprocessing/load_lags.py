# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lagged-load exogenous features (``include_load_lag_exog`` opt-in).

Appends lagged-load columns shifted by each ``L`` in
``ConfigMulti.load_lag_hours`` (every ``L`` must be ``>= 168`` hours, so the
value available at forecast time ``t`` is the raw series at ``t - L``, already
published — no future leakage). Three source families
(``ConfigMulti.load_lag_sources``):

- ``"total"`` — lags the primary target series itself
  (``load_lag_<L>``). Complementary to the native recursive lags
  (``{1, 2, 24}`` in the campaign), not redundant with them.
- ``"zones"`` — lags the four German TSO-zone series from
  ``spotforecast2_safe.downloader.entsoe.load_interim(mode="four_zone")``,
  resampled to an hourly mean (``load_<zone>_lag_<L>``).
- ``"zone_shares"`` — computes each zone's bottom-up load share
  (``share_z(t) = load_z(t) / sum_zones load(t)``) *before* lagging, keeps
  three of the four zones (dropping ``transnetbw``, the smallest), then lags
  the shares (``share_<zone>_lag_<L>``).

Fail-safe contract (mirrors `spotforecast2_safe.preprocessing.exog_providers`):
a source that cannot supply a NaN-free value for every requested timestamp —
after backfilling only the unavoidable leading warm-up gap the shift itself
creates, bounded to history (never past ``data_end``) — raises
:class:`LoadLagError` rather than silently imputing or forward-filling across
the leakage boundary. The forecast-horizon slice (``(data_end, cov_end]``) is
always hard-checked for NaN. Staleness is checked PER SOURCE COLUMN (a
quietly-stopped zone cannot hide behind its still-live siblings): any column
whose last observation lags ``data_end`` by more than ``max_staleness_hours``
raises, naming that column. ``multitask.base.build_exogenous_features``
governs the failure via ``ConfigMulti.on_load_lag_failure`` (``"raise"``
propagates; ``"skip"`` logs and omits the whole block).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# The four German TSO-zone column names, as written by
# spotforecast2_safe.downloader.entsoe.load_interim(mode="four_zone")
# (spotforecast2_safe.downloader._zones.ZONE_COLUMNS). Duplicated here as
# string literals (not imported) so this module stays free of any downloader
# dependency -- it only ever receives the already-loaded zone_frame.
ZONE_LOAD_COLUMNS: Tuple[str, ...] = (
    "load_50hertz",
    "load_amprion",
    "load_tennet",
    "load_transnetbw",
)

# Dropped from "zone_shares" — the smallest of the four zones; see the ADR's
# resolved design decision (bottom-up four-zone sum as the shares divisor).
_DROPPED_SHARE_ZONE = "load_transnetbw"


class LoadLagError(Exception):
    """A lagged-load exog source could not be built NaN-free over the window.

    Raised when the requested source (target total, per-zone loads, or
    zone shares) cannot supply a value for every requested timestamp after
    the leading warm-up backfill, when the forecast-horizon slice contains
    NaN, or when the source series is stale (its last observation lags
    ``data_end`` by more than ``max_staleness_hours``).
    ``multitask.base.build_exogenous_features`` translates this into either a
    hard failure or a skipped feature block, governed by
    ``ConfigMulti.on_load_lag_failure``.

    Examples:
        ```{python}
        from spotforecast2_safe.preprocessing.load_lags import LoadLagError

        try:
            raise LoadLagError("load_lags: source stale.")
        except LoadLagError as exc:
            print(type(exc).__name__, str(exc))
        assert issubclass(LoadLagError, Exception)
        ```
    """


def _harmonise_tz(obj: Union[pd.Series, pd.DataFrame], like_index: pd.DatetimeIndex):
    """Match *obj*'s ``DatetimeIndex`` tz-awareness to *like_index*."""
    obj = obj.copy()
    if like_index.tz is not None:
        obj.index = (
            obj.index.tz_localize("UTC")
            if obj.index.tz is None
            else obj.index.tz_convert(like_index.tz)
        )
    elif obj.index.tz is not None:
        obj.index = obj.index.tz_convert("UTC").tz_localize(None)
    return obj


def _as_timestamp(ts, *, like_index: pd.DatetimeIndex) -> pd.Timestamp:
    """Coerce *ts* to a ``pd.Timestamp`` with tz-awareness matching *like_index*."""
    out = pd.Timestamp(ts)
    if like_index.tz is not None and out.tz is None:
        out = out.tz_localize(like_index.tz)
    elif like_index.tz is None and out.tz is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out


def _resolve_source(
    *,
    target_series: pd.Series,
    zone_frame: Optional[pd.DataFrame],
    load_lag_sources: str,
    index: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return ``(source_hourly, name_prefix)`` for the requested source family.

    ``source_hourly`` is an hourly-mean-resampled frame (one column per
    lagged source); ``name_prefix`` maps each of its columns to the output
    column-name prefix (``"load"`` for total, ``"load_<zone>"`` for zones,
    ``"share_<zone>"`` for zone shares).

    Raises:
        LoadLagError: If *load_lag_sources* is not one of the three known
            values, ``zone_frame`` is required but missing, is missing one
            of the four required zone columns, or (for ``"zone_shares"``)
            the share division produces ``+/-inf``.
    """
    if load_lag_sources not in ("total", "zones", "zone_shares"):
        raise LoadLagError(
            f"load_lag_sources must be 'total', 'zones', or 'zone_shares'; "
            f"got {load_lag_sources!r}."
        )

    if load_lag_sources == "total":
        source = _harmonise_tz(target_series.rename("load"), index)
        source_hourly = source.to_frame().resample("h").mean()
        return source_hourly, {"load": "load"}

    if zone_frame is None:
        raise LoadLagError(
            f"load_lag_sources={load_lag_sources!r} requires zone_frame "
            "(spotforecast2_safe.downloader.entsoe.load_interim"
            "(mode='four_zone'))."
        )
    missing = [c for c in ZONE_LOAD_COLUMNS if c not in zone_frame.columns]
    if missing:
        raise LoadLagError(
            f"zone_frame is missing required zone column(s): {missing} "
            f"(expected all of {list(ZONE_LOAD_COLUMNS)})."
        )
    zf = _harmonise_tz(zone_frame[list(ZONE_LOAD_COLUMNS)], index)
    zf_hourly = zf.resample("h").mean()

    if load_lag_sources == "zones":
        return zf_hourly, {c: c for c in zf_hourly.columns}

    # load_lag_sources == "zone_shares" (the only remaining valid value; the
    # top-of-function guard already rejected anything else).
    total = zf_hourly.sum(axis=1, skipna=False)
    shares = zf_hourly.div(total, axis=0)
    # A zero (or near-zero) four-zone total -- e.g. negative-value
    # cancellation across zones -- produces +/-inf shares from a nonzero
    # numerator divided by zero. inf is NOT NaN, so it would silently sail
    # past every downstream ``isna()`` fail-safe check; catch it explicitly
    # here, at the source, with a clear diagnosis.
    if np.isinf(shares.to_numpy(dtype="float64")).any():
        raise LoadLagError(
            "load_lags (zone_shares): share computation produced +/-inf "
            "values (a zero or near-zero four-zone total, most likely from "
            "negative-value cancellation across zones); cannot safely "
            "compute shares."
        )
    keep_cols = [c for c in ZONE_LOAD_COLUMNS if c != _DROPPED_SHARE_ZONE]
    source_hourly = shares[keep_cols]
    name_prefix = {c: f"share_{c.removeprefix('load_')}" for c in keep_cols}
    return source_hourly, name_prefix


def build_load_lag_features(
    *,
    target_series: pd.Series,
    zone_frame: Optional[pd.DataFrame],
    index: pd.DatetimeIndex,
    data_end: pd.Timestamp,
    cov_end: pd.Timestamp,
    load_lag_hours: Sequence[int],
    load_lag_sources: str = "total",
    max_staleness_hours: int = 48,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build NaN-free lagged-load exogenous columns aligned to *index*.

    Pure-pandas builder (no I/O). Resamples the requested source to an hourly
    mean, computes zone shares first when requested, then shifts each source
    column by each entry of *load_lag_hours* via
    ``series.shift(freq=pd.Timedelta(hours=L))`` and reindexes onto *index*
    (``[data_start, cov_end]``). Only the unavoidable leading warm-up NaN run
    (``t < index[0] + max(load_lag_hours)``, bounded to ``t <= data_end`` so
    it can never reach into or draw from the forecast horizon) is
    back-filled from historical values only; the forecast-horizon slice
    ``(data_end, cov_end]`` is hard-checked for NaN and raises
    :class:`LoadLagError` rather than silently filling across the leakage
    boundary, as is any remaining NaN elsewhere in the frame. Each source
    column is also checked for staleness individually: its last valid
    observation must not lag *data_end* by more than *max_staleness_hours*
    (a per-column check, so one quietly-stopped zone cannot hide behind
    still-live siblings).

    Args:
        target_series: The primary target series (``df_pipeline[target]``),
            used when ``load_lag_sources == "total"``.
        zone_frame: The four-TSO-zone frame from
            ``load_interim(mode="four_zone")`` (columns
            :data:`ZONE_LOAD_COLUMNS`), used when *load_lag_sources* is
            ``"zones"`` or ``"zone_shares"``. ``None`` when
            *load_lag_sources* is ``"total"``.
        index: The target output ``DatetimeIndex`` (typically
            ``exogenous_features.index``, spanning ``[data_start, cov_end]``).
        data_end: Last timestamp of observed (historical) data; the forecast
            horizon is ``(data_end, cov_end]``.
        cov_end: Inclusive end of the covariate window (spans the forecast
            horizon beyond *data_end*).
        load_lag_hours: Lag lengths in hours; every entry should be ``>= 168``
            (enforced by ``ConfigMulti`` validation, not re-checked here) so
            the lagged value is already published at forecast time.
        load_lag_sources: One of ``"total"``, ``"zones"``, ``"zone_shares"``.
            Defaults to ``"total"``.
        max_staleness_hours: Maximum age, in hours, the source's last valid
            observation may lag behind *data_end*. Defaults to ``48``.

    Returns:
        tuple[pd.DataFrame, list[str]]: ``(lag_frame, lag_names)`` — a
        ``float32``, NaN-free frame indexed exactly by *index*, and the
        ordered list of its column names.

    Raises:
        LoadLagError: If *load_lag_sources* is invalid, ``zone_frame`` is
            required but missing (or missing a required zone column), any
            source column is stale, the ``"zone_shares"`` division produces
            ``+/-inf`` (zero or near-zero four-zone total), or any column
            contains NaN after the (history-bounded) warm-up backfill
            (including within the forecast-horizon slice).

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.load_lags import (
            build_load_lag_features,
        )

        # 300 hours of history + a 6-hour forecast horizon.
        idx = pd.date_range("2024-01-01", periods=306, freq="h", tz="UTC")
        data_end = idx[299]
        cov_end = idx[-1]
        target = pd.Series(range(300), index=idx[:300], dtype="float64", name="load")

        lag_frame, lag_names = build_load_lag_features(
            target_series=target,
            zone_frame=None,
            index=idx,
            data_end=data_end,
            cov_end=cov_end,
            load_lag_hours=(168,),
            load_lag_sources="total",
            max_staleness_hours=48,
        )
        print(lag_names)
        assert lag_names == ["load_lag_168"]
        assert not lag_frame.isna().any().any()
        # Leakage check: value at t equals the raw series at t - 168h.
        t = idx[250]
        assert lag_frame.loc[t, "load_lag_168"] == target.loc[t - pd.Timedelta(hours=168)]
        ```
    """
    if not load_lag_hours:
        raise LoadLagError("load_lag_hours must be a non-empty sequence of ints.")

    source_hourly, name_prefix = _resolve_source(
        target_series=target_series,
        zone_frame=zone_frame,
        load_lag_sources=load_lag_sources,
        index=index,
    )

    data_end_ts = _as_timestamp(data_end, like_index=index)
    cov_end_ts = _as_timestamp(cov_end, like_index=index)

    # Staleness gate: independent of the mechanical NaN check below — a
    # 168h+ lag may not actually *need* very recent source data, but a source
    # that has silently stopped updating is a data-quality signal in its own
    # right and must not be masked. Checked PER COLUMN: an aggregate
    # ``dropna(how="all")`` would let one quietly-stopped zone hide behind
    # its still-live siblings (a row survives the "all" drop as long as at
    # least one column has a value), so each column's own last valid
    # observation is checked and named individually in the error.
    for col in source_hourly.columns:
        col_valid = source_hourly[col].dropna()
        if col_valid.empty:
            raise LoadLagError(
                f"load_lags ({load_lag_sources}): source column {col!r} has "
                "no valid observations."
            )
        col_last_ts = col_valid.index.max()
        staleness = data_end_ts - col_last_ts
        if staleness > pd.Timedelta(hours=max_staleness_hours):
            raise LoadLagError(
                f"load_lags ({load_lag_sources}): source column {col!r} is "
                f"stale — last observation {col_last_ts} is {staleness} "
                f"behind data_end {data_end_ts} (max {max_staleness_hours}h "
                "allowed)."
            )

    max_lag_hours = max(int(h) for h in load_lag_hours)
    warmup_end = index[0] + pd.Timedelta(hours=max_lag_hours)
    # Bound the warm-up window to HISTORY ONLY (never past data_end). Without
    # the ``index <= data_end_ts`` bound, a short index[0]->data_end span
    # relative to max_lag_hours would let the mask reach into the
    # forecast-horizon slice (data_end, cov_end] too; the bfill below would
    # then be free to silently patch a genuine horizon coverage gap with a
    # value pulled forward from a LATER position in the (short) index,
    # before the horizon hard-check further down ever runs -- a wrong value
    # with no exception (reviewer-reproduced regression).
    warmup_mask = (index < warmup_end) & (index <= data_end_ts)

    result = pd.DataFrame(index=index)
    names: List[str] = []
    for lag_hours in load_lag_hours:
        shift = pd.Timedelta(hours=int(lag_hours))
        for col in source_hourly.columns:
            shifted = source_hourly[col].shift(freq=shift).reindex(index)
            if warmup_mask.any():
                # The backfill source must itself never reach past data_end:
                # mask out anything at/after the horizon BEFORE bfilling, so
                # a warm-up cell can only ever be filled from a genuine
                # historical value, never from a horizon-side position.
                historical_only = shifted.where(index <= data_end_ts)
                filled = historical_only.bfill()
                shifted = shifted.where(~warmup_mask, filled)
            colname = f"{name_prefix[col]}_lag_{int(lag_hours)}"
            result[colname] = shifted
            names.append(colname)

    # Forecast-horizon slice must never be silently filled: check it first
    # so a stale/short source is reported against the leakage-critical span.
    horizon_mask = (index > data_end_ts) & (index <= cov_end_ts)
    if horizon_mask.any():
        horizon_bad = result.loc[horizon_mask].isna().any()
        if horizon_bad.any():
            bad_cols = horizon_bad[horizon_bad].index.tolist()
            raise LoadLagError(
                f"load_lags ({load_lag_sources}): forecast-horizon slice "
                f"({data_end_ts}, {cov_end_ts}] contains NaN in column(s) "
                f"{bad_cols}; refusing to fill across the leakage boundary."
            )

    if result.isna().any().any():
        bad_cols = result.columns[result.isna().any()].tolist()
        raise LoadLagError(
            f"load_lags ({load_lag_sources}): column(s) {bad_cols} contain "
            "NaN outside the warm-up window; the source series does not "
            "cover the requested range."
        )

    return result.astype("float32"), names

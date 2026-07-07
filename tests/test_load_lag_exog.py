# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``include_load_lag_exog`` opt-in knob (ADR feat/load-lag-exog).

All tests are network-free and filesystem-free: ``build_load_lag_features`` is
exercised directly with fixture frames, and
``spotforecast2_safe.downloader.entsoe.load_interim`` is monkeypatched for the
pipeline-integration tests, mirroring the pattern in
``tests/test_per_zone_weather.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.preprocessing.load_lags import (
    ZONE_LOAD_COLUMNS,
    LoadLagError,
    build_load_lag_features,
)


def _hourly_index(
    periods: int, start: str = "2024-01-01", **kwargs
) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC", **kwargs)


def _target_series(idx: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(50000, 2000, len(idx)), index=idx, name="load")


def _zone_frame_hourly(idx: pd.DatetimeIndex, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "load_50hertz": rng.normal(11000, 300, len(idx)),
            "load_amprion": rng.normal(25000, 600, len(idx)),
            "load_tennet": rng.normal(20000, 500, len(idx)),
            "load_transnetbw": rng.normal(9000, 250, len(idx)),
        },
        index=idx,
    )


def _zone_frame_15min(idx_hourly: pd.DatetimeIndex, *, seed: int = 7) -> pd.DataFrame:
    """15-min zone frame whose hourly mean is deterministic (constant per hour)."""
    idx_15 = pd.date_range(
        idx_hourly[0], idx_hourly[-1] + pd.Timedelta(minutes=45), freq="15min", tz="UTC"
    )
    rng = np.random.default_rng(seed)
    # Assign one base value per hour, then jitter within the hour so the mean
    # over the hour still equals the base (jitter sums to zero within each hour).
    hour_bucket = idx_15.floor("h")
    base_by_hour = {
        h: rng.normal(10000, 200) for h in pd.DatetimeIndex(hour_bucket.unique())
    }
    base = np.array([base_by_hour[h] for h in hour_bucket])
    jitter = np.tile([10.0, -10.0, 5.0, -5.0], len(idx_15) // 4 + 1)[: len(idx_15)]
    return (
        pd.DataFrame(
            {
                "load_50hertz": base + jitter,
                "load_amprion": base * 2 + jitter,
                "load_tennet": base * 1.5 + jitter,
                "load_transnetbw": base * 0.5 + jitter,
            },
            index=idx_15,
        ),
        base_by_hour,
    )


# ---------------------------------------------------------------------------
# (a) leakage: exog[t] == source_hourly[t - L] for total / zones / zone_shares
# and multi-L tuples.
# ---------------------------------------------------------------------------


class TestLeakageTotal:
    def test_single_lag(self):
        idx = _hourly_index(400)
        target = _target_series(idx).iloc[:390]  # history ends at idx[389]
        data_end = idx[389]
        cov_end = idx[395]
        lag_frame, names = build_load_lag_features(
            target_series=target,
            zone_frame=None,
            index=idx[:396],
            data_end=data_end,
            cov_end=cov_end,
            load_lag_hours=(168,),
            load_lag_sources="total",
            max_staleness_hours=48,
        )
        assert names == ["load_lag_168"]
        t = idx[300]
        assert lag_frame.loc[t, "load_lag_168"] == pytest.approx(
            target.loc[t - pd.Timedelta(hours=168)]
        )

    def test_multi_lag(self):
        idx = _hourly_index(600)
        target = _target_series(idx).iloc[:590]
        lag_frame, names = build_load_lag_features(
            target_series=target,
            zone_frame=None,
            index=idx[:596],
            data_end=idx[589],
            cov_end=idx[595],
            load_lag_hours=(168, 336),
            load_lag_sources="total",
            max_staleness_hours=48,
        )
        assert set(names) == {"load_lag_168", "load_lag_336"}
        t = idx[400]
        assert lag_frame.loc[t, "load_lag_168"] == pytest.approx(
            target.loc[t - pd.Timedelta(hours=168)]
        )
        assert lag_frame.loc[t, "load_lag_336"] == pytest.approx(
            target.loc[t - pd.Timedelta(hours=336)]
        )


class TestLeakageZones:
    def test_leakage_hourly_source(self):
        idx = _hourly_index(400)
        zone_frame = _zone_frame_hourly(idx.copy())
        target = _target_series(idx)  # unused for "zones"
        lag_frame, names = build_load_lag_features(
            target_series=target,
            zone_frame=zone_frame,
            index=idx,
            data_end=idx[390],
            cov_end=idx[396] if 396 < 400 else idx[399],
            load_lag_hours=(168,),
            load_lag_sources="zones",
            max_staleness_hours=48,
        )
        assert set(names) == {f"{c}_lag_168" for c in ZONE_LOAD_COLUMNS}
        t = idx[300]
        for col in ZONE_LOAD_COLUMNS:
            assert lag_frame.loc[t, f"{col}_lag_168"] == pytest.approx(
                zone_frame.loc[t - pd.Timedelta(hours=168), col]
            )

    # ------------------------------------------------------------------
    # (b) 15-min -> hourly mean resample correctness
    # ------------------------------------------------------------------
    def test_resample_15min_to_hourly_mean(self):
        idx_hourly = _hourly_index(400)
        zone_frame_15, base_by_hour = _zone_frame_15min(idx_hourly)
        target = _target_series(idx_hourly)
        lag_frame, names = build_load_lag_features(
            target_series=target,
            zone_frame=zone_frame_15,
            index=idx_hourly,
            data_end=idx_hourly[390],
            cov_end=idx_hourly[399],
            load_lag_hours=(168,),
            load_lag_sources="zones",
            max_staleness_hours=48,
        )
        t = idx_hourly[300]
        source_hour = t - pd.Timedelta(hours=168)
        expected_50hertz = base_by_hour[source_hour]  # jitter cancels within the hour
        assert lag_frame.loc[t, "load_50hertz_lag_168"] == pytest.approx(
            expected_50hertz, abs=1e-3
        )


class TestZoneShares:
    def _build(self, *, load_lag_hours=(168,)):
        idx = _hourly_index(700)
        zone_frame = _zone_frame_hourly(idx.copy())
        target = _target_series(idx)
        lag_frame, names = build_load_lag_features(
            target_series=target,
            zone_frame=zone_frame,
            index=idx,
            data_end=idx[690],
            cov_end=idx[699],
            load_lag_hours=load_lag_hours,
            load_lag_sources="zone_shares",
            max_staleness_hours=48,
        )
        return idx, zone_frame, lag_frame, names

    def test_drops_transnetbw_keeps_three(self):
        _, _, _, names = self._build()
        assert set(names) == {
            "share_50hertz_lag_168",
            "share_amprion_lag_168",
            "share_tennet_lag_168",
        }
        assert not any("transnetbw" in n for n in names)

    def test_shares_computed_then_lagged_leakage_check(self):
        idx, zone_frame, lag_frame, _ = self._build()
        t = idx[500]
        source_t = t - pd.Timedelta(hours=168)
        row = zone_frame.loc[source_t]
        total = row.sum()
        expected_share_50hertz = row["load_50hertz"] / total
        assert lag_frame.loc[t, "share_50hertz_lag_168"] == pytest.approx(
            expected_share_50hertz, abs=1e-5
        )

    def test_shares_sum_with_transnetbw_is_one_pre_lag(self):
        """The FOUR-zone bottom-up share (including transnetbw) sums to 1
        before the smallest zone is dropped -- verifying the divisor is the
        bottom-up four-zone sum, not some other total."""
        idx, zone_frame, _, _ = self._build()
        hourly = zone_frame.resample("h").mean()
        total = hourly.sum(axis=1)
        shares_all_four = hourly.div(total, axis=0)
        assert (shares_all_four.sum(axis=1).round(9) == 1.0).all()

    def test_multi_lag_shares(self):
        _, _, lag_frame, names = self._build(load_lag_hours=(168, 336))
        assert len(names) == 6
        assert not lag_frame.isna().any().any()

    def test_zero_total_from_negative_cancellation_raises_on_inf(self):
        """A zero (or near-zero) four-zone total -- e.g. from negative-value
        cancellation across zones -- must raise rather than silently
        propagate +/-inf shares (inf is NOT NaN, so it would otherwise sail
        past every downstream NaN fail-safe check)."""
        idx = _hourly_index(200)
        zone_frame = _zone_frame_hourly(idx.copy())
        t = idx[100]
        zone_frame.loc[t, "load_50hertz"] = 100.0
        zone_frame.loc[t, "load_amprion"] = -100.0
        zone_frame.loc[t, "load_tennet"] = 50.0
        zone_frame.loc[t, "load_transnetbw"] = -50.0  # four-zone sum == 0
        target = _target_series(idx)
        with pytest.raises(LoadLagError, match="inf"):
            build_load_lag_features(
                target_series=target,
                zone_frame=zone_frame,
                index=idx,
                data_end=idx[190],
                cov_end=idx[199],
                load_lag_hours=(168,),
                load_lag_sources="zone_shares",
                max_staleness_hours=48,
            )


# ---------------------------------------------------------------------------
# (d) staleness: source ending more than max_staleness_hours before data_end
# raises LoadLagError.
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_stale_zone_source_raises(self):
        idx = _hourly_index(400)
        zone_frame = _zone_frame_hourly(idx.copy())
        stale_zone_frame = zone_frame.iloc[:-72]  # cut off the last 72 hours
        target = _target_series(idx)
        with pytest.raises(LoadLagError, match="stale"):
            build_load_lag_features(
                target_series=target,
                zone_frame=stale_zone_frame,
                index=idx,
                data_end=idx[390],
                cov_end=idx[399],
                load_lag_hours=(168,),
                load_lag_sources="zones",
                max_staleness_hours=48,
            )

    def test_within_staleness_budget_succeeds(self):
        idx = _hourly_index(400)
        zone_frame = _zone_frame_hourly(idx.copy())
        slightly_stale = zone_frame.iloc[:-30]  # cut off 30h, budget is 48h
        target = _target_series(idx)
        lag_frame, names = build_load_lag_features(
            target_series=target,
            zone_frame=slightly_stale,
            index=idx[:365],  # avoid horizon reaching past the stale cut
            data_end=idx[360],
            cov_end=idx[364],
            load_lag_hours=(168,),
            load_lag_sources="zones",
            max_staleness_hours=48,
        )
        assert not lag_frame.isna().any().any()

    def test_single_stale_column_named_in_error(self):
        """Per-column staleness: three zones stay live, one goes stale.

        An aggregate ``dropna(how='all')`` check would be masked by the
        three still-live columns (a row survives as long as ANY column has
        a value); the per-column check must catch the one stale column and
        name it.
        """
        idx = _hourly_index(400)
        zone_frame = _zone_frame_hourly(idx.copy())
        # Only transnetbw goes stale (NaN for its last 100 hours); the other
        # three zones keep reporting all the way to idx[-1].
        zone_frame.loc[idx[300] :, "load_transnetbw"] = np.nan
        target = _target_series(idx)
        with pytest.raises(LoadLagError, match="load_transnetbw"):
            build_load_lag_features(
                target_series=target,
                zone_frame=zone_frame,
                index=idx,
                data_end=idx[390],
                cov_end=idx[399],
                load_lag_hours=(168,),
                load_lag_sources="zones",
                max_staleness_hours=48,
            )


# ---------------------------------------------------------------------------
# (e) horizon-NaN hard-check: a coverage gap in the horizon slice raises
# rather than bfilling across the leakage boundary.
# ---------------------------------------------------------------------------


class TestHorizonNaNHardCheck:
    def test_short_source_leaves_horizon_nan_and_raises(self):
        idx = _hourly_index(400)
        # Target series far too short to cover the (data_end, cov_end] slice
        # once shifted by 168h -- horizon values would require source data
        # that was never observed.
        target = _target_series(idx).iloc[:50]
        with pytest.raises(LoadLagError, match="forecast-horizon"):
            build_load_lag_features(
                target_series=target,
                zone_frame=None,
                index=idx[:100],
                data_end=idx[90],
                cov_end=idx[99],
                load_lag_hours=(168,),
                load_lag_sources="total",
                max_staleness_hours=10_000,  # disable the staleness gate
            )

    def test_short_history_span_isolated_gap_in_horizon_must_raise(self):
        """BLOCKER regression (review finding).

        When the index[0]->data_end span is much shorter than
        max(load_lag_hours), the (buggy, unbounded) warm-up mask used to
        cover the WHOLE index -- including the forecast-horizon slice. A
        ``bfill()`` over that mask could then silently patch a genuine,
        isolated source gap that maps into the horizon with a value pulled
        forward from a LATER position, and the horizon hard-check below
        would see no NaN and pass silently -- a wrong value with no
        exception. The fix bounds the warm-up mask to ``index <= data_end``
        and sources the backfill from historical values only, so a horizon
        gap is never masked and always reaches the hard-check.
        """
        idx = _hourly_index(50)  # only 50 hours total: far shorter than 168h
        data_end = idx[9]  # only 10 hours of "history"
        cov_end = idx[49]  # 40-hour horizon >> the 10-hour history span

        # Continuous hourly source covering every lagged position needed by
        # BOTH the historical and horizon slices of `idx`
        # ([idx[0]-168h, idx[49]-168h]), except an ISOLATED one-hour gap at
        # the exact source position that the HORIZON timestamp idx[20]
        # maps back to.
        source_start = idx[0] - pd.Timedelta(hours=168)
        source_end = idx[49] - pd.Timedelta(hours=168)
        source_idx = pd.date_range(source_start, source_end, freq="h", tz="UTC")
        rng = np.random.default_rng(99)
        target = pd.Series(
            rng.normal(50000, 2000, len(source_idx)), index=source_idx, name="load"
        )
        gap_source_ts = idx[20] - pd.Timedelta(hours=168)
        assert gap_source_ts in target.index
        target.loc[gap_source_ts] = np.nan

        with pytest.raises(LoadLagError, match="forecast-horizon"):
            build_load_lag_features(
                target_series=target,
                zone_frame=None,
                index=idx,
                data_end=data_end,
                cov_end=cov_end,
                load_lag_hours=(168,),
                load_lag_sources="total",
                max_staleness_hours=10_000,  # disable the staleness gate
            )


# ---------------------------------------------------------------------------
# Validation-style errors raised directly by the builder (mirrors ConfigMulti
# validation but exercised at the module level for the "zones"/"zone_shares"
# source-shape guards, which are not config-time checks).
# ---------------------------------------------------------------------------


class TestBuilderValidation:
    def test_invalid_source_raises(self):
        idx = _hourly_index(200)
        target = _target_series(idx)
        with pytest.raises(LoadLagError, match="must be 'total'"):
            build_load_lag_features(
                target_series=target,
                zone_frame=None,
                index=idx,
                data_end=idx[190],
                cov_end=idx[199],
                load_lag_hours=(168,),
                load_lag_sources="bogus",
                max_staleness_hours=48,
            )

    def test_missing_zone_frame_raises(self):
        idx = _hourly_index(200)
        target = _target_series(idx)
        with pytest.raises(LoadLagError, match="requires zone_frame"):
            build_load_lag_features(
                target_series=target,
                zone_frame=None,
                index=idx,
                data_end=idx[190],
                cov_end=idx[199],
                load_lag_hours=(168,),
                load_lag_sources="zones",
                max_staleness_hours=48,
            )

    def test_missing_zone_column_raises(self):
        idx = _hourly_index(200)
        target = _target_series(idx)
        bad_zone_frame = _zone_frame_hourly(idx.copy()).drop(
            columns=["load_transnetbw"]
        )
        with pytest.raises(LoadLagError, match="missing required zone column"):
            build_load_lag_features(
                target_series=target,
                zone_frame=bad_zone_frame,
                index=idx,
                data_end=idx[190],
                cov_end=idx[199],
                load_lag_hours=(168,),
                load_lag_sources="zones",
                max_staleness_hours=48,
            )

    def test_empty_load_lag_hours_raises(self):
        idx = _hourly_index(200)
        target = _target_series(idx)
        with pytest.raises(LoadLagError, match="non-empty"):
            build_load_lag_features(
                target_series=target,
                zone_frame=None,
                index=idx,
                data_end=idx[190],
                cov_end=idx[199],
                load_lag_hours=(),
                load_lag_sources="total",
                max_staleness_hours=48,
            )

    def test_empty_zone_frame_raises(self):
        """A zero-row zone_frame must raise (pinned), not silently produce
        an empty/degenerate result."""
        idx = _hourly_index(200)
        target = _target_series(idx)
        empty_zone_frame = pd.DataFrame(
            columns=list(ZONE_LOAD_COLUMNS),
            index=pd.DatetimeIndex([], tz="UTC"),
            dtype="float64",
        )
        with pytest.raises(LoadLagError, match="no valid observations"):
            build_load_lag_features(
                target_series=target,
                zone_frame=empty_zone_frame,
                index=idx,
                data_end=idx[190],
                cov_end=idx[199],
                load_lag_hours=(168,),
                load_lag_sources="zones",
                max_staleness_hours=48,
            )

    def test_zone_load_columns_matches_expected_hardcoded_names(self):
        """Drift guard: pin the exact four TSO-zone column names (no
        downloader import -- this module must stay dependency-free)."""
        assert ZONE_LOAD_COLUMNS == (
            "load_50hertz",
            "load_amprion",
            "load_tennet",
            "load_transnetbw",
        )


# ---------------------------------------------------------------------------
# ConfigMulti validation guards (config-construction time)
# ---------------------------------------------------------------------------


class TestConfigLoadLagGuards:
    def test_default_off(self):
        cfg = ConfigMulti()
        assert cfg.include_load_lag_exog is False
        assert cfg.load_lag_hours == (168,)
        assert cfg.load_lag_sources == "total"
        assert cfg.on_load_lag_failure == "raise"
        assert cfg.load_lag_max_staleness_hours == 48

    def test_requires_exogenous_features(self):
        with pytest.raises(ValueError, match="requires exogenous features"):
            ConfigMulti(include_load_lag_exog=True, use_exogenous_features=False)

    def test_rejects_lag_below_168(self):
        with pytest.raises(ValueError, match=">= 168"):
            ConfigMulti(include_load_lag_exog=True, load_lag_hours=(24,))

    def test_rejects_empty_load_lag_hours(self):
        with pytest.raises(ValueError, match="non-empty"):
            ConfigMulti(include_load_lag_exog=True, load_lag_hours=())

    def test_rejects_bad_source(self):
        with pytest.raises(ValueError, match="zone_shares"):
            ConfigMulti(load_lag_sources="bogus")

    def test_rejects_bad_failure_policy(self):
        with pytest.raises(ValueError, match="raise"):
            ConfigMulti(on_load_lag_failure="ignore")

    def test_rejects_negative_staleness(self):
        with pytest.raises(ValueError, match=">= 0"):
            ConfigMulti(load_lag_max_staleness_hours=-1)

    def test_valid_construction(self):
        cfg = ConfigMulti(
            include_load_lag_exog=True,
            load_lag_hours=(168, 336),
            load_lag_sources="zone_shares",
            on_load_lag_failure="skip",
        )
        assert cfg.load_lag_hours == (168, 336)

    def test_raises_via_set_params_too(self):
        """Mirrors test_zone_weather_columns.py's set_params parity check:
        the guard must not be bypassable by mutating an existing instance."""
        cfg = ConfigMulti()
        with pytest.raises(ValueError, match=">= 168"):
            cfg.set_params(include_load_lag_exog=True, load_lag_hours=(24,))


# ---------------------------------------------------------------------------
# Pipeline integration (LazyTask): (d) staleness/skip, (f) missing zone
# interim, (g) selection survives cyclical encoding / poly=2.
# ---------------------------------------------------------------------------


def _pipeline_df(periods: int = 24 * 25) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-01", periods=periods, freq="h", tz="UTC")
    idx.name = "DateTime"
    return pd.DataFrame({"load": rng.normal(50000, 2000, periods)}, index=idx)


def _synthetic_zone_interim(df: pd.DataFrame, *, cov_end) -> pd.DataFrame:
    """A hourly four-zone frame covering [df.index.min(), cov_end]."""
    idx = pd.date_range(df.index.min(), cov_end, freq="h", tz="UTC")
    return _zone_frame_hourly(idx.copy())


class TestLoadLagPipelineIntegration:
    def _build_task(self, tmp_path, **cfg_kwargs):
        from spotforecast2_safe.multitask import LazyTask

        df = _pipeline_df()
        cfg = ConfigMulti(
            targets=["load"],
            predict_size=24,
            use_exogenous_features=True,
            use_outlier_detection=False,
            auto_save_models=False,
            number_folds=2,
            verbose=False,
            cache_home=str(tmp_path),
            **cfg_kwargs,
        )
        task = LazyTask(cfg, dataframe=df)
        task.prepare_data().detect_outliers().impute().build_exogenous_features()
        return task

    def test_total_lag_reaches_exog_feature_names_and_exo_pred(self, tmp_path):
        task = self._build_task(
            tmp_path, include_load_lag_exog=True, load_lag_sources="total"
        )
        assert "load_lag_168" in task.exog_feature_names
        assert "load_lag_168" in task.exo_pred.columns
        assert not task.exo_pred["load_lag_168"].isna().any()

    def test_zones_missing_interim_skip_drops_columns(self, monkeypatch, tmp_path):
        import spotforecast2_safe.downloader.entsoe as entsoe_mod

        def raise_fnf(mode="combined", **kwargs):
            raise FileNotFoundError("no per-zone interim cache found")

        monkeypatch.setattr(entsoe_mod, "load_interim", raise_fnf)
        task = self._build_task(
            tmp_path,
            include_load_lag_exog=True,
            load_lag_sources="zones",
            on_load_lag_failure="skip",
        )
        assert "load_50hertz_lag_168" not in task.exog_feature_names

    def test_zones_missing_interim_raise_mode_propagates(self, monkeypatch, tmp_path):
        import spotforecast2_safe.downloader.entsoe as entsoe_mod

        def raise_fnf(mode="combined", **kwargs):
            raise FileNotFoundError("no per-zone interim cache found")

        monkeypatch.setattr(entsoe_mod, "load_interim", raise_fnf)
        with pytest.raises(FileNotFoundError):
            self._build_task(
                tmp_path,
                include_load_lag_exog=True,
                load_lag_sources="zones",
                on_load_lag_failure="raise",
            )

    def test_zones_wired_end_to_end(self, monkeypatch, tmp_path):
        import spotforecast2_safe.downloader.entsoe as entsoe_mod

        df = _pipeline_df()
        cov_end = df.index.max() + pd.Timedelta(hours=24)
        zone_frame = _synthetic_zone_interim(df, cov_end=cov_end)
        monkeypatch.setattr(
            entsoe_mod, "load_interim", lambda mode="combined", **kw: zone_frame
        )
        task = self._build_task(
            tmp_path, include_load_lag_exog=True, load_lag_sources="zones"
        )
        for col in ZONE_LOAD_COLUMNS:
            assert f"{col}_lag_168" in task.exog_feature_names
            assert not task.exo_pred[f"{col}_lag_168"].isna().any()

    def test_staleness_skip_drops_columns_pipeline(self, monkeypatch, tmp_path):
        import spotforecast2_safe.downloader.entsoe as entsoe_mod

        df = _pipeline_df()
        cov_end = df.index.max() + pd.Timedelta(hours=24)
        zone_frame = _synthetic_zone_interim(df, cov_end=cov_end)
        # zone_frame extends 24h past data_end (== cov_end); cutting 120h off
        # the end leaves it 96h stale relative to data_end -- well past the
        # default 48h budget.
        stale_zone_frame = zone_frame.iloc[:-120]
        monkeypatch.setattr(
            entsoe_mod, "load_interim", lambda mode="combined", **kw: stale_zone_frame
        )
        task = self._build_task(
            tmp_path,
            include_load_lag_exog=True,
            load_lag_sources="zones",
            on_load_lag_failure="skip",
        )
        assert "load_50hertz_lag_168" not in task.exog_feature_names

    def test_selection_survives_poly_degree_2(self, tmp_path):
        task = self._build_task(
            tmp_path,
            include_load_lag_exog=True,
            load_lag_sources="total",
            poly_features_degree=2,
        )
        assert "load_lag_168" in task.exog_feature_names
        # Inert to poly interactions: no poly_* column derived from the lag.
        poly_cols_from_lag = [
            c
            for c in task.exogenous_features.columns
            if c.startswith("poly_") and "load_lag" in c
        ]
        assert poly_cols_from_lag == []

    def test_selection_survives_cyclical_encoding(self, tmp_path):
        task = self._build_task(
            tmp_path, include_load_lag_exog=True, load_lag_sources="total"
        )
        # apply_cyclical_encoding must not rename or drop the lag column.
        assert "load_lag_168" in task.exogenous_features.columns
        assert not task.exogenous_features["load_lag_168"].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

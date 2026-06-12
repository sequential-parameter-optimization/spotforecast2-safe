# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pluggable exogenous-feature providers.

This module turns "add another exogenous driver" into a small, mechanical
change. Each driver is an :class:`ExogFeatureProvider` — an object that, given
the hourly target index, returns a numeric feature frame aligned to that
index, NaN-free within the validated window (the full index unless a
``provider_window`` is set). Providers are registered in :data:`EXOG_PROVIDER_REGISTRY` against a
single boolean configuration flag, so :class:`spotforecast2_safe.configurator.config_entsoe.ConfigEntsoe`
(and ``ConfigMulti``) can switch each one on or off.

The providers are appended by
`spotforecast2_safe.preprocessing.exog_builder.ExogBuilder`, which already builds
the calendar / cyclical / holiday block. Adding a new driver therefore costs:

1. a provider class implementing :meth:`ExogFeatureProvider.build`,
2. one entry in :data:`EXOG_PROVIDER_REGISTRY` mapping a config flag to it,
3. one boolean field on the config classes (mirrored in ``_PARAM_NAMES``).

Design rules (consistent with the safety-critical contract):

- **Fail-safe.** A provider that cannot supply a value for every validated
  timestamp raises :class:`ExogProviderError` rather than silently imputing.
  ``ExogBuilder`` then either re-raises (``on_provider_failure="raise"``) or
  logs and skips that provider (``"skip"``). Values are never silently
  fabricated: the only imputation is the opt-in, bounded gap healing
  (``max_gap`` / ``max_tail_gap``), which is all-or-nothing per gap and logged
  at WARNING with count and span. The trailing-edge budget ``max_tail_gap``
  extends the interior budget specifically for the NaN run that touches
  ``index[-1]``; this matches the ENTSO-E day-ahead publication frontier, where
  the last published vintage is zero-order-held forward without introducing
  future leakage (CR-3-clean).
- **Leakage-aware (CR-3).** Only inputs genuinely available at forecast time are
  admissible: ENTSO-E *day-ahead* forecasts/price (published D-1) and the static
  published COVID vintage — never a realised quantity for the target day.
- **Deterministic.** Same input data and index produce the same output, bit for
  bit; no RNG, no wall-clock, stable column order.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Union

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
    from spotforecast2_safe.configurator.config_multi import ConfigMulti

logger = logging.getLogger(__name__)

DataHome = Optional[Union[str, Path]]


class ExogProviderError(RuntimeError):
    """A provider could not produce features covering the requested index.

    Raised when the backing data is missing, malformed, or does not cover every
    requested timestamp. ``ExogBuilder`` translates this into either a hard
    failure or a skipped provider depending on its ``on_provider_failure``
    policy, so the error is the single fail-safe signal for "this exogenous
    input is unavailable".

    Examples:
        ```{python}
        from spotforecast2_safe.preprocessing.exog_providers import ExogProviderError

        try:
            raise ExogProviderError("my_provider: data not found.")
        except ExogProviderError as exc:
            print(type(exc).__name__, str(exc))
        assert issubclass(ExogProviderError, RuntimeError)
        ```
    """


def _align_to_index(
    frame: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    provider: str,
    max_gap: int = 0,
    max_tail_gap: int = 0,
    validate_index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Reindex *frame* onto *index*, optionally healing small gaps (fail-safe).

    Harmonises timezone awareness with *index*, drops duplicate timestamps
    (keeping the last), reindexes exactly onto *index*, and refuses to return a
    frame that still contains NaN within the validated window — that would mean
    the provider cannot cover the requested range. The result is cast to
    ``float32`` to match the dtype the recursive forecaster expects for
    exogenous columns.

    Gap-healing uses a split budget: the contiguous NaN run that contains
    ``index[-1]`` (the trailing-edge run) receives
    ``max(max_gap, max_tail_gap)`` as its budget, while all other runs keep
    ``max_gap``. Setting ``max_gap=0, max_tail_gap>0`` therefore heals *only*
    the trailing edge — the canonical use case is zero-order-holding the last
    published ENTSO-E day-ahead vintage forward to the forecast horizon without
    touching interior gaps (CR-3-clean, no future leakage). Healing is
    all-or-nothing per run: a run whose length exceeds its budget is left
    entirely NaN and will trip the NaN gate if it intersects the validated
    window.

    Args:
        frame: Source feature frame with a ``DatetimeIndex``.
        index: Target hourly index the features must align to.
        provider: Provider name, used in log messages and the error.
        max_gap: Maximum length, in hours, of a contiguous interior (or
            leading/trailing) NaN run that will be healed. Interior gaps are
            time-interpolated; leading gaps are back-filled; trailing gaps are
            forward-filled. ``0`` (default) keeps the strict fail-safe (any
            gap raises). Healed runs are logged at WARNING with count and span.
            When ``max_gap=0``, neither interpolation nor back-fill runs
            regardless of ``max_tail_gap`` — leading gaps cannot be healed
            in that configuration.
        max_tail_gap: Extended budget, in hours, applied exclusively to the
            NaN run containing ``index[-1]`` (the trailing-edge run). The
            effective tail budget is ``max(max_gap, max_tail_gap)``, so a
            non-zero ``max_gap`` already heals the tail up to ``max_gap``; set
            ``max_tail_gap`` higher only when a longer trailing void is
            acceptable. ``0`` (default) leaves the tail budget equal to
            ``max_gap``, reproducing 16.1.x behaviour exactly.
        validate_index: If given, the NaN gate checks only the intersection
            of *index* with this index. Timestamps outside the validation
            window may carry NaN and are still returned (as ``float32`` NaN).
            ``None`` (default) validates the full *index*, matching prior
            behaviour exactly.

    Returns:
        pd.DataFrame: *frame* reindexed onto *index*, ``float32``. NaN-free
        within the validated window (after healing when ``max_gap > 0`` or
        ``max_tail_gap > 0``); out-of-window cells remain NaN when
        *validate_index* is supplied.

    Raises:
        ExogProviderError: If *frame* is not datetime-indexed or any requested
            timestamp in the validated window has no value after healing.
    """
    f = frame.copy()
    if not isinstance(f.index, pd.DatetimeIndex):
        raise ExogProviderError(f"{provider}: source data is not datetime-indexed.")

    if index.tz is not None:
        f.index = (
            f.index.tz_localize("UTC") if f.index.tz is None else f.index
        ).tz_convert(index.tz)
    elif f.index.tz is not None:
        f.index = f.index.tz_convert("UTC").tz_localize(None)

    f = f[~f.index.duplicated(keep="last")].sort_index()
    aligned = f.reindex(index)

    if (max_gap > 0 or max_tail_gap > 0) and bool(aligned.isna().any().any()):
        # Heal bounded gaps, all-or-nothing PER RUN with a split budget:
        # - interior (and leading) runs: budget = max_gap
        # - the trailing-edge run (the NaN run containing index[-1]):
        #   budget = max(max_gap, max_tail_gap)
        # A run whose length exceeds its budget is left entirely NaN (no
        # partial fabrication) and still trips the NaN gate below if it
        # intersects the validated window.  An oversize run outside the
        # window (e.g. a price series that starts years after data_start)
        # therefore does not block healing of small in-window gaps.
        pre_heal_na = aligned.isna()
        last_ts = aligned.index[-1]

        # Apply interpolate and bfill only when max_gap > 0 (pandas raises
        # ValueError when limit=0 is passed to these methods).
        if max_gap > 0:
            healed = aligned.interpolate(
                method="time",
                limit=max_gap,
                limit_area="inside",
                limit_direction="both",
            )
            healed = healed.bfill(limit=max_gap)
        else:
            healed = aligned.copy()

        # ffill runs with limit=max(max_gap, max_tail_gap) and may fill interior
        # gaps too, not just the trailing edge.  The re-mask loop below immediately
        # re-NaNs every run whose length exceeds its per-run budget, so only the
        # tail run (which gets the elevated budget) survives when max_gap=0.
        healed = healed.ffill(limit=max(max_gap, max_tail_gap))

        # Re-mask runs whose length exceeds their per-run budget.  The
        # interpolate/bfill/ffill ``limit`` caps fill steps per direction, so
        # an oversize run could otherwise be partially (or, healed from both
        # sides, even fully) filled.
        for col in aligned.columns:
            na = pre_heal_na[col]
            if not bool(na.any()):
                continue
            # cumsum trick: consecutive NaN cells share one group label
            # derived from the count of preceding non-NaN cells.
            run_id = (~na).cumsum()[na]
            run_len = run_id.groupby(run_id).transform("size")
            tail_run_id = run_id.get(last_ts)  # None when last cell is valid
            budget = pd.Series(max_gap, index=run_len.index)
            if tail_run_id is not None:
                budget[run_id == tail_run_id] = max(max_gap, max_tail_gap)
            oversize_idx = run_len.index[run_len > budget]
            if len(oversize_idx):
                healed.loc[oversize_idx, col] = float("nan")
        aligned = healed
        healed_mask = pre_heal_na & ~aligned.isna()
        if bool(healed_mask.any().any()):
            n_healed = int(healed_mask.sum().sum())
            healed_ts = aligned.index[healed_mask.any(axis=1)]
            # Identify contiguous runs of healed timestamps
            gaps: list = []
            if len(healed_ts) > 0:
                run_start = healed_ts[0]
                prev = healed_ts[0]
                for ts in healed_ts[1:]:
                    if ts - prev > pd.Timedelta(hours=1):
                        gaps.append((run_start, prev))
                        run_start = ts
                    prev = ts
                gaps.append((run_start, prev))
            n_gaps = len(gaps)
            spans_str = str([(str(s), str(e)) for s, e in gaps[:3]])
            logger.warning(
                "%s: healed %d missing cell(s) in %d gap(s) "
                "(max_gap=%d, max_tail_gap=%d); spans: %s",
                provider,
                n_healed,
                n_gaps,
                max_gap,
                max_tail_gap,
                spans_str,
            )

    if validate_index is None:
        check = aligned
        check_index = index
    else:
        check_index = validate_index.intersection(index)
        if len(check_index) == 0:
            raise ExogProviderError(
                f"{provider}: validate_index has no overlap with the request "
                "index; windowed validation would be vacuous."
            )
        check = aligned.reindex(check_index)

    if bool(check.isna().any().any()):
        missing_mask = check.isna().any(axis=1)
        n_missing = int(missing_mask.sum())
        first = [str(ts) for ts in check_index[missing_mask][:3]]
        raise ExogProviderError(
            f"{provider}: {n_missing} of {len(check_index)} requested timestamps have "
            f"no value (first: {first}). The provider cannot cover the requested "
            "range; supply the data or disable its flag."
        )
    return aligned.astype("float32")


class ExogFeatureProvider(ABC):
    """Contract for a pluggable exogenous-feature source.

    A provider maps the hourly target index to a numeric feature
    frame on that exact index. Subclasses set `name` (a short identifier
    used in logs and as the default column name) and implement `build`.

    Implementations should load their backing data lazily inside `build`
    and raise `ExogProviderError` when the data is missing or cannot
    cover the requested range, so the fail-safe policy lives in one place.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.exog_providers import (
            ExogFeatureProvider,
            ExogProviderError,
        )

        class ConstantProvider(ExogFeatureProvider):
            name = "constant"

            def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
                return pd.DataFrame({"constant": 1.0}, index=index).astype("float32")

        idx = pd.date_range("2023-06-01", periods=6, freq="h", tz="UTC")
        p = ConstantProvider()
        out = p.build(idx)
        print(p.name, out.shape, out.dtypes["constant"].name)
        assert out.shape == (6, 1)
        assert not out.isna().any().any()
        ```
    """

    name: str = "exog_provider"

    @abstractmethod
    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return features aligned to *index*.

        Args:
            index: Hourly ``DatetimeIndex`` (typically tz-aware UTC) covering the
                full training-plus-forecast window.

        Returns:
            pd.DataFrame: Numeric columns indexed exactly by *index*, NaN-free
                within the validated window (the full *index* unless a
                ``provider_window`` was set at construction).

        Raises:
            ExogProviderError: If the provider cannot cover *index*.

        Examples:
            ```{python}
            import pandas as pd
            from spotforecast2_safe.preprocessing.exog_providers import (
                ExogFeatureProvider,
            )

            class LinearProvider(ExogFeatureProvider):
                name = "linear"

                def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
                    vals = range(len(index))
                    return pd.DataFrame({"linear": list(vals)}, index=index).astype("float32")

            idx = pd.date_range("2023-06-01", periods=4, freq="h", tz="UTC")
            out = LinearProvider().build(idx)
            print(out.shape, out["linear"].tolist())
            assert out.shape == (4, 1)
            ```
        """
        raise NotImplementedError


class CovidInfectionRateProvider(ExogFeatureProvider):
    """German national COVID-19 7-day incidence as an exogenous level regressor.

    Reads the bundled, static RKI series
    (``datasets/csv/covid_infection_rate_de.csv``, CC-BY-4.0) and broadcasts the
    daily national 7-day incidence (per 100,000) onto the hourly target index:
    forward-filled within the data's date span and filled with *fill_outside*
    (``0.0`` by default) before the first and after the last observed day, since
    outside the pandemic window there is no signal.

    This is a slow socio-economic level input (a lockdown-stringency proxy). It
    carries the CR-3 release-lag caveat: the bundled file is the final published
    vintage, whereas on a true live path only the latest, lagged vintage is
    available. For training over historical data this is the standard treatment.

    Args:
        data_home: Unused (kept for a uniform provider signature); the dataset is
            package data, located via ``get_package_data_home()``.
        csv_path: Optional explicit path to the COVID CSV, overriding the bundled
            location.
        column: Output column name. Defaults to ``"covid_infection_rate"``.
        fill_outside: Value used outside the observed date span. Defaults to
            ``0.0``.
        max_gap: Maximum contiguous missing-value run healed by ``_align_to_index``.
            See :func:`_align_to_index` for full semantics. Defaults to ``0``.
        max_tail_gap: Extended healing budget for the trailing-edge NaN run
            (the run containing ``index[-1]``). The effective tail budget is
            ``max(max_gap, max_tail_gap)``. See :func:`_align_to_index`.
            Defaults to ``0``.
        provider_window: Validation index passed to ``_align_to_index`` as
            *validate_index*. See :func:`_align_to_index`. Defaults to ``None``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.exog_providers import (
            CovidInfectionRateProvider,
        )

        idx = pd.date_range("2021-12-01", periods=24, freq="h", tz="UTC")
        out = CovidInfectionRateProvider().build(idx)
        print(out.columns.tolist(), out.shape, bool(out.isna().any().any()))
        ```
    """

    name = "covid_infection_rate"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        csv_path: Optional[Union[str, Path]] = None,
        column: str = "covid_infection_rate",
        fill_outside: float = 0.0,
        max_gap: int = 0,
        max_tail_gap: int = 0,
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.column = column
        self.fill_outside = float(fill_outside)
        self.max_gap = max_gap
        self.max_tail_gap = max_tail_gap
        self.provider_window = provider_window

    def _load_daily(self) -> pd.Series:
        from spotforecast2_safe.data.fetch_data import get_package_data_home

        path = self.csv_path or (
            get_package_data_home() / "covid_infection_rate_de.csv"
        )
        if not path.exists():
            raise ExogProviderError(
                f"{self.name}: bundled dataset not found at {path}."
            )
        df = pd.read_csv(path)
        if "date" not in df.columns or "covid_infection_rate" not in df.columns:
            raise ExogProviderError(
                f"{self.name}: {path} must have 'date' and 'covid_infection_rate' "
                f"columns; got {list(df.columns)}."
            )
        daily = pd.Series(
            df["covid_infection_rate"].to_numpy(dtype=float),
            index=pd.to_datetime(df["date"], utc=True).dt.normalize(),
        )
        return daily[~daily.index.duplicated(keep="last")].sort_index()

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a single-column ``float32`` frame with the COVID incidence.

        Args:
            index: Hourly ``DatetimeIndex`` (tz-aware UTC) covering the
                training-plus-forecast window.

        Returns:
            pd.DataFrame: One column (``covid_infection_rate`` by default),
                ``float32``, indexed exactly by *index*. Values outside the
                pandemic date range are filled with ``fill_outside`` (0.0).

        Raises:
            ExogProviderError: If the bundled CSV is absent or malformed.

        Examples:
            ```{python}
            import pandas as pd
            from spotforecast2_safe.preprocessing.exog_providers import (
                CovidInfectionRateProvider,
            )

            idx = pd.date_range("2021-12-01", periods=24, freq="h", tz="UTC")
            provider = CovidInfectionRateProvider()
            out = provider.build(idx)
            print(out.columns.tolist(), out.shape, out.dtypes.iloc[0].name)
            assert out.shape == (24, 1)
            assert not out.isna().any().any()
            ```
        """
        daily = self._load_daily()
        if len(index) == 0:
            return pd.DataFrame({self.column: pd.Series(dtype="float32")}, index=index)
        if daily.empty:
            return _align_to_index(
                pd.DataFrame({self.column: self.fill_outside}, index=index),
                index,
                provider=self.name,
                max_gap=self.max_gap,
                max_tail_gap=self.max_tail_gap,
                validate_index=self.provider_window,
            )

        first, last = daily.index.min(), daily.index.max()
        target_dates = (
            index.tz_convert("UTC")
            if index.tz is not None
            else index.tz_localize("UTC")
        ).normalize()
        uniq = target_dates.unique().sort_values()

        grid = daily.reindex(daily.index.union(uniq)).sort_index().ffill()
        per_date = grid.reindex(uniq)
        per_date[(uniq < first) | (uniq > last)] = self.fill_outside
        per_date = per_date.fillna(self.fill_outside)

        mapping = dict(zip(uniq, per_date.to_numpy()))
        values = [mapping[d] for d in target_dates]
        return _align_to_index(
            pd.DataFrame({self.column: values}, index=index),
            index,
            provider=self.name,
            max_gap=self.max_gap,
            max_tail_gap=self.max_tail_gap,
            validate_index=self.provider_window,
        )


class EntsoeForecastLoadProvider(ExogFeatureProvider):
    """ENTSO-E day-ahead Forecasted Load as an exogenous near-oracle prior.

    Wraps `spotforecast2_safe.data.fetch_data.load_timeseries_forecast`, which
    reads the ``Forecasted Load`` column already merged into
    ``interim/energy_load.csv``. The day-ahead forecast is published on D-1 and
    is therefore genuinely available at forecast time (leakage-clean, CR-3).

    Args:
        data_home: Root data directory forwarded to the loader. ``None`` resolves
            via ``get_data_home()``.
        max_gap: Maximum contiguous missing-value run healed by ``_align_to_index``.
            See :func:`_align_to_index` for full semantics. Defaults to ``0``.
        max_tail_gap: Extended healing budget for the trailing-edge NaN run.
            See :func:`_align_to_index`. Defaults to ``0``.
        provider_window: Validation index passed to ``_align_to_index`` as
            *validate_index*. See :func:`_align_to_index`. Defaults to ``None``.

    Examples:
        ```{python}
        import os
        import shutil
        import tempfile

        import pandas as pd

        from spotforecast2_safe.preprocessing.exog_providers import (
            EntsoeForecastLoadProvider,
        )

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        interim = os.path.join(tmp, "interim")
        os.makedirs(interim, exist_ok=True)
        idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
        pd.DataFrame(
            {"Actual Load": 100.0, "Forecasted Load": 99.0}, index=idx
        ).rename_axis("Time (UTC)").to_csv(os.path.join(interim, "energy_load.csv"))

        out = EntsoeForecastLoadProvider().build(idx)
        print(out.columns.tolist(), out.shape)

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """

    name = "entsoe_forecasted_load"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        max_gap: int = 0,
        max_tail_gap: int = 0,
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.max_gap = max_gap
        self.max_tail_gap = max_tail_gap
        self.provider_window = provider_window

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return the day-ahead Forecasted Load aligned to *index*.

        Args:
            index: Hourly ``DatetimeIndex`` (tz-aware UTC) for the forecast window.

        Returns:
            pd.DataFrame: Single column ``entsoe_forecasted_load``, ``float32``.

        Raises:
            ExogProviderError: If the interim CSV is missing or the
                ``Forecasted Load`` column is absent.

        Examples:
            ```{python}
            import os
            import shutil
            import tempfile

            import pandas as pd

            from spotforecast2_safe.preprocessing.exog_providers import (
                EntsoeForecastLoadProvider,
            )

            tmp = tempfile.mkdtemp()
            os.environ["SPOTFORECAST2_DATA"] = tmp
            os.makedirs(os.path.join(tmp, "interim"), exist_ok=True)
            idx = pd.date_range("2023-06-01", periods=24, freq="h", tz="UTC")
            pd.DataFrame(
                {"Actual Load": 100.0, "Forecasted Load": 98.0}, index=idx
            ).rename_axis("Time (UTC)").to_csv(
                os.path.join(tmp, "interim", "energy_load.csv")
            )

            out = EntsoeForecastLoadProvider().build(idx)
            print(out.columns.tolist(), out.shape, out.dtypes.iloc[0].name)
            assert out.shape == (24, 1)

            shutil.rmtree(tmp)
            del os.environ["SPOTFORECAST2_DATA"]
            ```
        """
        from spotforecast2_safe.data.fetch_data import load_timeseries_forecast

        try:
            series = load_timeseries_forecast(
                data_home=self.data_home, on_missing="passthrough"
            )
        except (FileNotFoundError, KeyError) as exc:
            raise ExogProviderError(f"{self.name}: {exc}") from exc
        return _align_to_index(
            series.to_frame(self.name),
            index,
            provider=self.name,
            max_gap=self.max_gap,
            max_tail_gap=self.max_tail_gap,
            validate_index=self.provider_window,
        )


class EntsoeRenewableForecastProvider(ExogFeatureProvider):
    """ENTSO-E day-ahead wind and solar generation forecast.

    Reads ``interim/renewable_forecast.csv`` via
    `spotforecast2_safe.data.fetch_data.load_renewable_forecast` and emits two
    columns: ``entsoe_wind_forecast`` (sum of all wind columns) and
    ``entsoe_solar_forecast`` (sum of all solar columns). Day-ahead forecasts are
    leakage-clean; the realised generation must never be used.

    Args:
        data_home: Root data directory forwarded to the loader.
        max_gap: Maximum contiguous missing-value run healed by ``_align_to_index``.
            See `_align_to_index` for full semantics. Defaults to ``0``.
        max_tail_gap: Extended healing budget for the trailing-edge NaN run.
            See `_align_to_index`. Defaults to ``0``.
        provider_window: Validation index passed to ``_align_to_index`` as
            *validate_index*. See `_align_to_index`. Defaults to ``None``.

    Examples:
        ```{python}
        import os
        import shutil
        import tempfile

        import pandas as pd

        from spotforecast2_safe.preprocessing.exog_providers import (
            EntsoeRenewableForecastProvider,
        )

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        os.makedirs(os.path.join(tmp, "interim"), exist_ok=True)
        idx = pd.date_range("2023-06-01", periods=24, freq="h", tz="UTC")
        pd.DataFrame(
            {"Solar": 3.0, "Wind Onshore": 5.0}, index=idx
        ).rename_axis("Time (UTC)").to_csv(
            os.path.join(tmp, "interim", "renewable_forecast.csv")
        )

        provider = EntsoeRenewableForecastProvider()
        out = provider.build(idx)
        print(out.columns.tolist(), out.shape, out.dtypes.iloc[0].name)
        assert set(out.columns) == {"entsoe_wind_forecast", "entsoe_solar_forecast"}
        assert out.shape == (24, 2)

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """

    name = "entsoe_renewable_forecast"
    wind_col = "entsoe_wind_forecast"
    solar_col = "entsoe_solar_forecast"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        max_gap: int = 0,
        max_tail_gap: int = 0,
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.max_gap = max_gap
        self.max_tail_gap = max_tail_gap
        self.provider_window = provider_window

    def _load(self) -> pd.DataFrame:
        from spotforecast2_safe.data.fetch_data import load_renewable_forecast

        try:
            df = load_renewable_forecast(
                data_home=self.data_home, on_missing="passthrough"
            )
        except FileNotFoundError as exc:
            raise ExogProviderError(f"{self.name}: {exc}") from exc

        wind = [c for c in df.columns if "wind" in c.lower()]
        solar = [c for c in df.columns if "solar" in c.lower()]
        if not wind and not solar:
            raise ExogProviderError(
                f"{self.name}: no wind/solar columns in renewable_forecast.csv "
                f"(got {list(df.columns)})."
            )
        out = pd.DataFrame(index=df.index)
        out[self.wind_col] = df[wind].sum(axis=1, skipna=False) if wind else 0.0
        out[self.solar_col] = df[solar].sum(axis=1, skipna=False) if solar else 0.0
        return out

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return wind and solar forecast columns aligned to *index*.

        Args:
            index: Hourly ``DatetimeIndex`` (tz-aware UTC) for the forecast window.

        Returns:
            pd.DataFrame: Two columns — ``entsoe_wind_forecast`` and
                ``entsoe_solar_forecast`` — as ``float32``.

        Raises:
            ExogProviderError: If ``interim/renewable_forecast.csv`` is missing
                or contains no wind or solar columns.

        Examples:
            ```{python}
            import os
            import shutil
            import tempfile

            import pandas as pd

            from spotforecast2_safe.preprocessing.exog_providers import (
                EntsoeRenewableForecastProvider,
            )

            tmp = tempfile.mkdtemp()
            os.environ["SPOTFORECAST2_DATA"] = tmp
            os.makedirs(os.path.join(tmp, "interim"), exist_ok=True)
            idx = pd.date_range("2023-06-01", periods=12, freq="h", tz="UTC")
            pd.DataFrame(
                {"Solar": 2.0, "Wind Onshore": 4.0}, index=idx
            ).rename_axis("Time (UTC)").to_csv(
                os.path.join(tmp, "interim", "renewable_forecast.csv")
            )

            out = EntsoeRenewableForecastProvider().build(idx)
            print(out.columns.tolist(), out.shape)
            assert not out.isna().any().any()

            shutil.rmtree(tmp)
            del os.environ["SPOTFORECAST2_DATA"]
            ```
        """
        return _align_to_index(
            self._load(),
            index,
            provider=self.name,
            max_gap=self.max_gap,
            max_tail_gap=self.max_tail_gap,
            validate_index=self.provider_window,
        )


class EntsoeNetLoadProvider(ExogFeatureProvider):
    """ENTSO-E day-ahead net load = Forecasted Load − (wind + solar) forecast.

    Combines the day-ahead Forecasted Load with the day-ahead renewable
    forecast to form the net-load prior the residual is often modelled against.
    Both inputs are day-ahead (leakage-clean). Raises
    `ExogProviderError` if either input is unavailable.

    Args:
        data_home: Root data directory forwarded to the loaders.
        max_gap: Maximum contiguous missing-value run healed by ``_align_to_index``.
            See `_align_to_index` for full semantics. Defaults to ``0``.
        max_tail_gap: Extended healing budget for the trailing-edge NaN run.
            See `_align_to_index`. Defaults to ``0``.
        provider_window: Validation index passed to ``_align_to_index`` as
            *validate_index*. See `_align_to_index`. Defaults to ``None``.

    Examples:
        ```{python}
        import os
        import shutil
        import tempfile

        import pandas as pd

        from spotforecast2_safe.preprocessing.exog_providers import (
            EntsoeNetLoadProvider,
        )

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        os.makedirs(os.path.join(tmp, "interim"), exist_ok=True)
        idx = pd.date_range("2023-06-01", periods=24, freq="h", tz="UTC")
        pd.DataFrame(
            {"Actual Load": 100.0, "Forecasted Load": 90.0}, index=idx
        ).rename_axis("Time (UTC)").to_csv(
            os.path.join(tmp, "interim", "energy_load.csv")
        )
        pd.DataFrame(
            {"Solar": 3.0, "Wind Onshore": 5.0}, index=idx
        ).rename_axis("Time (UTC)").to_csv(
            os.path.join(tmp, "interim", "renewable_forecast.csv")
        )

        provider = EntsoeNetLoadProvider()
        out = provider.build(idx)
        print(out.columns.tolist(), out.shape, float(out.iloc[0, 0]))
        assert out.shape == (24, 1)
        assert abs(float(out.iloc[0, 0]) - 82.0) < 0.1  # 90 - (3 + 5)

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """

    name = "entsoe_net_load"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        max_gap: int = 0,
        max_tail_gap: int = 0,
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.max_gap = max_gap
        self.max_tail_gap = max_tail_gap
        self.provider_window = provider_window

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return the day-ahead net load (Forecasted Load minus renewables).

        Args:
            index: Hourly ``DatetimeIndex`` (tz-aware UTC) for the forecast window.

        Returns:
            pd.DataFrame: Single column ``entsoe_net_load``, ``float32``.

        Raises:
            ExogProviderError: If either ``energy_load.csv`` or
                ``renewable_forecast.csv`` is missing.

        Examples:
            ```{python}
            import os
            import shutil
            import tempfile

            import pandas as pd

            from spotforecast2_safe.preprocessing.exog_providers import (
                EntsoeNetLoadProvider,
            )

            tmp = tempfile.mkdtemp()
            os.environ["SPOTFORECAST2_DATA"] = tmp
            os.makedirs(os.path.join(tmp, "interim"), exist_ok=True)
            idx = pd.date_range("2023-06-01", periods=12, freq="h", tz="UTC")
            pd.DataFrame(
                {"Actual Load": 100.0, "Forecasted Load": 80.0}, index=idx
            ).rename_axis("Time (UTC)").to_csv(
                os.path.join(tmp, "interim", "energy_load.csv")
            )
            pd.DataFrame(
                {"Solar": 2.0, "Wind Onshore": 6.0}, index=idx
            ).rename_axis("Time (UTC)").to_csv(
                os.path.join(tmp, "interim", "renewable_forecast.csv")
            )

            out = EntsoeNetLoadProvider().build(idx)
            print(out.columns.tolist(), out.shape, float(out.iloc[0, 0]))
            assert out.shape == (12, 1)

            shutil.rmtree(tmp)
            del os.environ["SPOTFORECAST2_DATA"]
            ```
        """
        from spotforecast2_safe.data.fetch_data import (
            load_renewable_forecast,
            load_timeseries_forecast,
        )

        try:
            load = load_timeseries_forecast(
                data_home=self.data_home, on_missing="passthrough"
            )
            renewable = load_renewable_forecast(
                data_home=self.data_home, on_missing="passthrough"
            )
        except (FileNotFoundError, KeyError) as exc:
            raise ExogProviderError(f"{self.name}: {exc}") from exc

        renewable_total = renewable.sum(axis=1, skipna=False)
        net = load - renewable_total
        return _align_to_index(
            net.to_frame(self.name),
            index,
            provider=self.name,
            max_gap=self.max_gap,
            max_tail_gap=self.max_tail_gap,
            validate_index=self.provider_window,
        )


class EntsoeDayAheadPriceProvider(ExogFeatureProvider):
    """ENTSO-E day-ahead spot price (DE/LU) as an exogenous input.

    Reads ``interim/day_ahead_price.csv`` via
    `spotforecast2_safe.data.fetch_data.load_day_ahead_price`. The day-ahead
    auction price is published on D-1 and is leakage-clean at forecast time;
    the realised price must never be used.

    Args:
        data_home: Root data directory forwarded to the loader.
        max_gap: Maximum contiguous missing-value run healed by ``_align_to_index``.
            See `_align_to_index` for full semantics. Defaults to ``0``.
        max_tail_gap: Extended healing budget for the trailing-edge NaN run.
            See `_align_to_index`. Defaults to ``0``.
        provider_window: Validation index passed to ``_align_to_index`` as
            *validate_index*. See `_align_to_index`. Defaults to ``None``.

    Examples:
        ```{python}
        import os
        import shutil
        import tempfile

        import pandas as pd

        from spotforecast2_safe.preprocessing.exog_providers import (
            EntsoeDayAheadPriceProvider,
        )

        tmp = tempfile.mkdtemp()
        os.environ["SPOTFORECAST2_DATA"] = tmp
        os.makedirs(os.path.join(tmp, "interim"), exist_ok=True)
        idx = pd.date_range("2023-06-01", periods=24, freq="h", tz="UTC")
        pd.DataFrame(
            {"Day-ahead Price": 95.0}, index=idx
        ).rename_axis("Time (UTC)").to_csv(
            os.path.join(tmp, "interim", "day_ahead_price.csv")
        )

        provider = EntsoeDayAheadPriceProvider()
        out = provider.build(idx)
        print(out.columns.tolist(), out.shape, out.dtypes.iloc[0].name)
        assert out.shape == (24, 1)
        assert not out.isna().any().any()

        shutil.rmtree(tmp)
        del os.environ["SPOTFORECAST2_DATA"]
        ```
    """

    name = "entsoe_day_ahead_price"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        max_gap: int = 0,
        max_tail_gap: int = 0,
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.max_gap = max_gap
        self.max_tail_gap = max_tail_gap
        self.provider_window = provider_window

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return the day-ahead price series aligned to *index*.

        Args:
            index: Hourly ``DatetimeIndex`` (tz-aware UTC) for the forecast window.

        Returns:
            pd.DataFrame: Single column ``entsoe_day_ahead_price``, ``float32``.

        Raises:
            ExogProviderError: If ``interim/day_ahead_price.csv`` is missing or
                the ``Day-ahead Price`` column is absent.

        Examples:
            ```{python}
            import os
            import shutil
            import tempfile

            import pandas as pd

            from spotforecast2_safe.preprocessing.exog_providers import (
                EntsoeDayAheadPriceProvider,
            )

            tmp = tempfile.mkdtemp()
            os.environ["SPOTFORECAST2_DATA"] = tmp
            os.makedirs(os.path.join(tmp, "interim"), exist_ok=True)
            idx = pd.date_range("2023-06-01", periods=12, freq="h", tz="UTC")
            pd.DataFrame(
                {"Day-ahead Price": 88.5}, index=idx
            ).rename_axis("Time (UTC)").to_csv(
                os.path.join(tmp, "interim", "day_ahead_price.csv")
            )

            out = EntsoeDayAheadPriceProvider().build(idx)
            print(out.columns.tolist(), out.shape, float(out.iloc[0, 0]))
            assert out.shape == (12, 1)

            shutil.rmtree(tmp)
            del os.environ["SPOTFORECAST2_DATA"]
            ```
        """
        from spotforecast2_safe.data.fetch_data import load_day_ahead_price

        try:
            series = load_day_ahead_price(
                data_home=self.data_home, on_missing="passthrough"
            )
        except (FileNotFoundError, KeyError) as exc:
            raise ExogProviderError(f"{self.name}: {exc}") from exc
        return _align_to_index(
            series.to_frame(self.name),
            index,
            provider=self.name,
            max_gap=self.max_gap,
            max_tail_gap=self.max_tail_gap,
            validate_index=self.provider_window,
        )


class EventWindowProvider(ExogFeatureProvider):
    """Generic event-window provider driven by a bundled CSV file.

    Reads a CSV with columns ``event``, ``start_utc``, ``end_utc``, and an
    optional ``intensity`` column (extra columns such as ``match_kickoff_utc``
    or ``provisional`` are silently ignored — they are documentation only).
    For each timestamp ``t`` in the requested ``DatetimeIndex`` the output
    value is ``max(intensity)`` over all CSV rows whose window contains ``t``
    (inclusive: ``start_utc <= t <= end_utc``), or ``0.0`` when no window
    covers ``t``.

    The ``intensity`` column defaults to ``1.0`` when absent. Membership is
    evaluated directly on the provided index timestamps so the provider works
    at any cadence (hourly, 15-minute, etc.). Timestamps in the CSV are
    ISO-8601 with UTC offset; timezone harmonisation follows the same logic as
    the other providers in this module. No ``fill_outside`` knob exists:
    outside any window the value is structurally ``0.0``.

    Subclasses set ``csv_filename`` and the default ``column`` name; they do
    not need to override any method.

    Args:
        data_home: Unused (kept for a uniform provider signature); the dataset
            is package data located via ``get_package_data_home()``.
        csv_path: Optional explicit path to the event-window CSV, overriding
            the bundled location derived from ``csv_filename``.
        column: Output column name. Defaults to the subclass ``column`` class
            attribute.
        max_gap: Accepted for provider-factory API uniformity; has no effect.
            Values are structurally ``0.0`` outside event windows (no NaN can
            arise), so gap healing is not applicable.
        max_tail_gap: Accepted for provider-factory API uniformity; has no
            effect (same reasoning as ``max_gap``).
        provider_window: Accepted for provider-factory API uniformity; has no
            effect (same reasoning as ``max_gap``).

    Notes:
        Rejected / deferred drivers documented here for the record:

        - Open-ended Ukraine-invasion step dummy (2022-02-24 onward): rejected
          — the shift is gradual and non-permanent, and in recent training
          windows the signal is near-constant and uninformative to GBDT.
        - Monthly Destatis PPI proxy: deferred — ``include_entsoe_day_ahead_price``
          already covers the price channel.
        - Population / refugee level index: deferred — effect is 0.3–1 % of
          mean load, below day-ahead noise; any future build must use YoY
          growth rates to avoid the Zensus-2022 −1.4 M rebase hazard.
        - Half-time TV-pickup sub-column, eclipse / strike / nuclear-phase-out
          flags: rejected.
        - Christmas-shutdown and DST days: already covered by the existing
          holiday / day-type features.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.exog_providers import (
            FootballMatchWindowProvider,
        )

        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        print(out.columns.tolist(), out.shape, out.dtypes.iloc[0].name)
        assert out.loc["2024-06-14T19:00:00Z", "football_match_window"] == 1.0
        assert out.loc["2024-06-14T17:00:00Z", "football_match_window"] == 0.0
        ```
    """

    name: str = "event_window"
    csv_filename: str = ""
    column: str = "event_window"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        csv_path: Optional[Union[str, Path]] = None,
        column: Optional[str] = None,
        max_gap: int = 0,  # accepted for uniform factory signature; unused
        max_tail_gap: int = 0,  # accepted for uniform factory signature; unused
        provider_window: Optional[pd.DatetimeIndex] = None,  # accepted; unused
    ) -> None:
        self.data_home = data_home
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.column = column if column is not None else self.__class__.column
        self.max_gap = max_gap
        self.max_tail_gap = max_tail_gap
        self.provider_window = provider_window
        # gap/window knobs are stored for provider-factory API uniformity but
        # have no effect: values are structurally 0.0 outside any event window
        # (no NaN can arise), so healing and windowed validation are not
        # applicable.

    def _load_windows(self) -> pd.DataFrame:
        """Load and validate the event-window CSV.

        Returns:
            pd.DataFrame: Validated frame with columns ``start_utc``,
                ``end_utc``, and ``intensity`` (float), all timestamps
                tz-aware UTC.

        Raises:
            ExogProviderError: If the file is absent or missing required
                columns.
        """
        from spotforecast2_safe.data.fetch_data import get_package_data_home

        path = self.csv_path or (get_package_data_home() / self.csv_filename)
        if not path.exists():
            raise ExogProviderError(
                f"{self.name}: bundled dataset not found at {path}."
            )
        df = pd.read_csv(path)
        for required in ("event", "start_utc", "end_utc"):
            if required not in df.columns:
                raise ExogProviderError(
                    f"{self.name}: {path} must have '{required}' column; "
                    f"got {list(df.columns)}."
                )
        df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
        df["end_utc"] = pd.to_datetime(df["end_utc"], utc=True)
        if "intensity" not in df.columns:
            df["intensity"] = 1.0
        else:
            df["intensity"] = df["intensity"].astype(float)
        return df[["start_utc", "end_utc", "intensity"]]

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a single-column ``float32`` frame with event-window values.

        For each timestamp ``t`` in *index* the value is
        ``max(intensity)`` over all rows whose window contains ``t``
        (``start_utc <= t <= end_utc``), or ``0.0`` when no row covers ``t``.

        Args:
            index: ``DatetimeIndex`` (any cadence, tz-aware or tz-naive)
                covering the training-plus-forecast window.

        Returns:
            pd.DataFrame: One column (named by ``self.column``), ``float32``,
                indexed exactly by *index*. NaN-free; ``0.0`` outside all
                event windows.

        Raises:
            ExogProviderError: If the bundled CSV is absent or malformed.

        Examples:
            ```{python}
            import pandas as pd
            from spotforecast2_safe.preprocessing.exog_providers import (
                FootballMatchWindowProvider,
            )

            idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
            provider = FootballMatchWindowProvider()
            out = provider.build(idx)
            print(out.columns.tolist(), out.shape, out.dtypes.iloc[0].name)
            assert out.shape == (48, 1)
            assert not out.isna().any().any()
            assert out.loc["2024-06-14T19:00:00Z", "football_match_window"] == 1.0
            ```
        """
        windows = self._load_windows()
        if len(index) == 0:
            return pd.DataFrame({self.column: pd.Series(dtype="float32")}, index=index)

        # Normalise the index to UTC for membership testing.
        if index.tz is not None:
            idx_utc = index.tz_convert("UTC")
        else:
            idx_utc = index.tz_localize("UTC")

        values = pd.Series(0.0, index=idx_utc, dtype="float64")
        for _, row in windows.iterrows():
            mask = (idx_utc >= row["start_utc"]) & (idx_utc <= row["end_utc"])
            values[mask] = values[mask].clip(
                lower=row["intensity"]
            )  # clip(lower=x) == max(current, x); valid because values start at 0.0

        result = pd.DataFrame(
            {self.column: values.to_numpy(dtype="float32")}, index=index
        )
        return result


class FootballMatchWindowProvider(EventWindowProvider):
    """German football match event-window provider.

    Produces a single ``float32`` column ``football_match_window`` whose value
    is ``1.0`` during configured match windows (pre-kickoff to post-final-whistle
    including overtime allowance) and ``0.0`` otherwise. The bundled CSV covers
    all German national-team matches and every tournament final from UEFA Euro
    2016 through FIFA World Cup 2026.

    Row-selection rule: every match involving the German national team plus
    every tournament final (Euro 2016 → WC 2026). Window rule: [kickoff −
    30 min floored to the hour, kickoff + D + 60 min ceiled to the hour] where
    D = 120 min for group-stage matches and D = 165 min for knockout matches.

    Rows with ``provisional = 1`` mark WC-2026 knockout slots on Germany's two
    potential bracket paths (slot times fixed, teams unresolved at the time of
    publication). The provider uses these rows identically to resolved rows.
    They may be revised when the bracket is finalised; the ``provisional``
    column is documentation only and is not read by the provider.

    Args:
        data_home: Unused (kept for a uniform provider signature).
        csv_path: Optional explicit path to the football-match CSV, overriding
            the bundled ``football_match_windows_de.csv``.
        column: Output column name. Defaults to ``"football_match_window"``.
        max_gap: Accepted for provider-factory API uniformity; has no effect.
        max_tail_gap: Accepted for provider-factory API uniformity; has no
            effect.
        provider_window: Accepted for provider-factory API uniformity; has no
            effect.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.exog_providers import (
            FootballMatchWindowProvider,
        )

        # EURO 2024: Germany vs Scotland kicked off 2024-06-14 21:00 CEST
        # = 2024-06-14 19:00 UTC; window 18:00–22:00 UTC.
        idx = pd.date_range("2024-06-14", periods=48, freq="h", tz="UTC")
        out = FootballMatchWindowProvider().build(idx)
        print(out.columns.tolist(), out.shape, out.dtypes.iloc[0].name)
        assert out.loc["2024-06-14T19:00:00Z", "football_match_window"] == 1.0
        assert out.loc["2024-06-14T17:00:00Z", "football_match_window"] == 0.0
        ```
    """

    name = "football_match_window"
    csv_filename = "football_match_windows_de.csv"
    column = "football_match_window"


class EnergyCrisisWindowProvider(EventWindowProvider):
    """German energy-saving regulatory window provider.

    Produces a single ``float32`` column ``energy_saving_window`` whose value
    is ``1.0`` during the two regulatory energy-saving periods in force during
    the 2022–2023 European energy crisis and ``0.0`` otherwise.

    The two bundled windows are:

    - **EnSikuMaV** (Kurzfristenergieversorgungssicherungsmaßnahmenverordnung):
      in force 2022-09-01 00:00 CEST through 2023-04-15 23:00 CEST, including
      the February 2023 extension.
    - **EU Council Regulation 2022/1854**: mandatory −5 % peak-hour consumption
      reduction 2022-12-01 00:00 CET through 2023-03-31 23:00 CEST.

    The ``intensity`` values are ``1.0`` for both rows; overlapping hours
    (December 2022 through March 2023) therefore also yield ``1.0``.

    Args:
        data_home: Unused (kept for a uniform provider signature).
        csv_path: Optional explicit path to the energy-saving CSV, overriding
            the bundled ``energy_saving_windows_de.csv``.
        column: Output column name. Defaults to ``"energy_saving_window"``.
        max_gap: Accepted for provider-factory API uniformity; has no effect.
        max_tail_gap: Accepted for provider-factory API uniformity; has no
            effect.
        provider_window: Accepted for provider-factory API uniformity; has no
            effect.

    Notes:
        An open-ended Ukraine-invasion step dummy (2022-02-24 onward) was
        considered and rejected: the load effect is gradual and non-permanent,
        and in recent training windows the signal would be near-constant and
        uninformative to tree-based models. The ``include_entsoe_day_ahead_price``
        flag already captures the energy-price channel that the invasion
        primarily affected.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.exog_providers import (
            EnergyCrisisWindowProvider,
        )

        # Window is active: 2022-10-01 is inside ENSIKUMAV_DE
        idx_in = pd.date_range("2022-10-01", periods=24, freq="h", tz="UTC")
        out_in = EnergyCrisisWindowProvider().build(idx_in)
        print(out_in.columns.tolist(), out_in.shape)
        assert (out_in["energy_saving_window"] == 1.0).all()

        # Window is inactive: 2022-07-01 predates both regulatory periods
        idx_out = pd.date_range("2022-07-01", periods=24, freq="h", tz="UTC")
        out_out = EnergyCrisisWindowProvider().build(idx_out)
        assert (out_out["energy_saving_window"] == 0.0).all()
        ```
    """

    name = "energy_saving_window"
    csv_filename = "energy_saving_windows_de.csv"
    column = "energy_saving_window"


# Maps a configuration flag name -> the provider constructed when it is True.
# To add a new exogenous driver: implement an ExogFeatureProvider and add one
# entry here, then declare the matching boolean flag on the config classes.
EXOG_PROVIDER_REGISTRY: Dict[str, Callable[..., ExogFeatureProvider]] = {
    "include_covid_infection_rate": CovidInfectionRateProvider,
    "include_entsoe_forecast_load": EntsoeForecastLoadProvider,
    "include_entsoe_renewable_forecast": EntsoeRenewableForecastProvider,
    "include_entsoe_net_load": EntsoeNetLoadProvider,
    "include_entsoe_day_ahead_price": EntsoeDayAheadPriceProvider,
    "include_football_match_window": FootballMatchWindowProvider,
    "include_energy_saving_window": EnergyCrisisWindowProvider,
}


def build_providers(
    flags: Mapping[str, bool],
    *,
    data_home: DataHome = None,
    max_gap: int = 0,
    max_tail_gap: int = 0,
    provider_window: Optional[pd.DatetimeIndex] = None,
) -> List[ExogFeatureProvider]:
    """Construct the providers whose flags are truthy, in registry order.

    Args:
        flags: Mapping from registry flag name to a boolean. Unknown keys are
            ignored; missing keys are treated as ``False``.
        data_home: Root data directory forwarded to each provider.
        max_gap: Maximum contiguous missing-value run forwarded to each
            provider. See `_align_to_index`. Defaults to ``0``.
        max_tail_gap: Extended trailing-edge healing budget forwarded to each
            provider. See `_align_to_index`. Defaults to ``0``.
        provider_window: Validation index forwarded to each provider as
            *provider_window*. See `_align_to_index`. Defaults to ``None``.

    Returns:
        List[ExogFeatureProvider]: Providers for the enabled flags, in the fixed
        order of `EXOG_PROVIDER_REGISTRY` (deterministic column ordering).

    Examples:
        ```{python}
        from spotforecast2_safe.preprocessing.exog_providers import build_providers

        providers = build_providers({"include_entsoe_forecast_load": True})
        print([p.name for p in providers])
        ```
    """
    providers: List[ExogFeatureProvider] = []
    for flag, factory in EXOG_PROVIDER_REGISTRY.items():
        if flags.get(flag, False):
            providers.append(
                factory(
                    data_home=data_home,
                    max_gap=max_gap,
                    max_tail_gap=max_tail_gap,
                    provider_window=provider_window,
                )
            )
    return providers


def build_providers_from_config(
    config: Union["ConfigEntsoe", "ConfigMulti", object],
    *,
    data_home: DataHome = None,
    provider_window: Optional[pd.DatetimeIndex] = None,
) -> List[ExogFeatureProvider]:
    """Construct providers by reading the registry flags off a config object.

    Args:
        config: A config object (e.g. ``ConfigEntsoe`` / ``ConfigMulti``) whose
            attributes include the `EXOG_PROVIDER_REGISTRY` flag names.
        data_home: Root data directory forwarded to each provider.
        provider_window: Validation index forwarded to each provider. Overrides
            the per-provider window; ``None`` uses the full request index.

    Returns:
        List[ExogFeatureProvider]: Providers for the flags set to ``True`` on
        *config*.

    Examples:
        ```{python}
        from spotforecast2_safe.preprocessing.exog_providers import (
            build_providers_from_config,
        )

        class SimpleConfig:
            include_covid_infection_rate = True
            include_entsoe_forecast_load = False
            include_entsoe_renewable_forecast = False
            include_entsoe_net_load = False
            include_entsoe_day_ahead_price = False
            exog_max_gap_hours = 0
            exog_max_tail_gap_hours = 0

        providers = build_providers_from_config(SimpleConfig())
        print([p.name for p in providers])
        assert len(providers) == 1
        assert providers[0].name == "covid_infection_rate"
        ```
    """
    flags = {
        flag: bool(getattr(config, flag, False)) for flag in EXOG_PROVIDER_REGISTRY
    }
    max_gap = int(getattr(config, "exog_max_gap_hours", 0))
    max_tail_gap = int(getattr(config, "exog_max_tail_gap_hours", 0))
    return build_providers(
        flags,
        data_home=data_home,
        max_gap=max_gap,
        max_tail_gap=max_tail_gap,
        provider_window=provider_window,
    )

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
  (``max_gap``), which is all-or-nothing per gap and logged at WARNING with
  count and span.
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
    """


def _align_to_index(
    frame: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    provider: str,
    max_gap: int = 0,
    validate_index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Reindex *frame* onto *index*, optionally healing small gaps (fail-safe).

    Harmonises timezone awareness with *index*, drops duplicate timestamps
    (keeping the last), reindexes exactly onto *index*, and refuses to return a
    frame that still contains NaN within the validated window — that would mean
    the provider cannot cover the requested range. The result is cast to
    ``float32`` to match the dtype the recursive forecaster expects for
    exogenous columns.

    Args:
        frame: Source feature frame with a ``DatetimeIndex``.
        index: Target hourly index the features must align to.
        provider: Provider name, used in log messages and the error.
        max_gap: Maximum length, in hours, of a contiguous run of missing
            values that will be healed before the provider is rejected.
            Interior gaps are time-interpolated; leading/trailing edge gaps
            are back-/forward-filled. ``0`` (default) keeps the strict
            fail-safe (any gap raises). Healed runs are logged at WARNING
            with count and span.
        validate_index: If given, the NaN gate checks only the intersection
            of *index* with this index. Timestamps outside the validation
            window may carry NaN and are still returned (as ``float32`` NaN).
            ``None`` (default) validates the full *index*, matching prior
            behaviour exactly.

    Returns:
        pd.DataFrame: *frame* reindexed onto *index*, ``float32``. NaN-free
        within the validated window (after healing when ``max_gap > 0``);
        out-of-window cells remain NaN when *validate_index* is supplied.

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

    if max_gap > 0 and bool(aligned.isna().any().any()):
        # Heal bounded gaps, all-or-nothing PER RUN: a contiguous NaN run no
        # longer than max_gap is healed in full; a longer run is left
        # entirely NaN (no partial fabrication) and still trips the NaN gate
        # below if it intersects the validated window. An oversize run
        # outside the window (e.g. a price series that starts years after
        # data_start) therefore does not block healing of small in-window
        # gaps.
        pre_heal_na = aligned.isna()
        healed = aligned.interpolate(
            method="time",
            limit=max_gap,
            limit_area="inside",
            limit_direction="both",
        )
        healed = healed.bfill(limit=max_gap)
        healed = healed.ffill(limit=max_gap)
        # Re-mask runs longer than max_gap: the interpolate/bfill/ffill
        # ``limit`` caps fill steps per direction, so an oversize run could
        # otherwise be partially (or, healed from both sides, even fully)
        # filled.
        for col in aligned.columns:
            na = pre_heal_na[col]
            if not bool(na.any()):
                continue
            # cumsum trick: consecutive NaN cells share one group label
            # derived from the count of preceding non-NaN cells.
            run_id = (~na).cumsum()[na]
            run_len = run_id.groupby(run_id).transform("size")
            oversize_idx = run_len.index[run_len > max_gap]
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
                "%s: healed %d missing cell(s) in %d gap(s) (max_gap=%d); spans: %s",
                provider,
                n_healed,
                n_gaps,
                max_gap,
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
    frame on that exact index. Subclasses set :attr:`name` (a short identifier
    used in logs and as the default column name) and implement :meth:`build`.

    Implementations should load their backing data lazily inside :meth:`build`
    and raise :class:`ExogProviderError` when the data is missing or cannot
    cover the requested range, so the fail-safe policy lives in one place.
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
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.column = column
        self.fill_outside = float(fill_outside)
        self.max_gap = max_gap
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
        daily = self._load_daily()
        if len(index) == 0:
            return pd.DataFrame({self.column: pd.Series(dtype="float32")}, index=index)
        if daily.empty:
            return _align_to_index(
                pd.DataFrame({self.column: self.fill_outside}, index=index),
                index,
                provider=self.name,
                max_gap=self.max_gap,
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
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.max_gap = max_gap
        self.provider_window = provider_window

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
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
            See :func:`_align_to_index` for full semantics. Defaults to ``0``.
        provider_window: Validation index passed to ``_align_to_index`` as
            *validate_index*. See :func:`_align_to_index`. Defaults to ``None``.
    """

    name = "entsoe_renewable_forecast"
    wind_col = "entsoe_wind_forecast"
    solar_col = "entsoe_solar_forecast"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        max_gap: int = 0,
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.max_gap = max_gap
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
        return _align_to_index(
            self._load(),
            index,
            provider=self.name,
            max_gap=self.max_gap,
            validate_index=self.provider_window,
        )


class EntsoeNetLoadProvider(ExogFeatureProvider):
    """ENTSO-E day-ahead net load = Forecasted Load − (wind + solar) forecast.

    Combines the day-ahead Forecasted Load with the day-ahead renewable
    forecast to form the net-load prior the residual is often modelled against.
    Both inputs are day-ahead (leakage-clean). Raises
    :class:`ExogProviderError` if either input is unavailable.

    Args:
        data_home: Root data directory forwarded to the loaders.
        max_gap: Maximum contiguous missing-value run healed by ``_align_to_index``.
            See :func:`_align_to_index` for full semantics. Defaults to ``0``.
        provider_window: Validation index passed to ``_align_to_index`` as
            *validate_index*. See :func:`_align_to_index`. Defaults to ``None``.
    """

    name = "entsoe_net_load"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        max_gap: int = 0,
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.max_gap = max_gap
        self.provider_window = provider_window

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
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
            See :func:`_align_to_index` for full semantics. Defaults to ``0``.
        provider_window: Validation index passed to ``_align_to_index`` as
            *validate_index*. See :func:`_align_to_index`. Defaults to ``None``.
    """

    name = "entsoe_day_ahead_price"

    def __init__(
        self,
        *,
        data_home: DataHome = None,
        max_gap: int = 0,
        provider_window: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.data_home = data_home
        self.max_gap = max_gap
        self.provider_window = provider_window

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
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
            validate_index=self.provider_window,
        )


# Maps a configuration flag name -> the provider constructed when it is True.
# To add a new exogenous driver: implement an ExogFeatureProvider and add one
# entry here, then declare the matching boolean flag on the config classes.
EXOG_PROVIDER_REGISTRY: Dict[str, Callable[..., ExogFeatureProvider]] = {
    "include_covid_infection_rate": CovidInfectionRateProvider,
    "include_entsoe_forecast_load": EntsoeForecastLoadProvider,
    "include_entsoe_renewable_forecast": EntsoeRenewableForecastProvider,
    "include_entsoe_net_load": EntsoeNetLoadProvider,
    "include_entsoe_day_ahead_price": EntsoeDayAheadPriceProvider,
}


def build_providers(
    flags: Mapping[str, bool],
    *,
    data_home: DataHome = None,
    max_gap: int = 0,
    provider_window: Optional[pd.DatetimeIndex] = None,
) -> List[ExogFeatureProvider]:
    """Construct the providers whose flags are truthy, in registry order.

    Args:
        flags: Mapping from registry flag name to a boolean. Unknown keys are
            ignored; missing keys are treated as ``False``.
        data_home: Root data directory forwarded to each provider.
        max_gap: Maximum contiguous missing-value run forwarded to each
            provider. See :func:`_align_to_index`. Defaults to ``0``.
        provider_window: Validation index forwarded to each provider as
            *provider_window*. See :func:`_align_to_index`. Defaults to ``None``.

    Returns:
        List[ExogFeatureProvider]: Providers for the enabled flags, in the fixed
        order of :data:`EXOG_PROVIDER_REGISTRY` (deterministic column ordering).

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
            attributes include the :data:`EXOG_PROVIDER_REGISTRY` flag names.
        data_home: Root data directory forwarded to each provider.
        provider_window: Validation index forwarded to each provider. Overrides
            the per-provider window; ``None`` uses the full request index.

    Returns:
        List[ExogFeatureProvider]: Providers for the flags set to ``True`` on
        *config*.
    """
    flags = {
        flag: bool(getattr(config, flag, False)) for flag in EXOG_PROVIDER_REGISTRY
    }
    max_gap = int(getattr(config, "exog_max_gap_hours", 0))
    return build_providers(
        flags,
        data_home=data_home,
        max_gap=max_gap,
        provider_window=provider_window,
    )

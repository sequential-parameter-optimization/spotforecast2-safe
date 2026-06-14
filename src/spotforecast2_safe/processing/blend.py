# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-hoc blend of a model forecast with an external prior — pure computation.

Motivation (2026-06-13 team_4 post-mortem): the ``--entsoe`` variant fed the
ENTSO-E day-ahead *Forecasted Load* in as a near-oracle model **feature** and
did *worse* on a low-load Saturday because that prior was itself biased high.
The obvious "down-weight the prior" idea — scaling the feature column — is a
**no-op** for gradient-boosted trees: tree splits are invariant to any
monotonic rescaling of a single feature.  The sound way to down-weight a prior
is therefore a *post-hoc convex blend* of the trained model's forecast with the
prior, which this module provides.  The operator keeps the prior out of the
model (or in it) and tunes its influence at the output stage.
"""

from __future__ import annotations

import pandas as pd


def blend_with_prior(
    model_forecast: pd.Series,
    prior: pd.Series,
    *,
    weight: float,
) -> pd.Series:
    """Convex-blend a model forecast with an external prior.

    Returns ``(1 - weight) * model_forecast + weight * prior`` on the index
    intersection of the two series.  ``weight`` is the trust placed in the
    prior: ``0.0`` returns the model forecast unchanged (prior ignored),
    ``1.0`` returns the prior, and intermediate values interpolate.  This is the
    correct lever for down-weighting a near-oracle prior whose influence a
    tree model cannot be tuned through feature scaling.

    The function is **pure**: it does not mutate its inputs and emits no
    warnings.  The result carries ``model_forecast``'s name.

    Args:
        model_forecast: The trained model's forecast.
        prior: The external prior to blend in (e.g. the ENTSO-E day-ahead
            forecast), aligned by index.
        weight: Blend weight in ``[0.0, 1.0]`` — the trust placed in ``prior``.

    Returns:
        A new ``pd.Series`` over the index intersection, named like
        ``model_forecast``.

    Raises:
        TypeError: When ``model_forecast`` or ``prior`` is not a ``pd.Series``.
        ValueError: When ``weight`` is outside ``[0.0, 1.0]`` or the two series
            share no index positions.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.processing.blend import blend_with_prior

        idx = pd.date_range("2026-06-13 00:00", periods=4, freq="h", tz="UTC")
        model = pd.Series([100.0, 110.0, 120.0, 130.0], index=idx, name="y0")
        prior = pd.Series([140.0, 140.0, 140.0, 140.0], index=idx)

        # weight=0 -> model unchanged; weight=1 -> prior; 0.25 -> 75/25 mix.
        print(blend_with_prior(model, prior, weight=0.0).tolist())
        print(blend_with_prior(model, prior, weight=1.0).tolist())
        print(blend_with_prior(model, prior, weight=0.25).tolist())
        assert blend_with_prior(model, prior, weight=0.0).equals(model)
        ```
    """
    if not isinstance(model_forecast, pd.Series):
        raise TypeError(
            f"model_forecast must be a pd.Series, got "
            f"{type(model_forecast).__name__!r}."
        )
    if not isinstance(prior, pd.Series):
        raise TypeError(f"prior must be a pd.Series, got {type(prior).__name__!r}.")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0.0, 1.0], got {weight}.")

    common = model_forecast.index.intersection(prior.index)
    if len(common) == 0:
        raise ValueError("model_forecast and prior share no index positions.")

    blended = (1.0 - weight) * model_forecast.loc[common] + weight * prior.loc[common]
    blended.name = model_forecast.name
    return blended

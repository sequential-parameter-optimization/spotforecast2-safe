# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fourier-based deterministic-process feature builder.

Wraps `statsmodels.tsa.deterministic.DeterministicProcess` with a
Fourier seasonal basis for arbitrary period lengths.  Companion to
`ExogBuilder` for users who prefer a Fourier representation over a
RepeatingBasisFunction.
"""

from typing import Sequence

import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess, Fourier


def build_deterministic_process(
    index: pd.DatetimeIndex,
    *,
    periods: Sequence[int],
    fourier_order: int = 3,
    constant: bool = True,
    trend_order: int = 1,
) -> DeterministicProcess:
    """Construct a Fourier-based deterministic process for a date index.

    Args:
        index: Datetime index the process will index against.
        periods: One Fourier basis is added per entry, with the given
            period length (e.g. ``24`` for a daily cycle on an hourly
            index, ``168`` for a weekly cycle).
        fourier_order: Number of sine/cosine pairs per period.
        constant: Whether to include a bias (y-intercept) column.
        trend_order: Polynomial trend order.  ``1`` yields a linear trend;
            ``0`` disables trend; higher values add quadratic / cubic terms.

    Returns:
        A configured `DeterministicProcess`.  Call ``.in_sample()`` to
        materialise the feature DataFrame.

    Examples:
        ```{python}
        import pandas as pd

        from spotforecast2_safe.preprocessing.deterministic import (
            build_deterministic_process,
        )

        idx = pd.date_range("2026-01-01", periods=48, freq="h")
        dp = build_deterministic_process(idx, periods=[24], fourier_order=2)
        features = dp.in_sample()
        print(features.shape)
        ```
    """
    fouriers = [Fourier(period=period, order=fourier_order) for period in periods]
    return DeterministicProcess(
        index=index,
        constant=constant,
        order=trend_order,
        seasonal=False,
        additional_terms=fouriers,
        drop=True,
    )

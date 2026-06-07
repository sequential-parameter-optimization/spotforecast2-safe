# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Spectral diagnostics — periodogram of a time series."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram

from spotforecast2_safe.preprocessing.checking import check_y


@dataclass(frozen=True)
class PeriodogramResult:
    """Container for the output of `compute_periodogram()`.

    Attributes:
        spectrum: ``pd.DataFrame`` indexed by frequency with one column
            ``"power"``.  Frequency is expressed in units of "cycles per
            sample" (the convention used by `scipy.signal.periodogram`).
        top_periods: ``pd.Series`` of the top-k periods (1 / frequency)
            indexed by their spectral power, sorted from strongest to
            weakest peak.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd

        from spotforecast2_safe.stats.spectral import compute_periodogram

        rng = np.random.default_rng(0)
        t = np.arange(256)
        y = pd.Series(np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(256))
        result = compute_periodogram(y, max_peaks=3)
        print(type(result).__name__)
        print(result.spectrum.columns.tolist())
        assert isinstance(result.top_periods, pd.Series)
        assert len(result.top_periods) == 3
        print(f"Dominant period: {result.top_periods.index[0]:.1f} samples")
        ```
    """

    spectrum: pd.DataFrame
    top_periods: pd.Series


def compute_periodogram(
    y: pd.Series,
    *,
    max_peaks: int = 5,
    scaling: str = "spectrum",
    fs: Optional[float] = None,
) -> PeriodogramResult:
    """Compute the periodogram of a time series.

    Wraps `scipy.signal.periodogram` with a ``boxcar`` window to match
    the reference implementation.

    Args:
        y: Series to analyse.  Must be a non-empty pandas Series without
            missing values (validated via `check_y`).
        max_peaks: Number of top peaks to surface.
        scaling: Spectrum scaling forwarded to ``scipy``.
        fs: Sampling frequency.  Defaults to ``1.0`` (cycles per sample).

    Returns:
        A `PeriodogramResult` with the full spectrum and the
        top-``max_peaks`` periods.

    Raises:
        TypeError: If ``y`` is not a pandas Series.
        ValueError: If ``y`` contains missing values.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd

        from spotforecast2_safe.stats.spectral import compute_periodogram

        t = np.arange(512)
        y = pd.Series(np.sin(2 * np.pi * t / 32))
        result = compute_periodogram(y, max_peaks=1)
        print(int(round(result.top_periods.index[0])))
        ```
    """
    check_y(y)

    freqs, power = periodogram(
        y.to_numpy(),
        fs=fs if fs is not None else 1.0,
        window="boxcar",
        scaling=scaling,
    )

    spectrum = pd.DataFrame({"power": power}, index=pd.Index(freqs, name="frequency"))

    # Top-k peaks by power, skipping the DC component (frequency = 0).
    mask = freqs > 0
    freqs_nz = freqs[mask]
    power_nz = power[mask]
    if power_nz.size == 0:
        top_periods = pd.Series(dtype=float, name="power")
    else:
        order = np.argsort(power_nz)[::-1][:max_peaks]
        top_periods = pd.Series(
            power_nz[order],
            index=pd.Index(1.0 / freqs_nz[order], name="period"),
            name="power",
        )

    return PeriodogramResult(spectrum=spectrum, top_periods=top_periods)

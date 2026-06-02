# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``preprocessing.outlier.get_outliers``.

Covers the fail-safe empty-input guard and both ``data_original`` branches,
which the docstring example exercised only at doc-render time.
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.outlier import get_outliers


def _data_with_outliers():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "A": np.concatenate([rng.normal(0.0, 1.0, 100), [10.0, 11.0, 12.0]]),
            "B": np.concatenate([rng.normal(5.0, 2.0, 100), [100.0, 110.0, 120.0]]),
        }
    )


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        get_outliers(pd.DataFrame())


def test_returns_outliers_per_column_with_data_original():
    data = _data_with_outliers()
    out = get_outliers(data, data_original=data.copy(), contamination=0.03)
    assert set(out) == {"A", "B"}
    assert all(len(v) > 0 for v in out.values())


def test_data_original_none_uses_current_data():
    data = _data_with_outliers()
    out = get_outliers(data, data_original=None, contamination=0.03)
    assert set(out) == {"A", "B"}
    # With data_original=None the returned values come from `data` itself.
    for col, vals in out.items():
        assert vals.isin(data[col]).all()

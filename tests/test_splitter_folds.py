# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the CV splitters' ``split`` logic.

The headline property a time-series CV splitter must guarantee is **no future
leakage**: every fold's training window must end at or before its test window
starts. These tests assert that invariant directly for ``TimeSeriesFold`` across
refit/gap configurations, and exercise ``OneStepAheadFold.split`` (which had no
direct test) plus the splitters' fail-safe input guards.
"""

import pandas as pd
import pytest

from spotforecast2_safe.splitter import OneStepAheadFold, TimeSeriesFold


def _y(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(range(n), index=idx, name="y")


class TestTimeSeriesFoldLeakage:
    @pytest.mark.parametrize("refit", [False, True])
    @pytest.mark.parametrize("gap", [0, 5])
    def test_no_future_leakage(self, refit, gap):
        y = _y(200)
        cv = TimeSeriesFold(
            steps=24,
            initial_train_size=100,
            window_size=24,
            refit=refit,
            gap=gap,
            verbose=False,
        )
        folds = cv.split(y)
        assert len(folds) > 1
        for fold in folds:
            train_end = fold[1][1]  # exclusive
            test_start = fold[3][0]
            assert train_end <= test_start, fold

    def test_fold_stride_overlap_flagged(self):
        cv = TimeSeriesFold(steps=24, fold_stride=12, initial_train_size=100)
        assert cv.overlapping_folds is True


class TestTimeSeriesFoldGuards:
    def test_non_pandas_input_raises(self):
        with pytest.raises(TypeError, match="pandas Series"):
            TimeSeriesFold(steps=24, initial_train_size=100).split([1, 2, 3])

    def test_initial_train_none_window_none_raises(self):
        cv = TimeSeriesFold(steps=24, initial_train_size=None, window_size=None)
        with pytest.raises(ValueError, match="window_size"):
            cv.split(_y(100))

    def test_refit_with_initial_train_none_raises(self):
        cv = TimeSeriesFold(
            steps=24, initial_train_size=None, window_size=24, refit=True
        )
        with pytest.raises(ValueError, match="refit"):
            cv.split(_y(100))


class TestOneStepAheadFoldSplit:
    def test_split_returns_contiguous_train_test(self):
        y = _y(200)
        # OneStepAheadFold.split returns a single fold, not a list of folds.
        fold = OneStepAheadFold(
            initial_train_size=100, window_size=24, verbose=False
        ).split(y)
        train_start, train_end = fold[1]
        test_start, test_end = fold[2]
        assert train_start == 0
        assert train_end == test_start  # contiguous, no leakage
        assert test_end == len(y)
        assert fold[3] is True

    def test_split_as_pandas(self):
        df = OneStepAheadFold(
            initial_train_size=100, window_size=24, verbose=False
        ).split(_y(150), as_pandas=True)
        assert isinstance(df, pd.DataFrame)
        assert {"train_start", "train_end", "test_start", "test_end"}.issubset(
            df.columns
        )

    def test_non_pandas_input_raises(self):
        with pytest.raises(TypeError, match="pandas Series"):
            OneStepAheadFold(initial_train_size=100).split([1, 2, 3])

import pytest
import numpy as np
import pandas as pd
from spotforecast2_safe.model_selection import TimeSeriesFold
from spotforecast2_safe.model_selection import OneStepAheadFold


class TestTimeSeriesFold:

    def test_init_raises_exception_when_steps_is_not_int(self):
        """
        Test that TypeError is raised when steps is not an int.
        """
        with pytest.raises(
            ValueError, match="`steps` must be an integer greater than 0"
        ):
            TimeSeriesFold(steps="10")

    def test_init_raises_exception_when_steps_less_than_1(self):
        """
        Test that ValueError is raised when steps is less than 1.
        """
        with pytest.raises(
            ValueError, match="`steps` must be an integer greater than 0"
        ):
            TimeSeriesFold(steps=0)

    def test_split_with_range_index(self):
        """
        Test split with RangeIndex.
        """
        y = pd.Series(np.arange(10))
        cv = TimeSeriesFold(
            steps=2, initial_train_size=4, gap=0, fold_stride=2, window_size=2
        )
        folds = cv.split(X=y)

        # Expected folds:
        # Fold 0: Train [0-3], Last Window [2-3], Test [4-5]
        # Indices: 0, 1, 2, 3. 4 (exclusive).

        assert len(folds) == 3
        # Fold 0
        assert folds[0][0] == 0
        assert folds[0][1] == [0, 4]  # Train start, end (exclusive)
        assert folds[0][2] == [2, 4]  # Last window start, end (exclusive)
        assert folds[0][3] == [4, 6]  # Test start, end (exclusive)

        # Fold 1
        # refit=False, so training indices are copied from previous fold.
        # But Last Window moves with Test set.
        assert folds[1][0] == 1
        assert folds[1][1] == [0, 4]  # Copied from Fold 0 if refit=False
        assert folds[1][2] == [4, 6]  # Last Window updates!
        assert folds[1][3] == [6, 8]

        # Fold 2
        assert folds[2][0] == 2
        assert folds[2][1] == [0, 4]  # Copied
        assert folds[2][2] == [6, 8]  # Last Window updates
        assert folds[2][3] == [8, 10]

    def test_split_with_date_index_and_str_initial_train_size(self):
        """
        Test split with DatetimeIndex and initial_train_size as string.
        """
        y = pd.Series(
            np.arange(10), index=pd.date_range("2020-01-01", periods=10, freq="D")
        )
        cv = TimeSeriesFold(
            steps=2,
            initial_train_size="2020-01-04",
            gap=0,
            fold_stride=2,
            window_size=2,
        )
        folds = cv.split(X=y)

        assert len(folds) == 3
        # Fold 0
        assert folds[0][0] == 0
        assert folds[0][1] == [
            0,
            4,
        ]  # 2020-01-01 to 2020-01-04 is 4 observations (index 0, 1, 2, 3). End 4.
        assert folds[0][3] == [4, 6]

    def test_split_with_gap(self):
        """
        Test split with gap.
        """
        y = pd.Series(np.arange(12))
        cv = TimeSeriesFold(
            steps=2, initial_train_size=4, gap=1, fold_stride=2, window_size=2
        )
        # Train: 0, 1, 2, 3
        # Gap: 4
        # Test (effective): 5, 6

        folds = cv.split(X=y)

        # Fold 0
        assert folds[0][0] == 0
        assert folds[0][1] == [0, 4]
        # fold[3] includes gap -> [4, 7]
        assert folds[0][3] == [4, 7]
        # fold[4] excludes gap (effective test set) -> [5, 7]
        assert folds[0][4] == [5, 7]

        # Fold 1
        assert folds[1][0] == 1
        assert folds[1][1] == [0, 4]  # refit=False
        # Test starts at 6 + 1 (Gap at 6) -> 7, 8.
        # fold[3] -> [6, 9] (includes gap 6)
        assert folds[1][3] == [6, 9]

    def test_split_return_all_indexes(self):
        """
        Test split with return_all_indexes=True.
        """
        y = pd.Series(np.arange(10))
        cv = TimeSeriesFold(
            steps=2,
            initial_train_size=4,
            fold_stride=2,
            window_size=2,
            return_all_indexes=True,
        )
        folds = cv.split(X=y)

        # Ranges are used instead of [start, end]
        assert isinstance(folds[0][1], range)
        assert list(folds[0][1]) == [0, 1, 2, 3]
        # Test start 4, end 6 -> range(4, 6) -> 4, 5
        # fold[3] is test_range.
        assert isinstance(folds[0][3], range)
        assert list(folds[0][3]) == [4, 5]


class TestOneStepAheadFold:

    def test_init_raises_exception_when_initial_train_size_invalid(self):
        with pytest.raises(ValueError):
            OneStepAheadFold(initial_train_size=0)

    def test_split(self):
        """
        Test OneStepAheadFold split.
        """
        y = pd.Series(np.arange(6))
        cv = OneStepAheadFold(initial_train_size=3, window_size=2)
        # Train: 0, 1, 2

        folds = cv.split(X=y)

        # Fold 0: Train [0-3] (0, 1, 2), Test [3-6] (3, 4, 5)

        assert len(folds) == 4
        assert folds[0] == 0
        assert folds[1] == [0, 3]  # Train end exclusive
        assert folds[2] == [3, 6]  # Test start, end exclusive
        assert folds[3] is True

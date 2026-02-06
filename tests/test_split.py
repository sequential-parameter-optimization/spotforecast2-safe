import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.split import (
    split_abs_train_val_test,
    split_rel_train_val_test,
)


def _sample_data(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"a": range(n)}, index=idx)


def test_split_abs_train_val_test_basic(capsys):
    data = _sample_data(6)
    end_train = data.index[2]
    end_validation = data.index[4]

    train, val, test = split_abs_train_val_test(
        data, end_train=end_train, end_validation=end_validation, verbose=True
    )

    captured = capsys.readouterr().out
    assert "Start date" in captured
    assert "End date" in captured

    assert train.index.min() == data.index[0]
    assert train.index.max() == end_train
    assert val.index.min() == end_train
    assert val.index.max() == end_validation
    assert test.index.min() == end_validation
    assert test.index.max() == data.index[-1]

    assert len(train) + len(val) + len(test) >= len(data)


def test_split_rel_train_val_test_basic_sizes():
    data = _sample_data(10)

    train, val, test = split_rel_train_val_test(
        data, perc_train=0.6, perc_val=0.2, verbose=False
    )

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert len(train) + len(val) + len(test) == len(data)


def test_split_rel_train_val_test_rounding_safe():
    data = _sample_data(7)

    train, val, test = split_rel_train_val_test(
        data, perc_train=0.33, perc_val=0.33, verbose=False
    )

    assert len(train) + len(val) + len(test) == len(data)
    assert train.index.max() == data.index[len(train) - 1]
    assert val.index.min() == data.index[len(train)]
    assert test.index.min() == data.index[len(train) + len(val)]


def test_split_rel_train_val_test_invalid_percentages():
    data = _sample_data(5)

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_rel_train_val_test(data, perc_train=-0.1, perc_val=0.5)

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_rel_train_val_test(data, perc_train=0.5, perc_val=1.1)

    with pytest.raises(ValueError, match="sum to 1 or less"):
        split_rel_train_val_test(data, perc_train=0.8, perc_val=0.3)


def test_split_rel_train_val_test_empty_data():
    data = pd.DataFrame({"a": []})
    with pytest.raises(ValueError, match="Input data is empty"):
        split_rel_train_val_test(data, perc_train=0.7, perc_val=0.2)


def test_split_rel_train_val_test_all_train():
    data = _sample_data(6)

    train, val, test = split_rel_train_val_test(
        data, perc_train=1.0, perc_val=0.0, verbose=False
    )

    assert len(train) == len(data)
    assert len(val) == 0
    assert len(test) == 0


def test_split_rel_train_val_test_all_val():
    data = _sample_data(6)

    train, val, test = split_rel_train_val_test(
        data, perc_train=0.0, perc_val=1.0, verbose=False
    )

    assert len(train) == 0
    assert len(val) == len(data)
    assert len(test) == 0


def test_split_rel_train_val_test_all_test():
    data = _sample_data(6)

    train, val, test = split_rel_train_val_test(
        data, perc_train=0.0, perc_val=0.0, verbose=False
    )

    assert len(train) == 0
    assert len(val) == 0
    assert len(test) == len(data)


def test_split_rel_train_val_test_empty_splits():
    """Test that individual splits can be empty while sum equals 1.0."""
    data = _sample_data(10)

    # Empty test set (sum = 1.0)
    train, val, test = split_rel_train_val_test(
        data, perc_train=0.7, perc_val=0.3, verbose=False
    )
    assert len(train) == 7
    assert len(val) == 3
    assert len(test) == 0
    assert len(train) + len(val) + len(test) == len(data)

    # Empty train set
    train, val, test = split_rel_train_val_test(
        data, perc_train=0.0, perc_val=0.6, verbose=False
    )
    assert len(train) == 0
    assert len(val) == 6
    assert len(test) == 4
    assert len(train) + len(val) + len(test) == len(data)

    # Empty val set
    train, val, test = split_rel_train_val_test(
        data, perc_train=0.8, perc_val=0.0, verbose=False
    )
    assert len(train) == 8
    assert len(val) == 0
    assert len(test) == 2
    assert len(train) + len(val) + len(test) == len(data)

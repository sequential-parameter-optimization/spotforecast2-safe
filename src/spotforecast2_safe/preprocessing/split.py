import pandas as pd


def split_abs_train_val_test(
    data: pd.DataFrame,
    end_train: pd.Timestamp,
    end_validation: pd.Timestamp,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a time series DataFrame into training, validation, and test sets based on absolute timestamps.

    Args:
        data (pd.DataFrame): The time series data with a DateTimeIndex.
        end_train (pd.Timestamp): The end date for the training set.
        end_validation (pd.Timestamp): The end date for the validation set.

    Returns:
        tuple: A tuple containing:
            - data_train (pd.DataFrame): The training set.
            - data_val (pd.DataFrame): The validation set.
            - data_test (pd.DataFrame): The test set.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_data
        >>> from spotforecast2.preprocessing.split import split_train_val_test
        >>> data = fetch_data()
        >>> end_train = pd.Timestamp('2020-12-31 23:00:00')
        >>> end_validation = pd.Timestamp('2021-06-30 23:00:00')
        >>> data_train, data_val, data_test = split_train_val_test(
        ...     data,
        ...     end_train=end_train,
        ...     end_validation=end_validation,
        ...     verbose=True
        ... )
    """
    data = data.copy()
    start_date = data.index.min()
    end_date = data.index.max()
    if verbose:
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")
    data_train = data.loc[:end_train, :].copy()
    data_val = data.loc[end_train:end_validation, :].copy()
    data_test = data.loc[end_validation:, :].copy()

    if verbose:
        print(
            f"Train: {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})"
        )
        print(
            f"Val: {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})"
        )
        print(
            f"Test: {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})"
        )

    return data_train, data_val, data_test


def split_rel_train_val_test(
    data: pd.DataFrame,
    perc_train: float,
    perc_val: float,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a time series DataFrame into training, validation, and test sets by percentages.

    The test percentage is computed as 1 - perc_train - perc_val.
    Sizes are rounded to ensure the splits sum to the full dataset size.

    Args:
        data (pd.DataFrame): The time series data with a DateTimeIndex.
        perc_train (float): Fraction of data used for training.
        perc_val (float): Fraction of data used for validation.
        verbose (bool): Whether to print additional information.

    Returns:
        tuple: A tuple containing:
            - data_train (pd.DataFrame): The training set.
            - data_val (pd.DataFrame): The validation set.
            - data_test (pd.DataFrame): The test set.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_data
        >>> from spotforecast2.preprocessing.split import split_rel_train_val_test
        >>> data = fetch_data()
        >>> data_train, data_val, data_test = split_rel_train_val_test(
        ...     data,
        ...     perc_train=0.7,
        ...     perc_val=0.2,
        ...     verbose=True
        ... )
    """
    data = data.copy()
    if data.shape[0] == 0:
        raise ValueError("Input data is empty.")
    if not (0 <= perc_train <= 1) or not (0 <= perc_val <= 1):
        raise ValueError("perc_train and perc_val must be between 0 and 1 (inclusive).")

    perc_test = 1 - perc_train - perc_val
    if verbose:
        print(
            f"Splitting data into train/val/test with percentages: "
            f"{perc_train:.4%} / {perc_val:.4%} / {perc_test:.4%}"
        )
    if round(perc_test, 10) < 0.0:
        print(
            f"Splitting data into train/val/test with percentages: "
            f"{perc_train:.4%} / {perc_val:.4%} / {perc_test:.4%}"
        )
        raise ValueError(
            "perc_train and perc_val must sum to 1 or less to leave room for a test set."
        )

    n_total = len(data)
    n_train = int(round(n_total * perc_train))
    n_val = int(round(n_total * perc_val))
    n_test = n_total - n_train - n_val

    if n_test < 0:
        n_test = 0
        n_val = n_total - n_train
    if n_val < 0:
        n_val = 0
        n_train = n_total

    end_train_idx = n_train
    end_val_idx = n_train + n_val

    data_train = data.iloc[:end_train_idx, :].copy()
    data_val = data.iloc[end_train_idx:end_val_idx, :].copy()
    data_test = data.iloc[end_val_idx:, :].copy()

    if verbose:
        print(f"Train size: {len(data_train)} ({len(data_train) / n_total:.2%})")
        print(f"Val size: {len(data_val)} ({len(data_val) / n_total:.2%})")
        print(f"Test size: {len(data_test)} ({len(data_test) / n_total:.2%})")

    return data_train, data_val, data_test

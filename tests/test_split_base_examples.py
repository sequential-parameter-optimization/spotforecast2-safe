# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from spotforecast2_safe.model_selection import TimeSeriesFold


def test_validate_params_example():
    """
    Test the example from BaseFold._validate_params docstring.
    Note: The original example cv = TimeSeriesFold() might fail because steps is required.
    """
    # The corrected example code:
    cv = TimeSeriesFold(steps=1)

    cv._validate_params(
        cv_name="TimeSeriesFold",
        steps=1,
        initial_train_size=1,
        fold_stride=1,
        window_size=1,
        differentiation=1,
        refit=False,
        fixed_train_size=True,
        gap=0,
        skip_folds=None,
        allow_incomplete_fold=True,
        return_all_indexes=False,
        verbose=True,
    )

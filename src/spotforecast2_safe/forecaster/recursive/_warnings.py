# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause


class DataTransformationWarning(UserWarning):
    """
    Warning used when data transformation is not possible or changes the data in
    an unexpected way.
    """

    pass


class ResidualsUsageWarning(UserWarning):
    """
    Warning used when the residuals are used in a way that is not recommended.
    """

    pass

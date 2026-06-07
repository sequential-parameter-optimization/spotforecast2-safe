# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause


class DataTransformationWarning(UserWarning):
    """Warning used when data transformation is not possible or changes the data in
    an unexpected way.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.forecaster.recursive._warnings import (
            DataTransformationWarning,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn(
                "Transformation could not be applied.", DataTransformationWarning
            )

        assert len(caught) == 1
        assert issubclass(caught[0].category, DataTransformationWarning)
        print(caught[0].category.__name__)
        ```
    """

    pass


class ResidualsUsageWarning(UserWarning):
    """Warning used when the residuals are used in a way that is not recommended.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.forecaster.recursive._warnings import (
            ResidualsUsageWarning,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn(
                "Residuals are used in a non-recommended way.",
                ResidualsUsageWarning,
            )

        assert len(caught) == 1
        assert issubclass(caught[0].category, ResidualsUsageWarning)
        print(caught[0].category.__name__)
        ```
    """

    pass

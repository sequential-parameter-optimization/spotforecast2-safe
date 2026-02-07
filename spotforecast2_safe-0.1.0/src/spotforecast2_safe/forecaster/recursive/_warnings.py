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

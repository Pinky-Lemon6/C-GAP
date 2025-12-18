class CGAPError(Exception):
    """Base exception for C-GAP."""


class DataValidationError(CGAPError):
    pass


class PipelineError(CGAPError):
    pass

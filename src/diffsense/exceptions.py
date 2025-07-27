"""
Custom exceptions
"""


class DiffSenseError(Exception):
    """Base exception"""
    pass


class ModelError(DiffSenseError):
    """Issues with the LLM model"""
    pass


class DiffError(DiffSenseError):
    """Issues computing diffs"""
    pass


class FormattingError(DiffSenseError):
    """Issues formatting output"""
    pass
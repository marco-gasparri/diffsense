__version__ = "0.3.0"  # Updated with Git integration

# Package metadata
__author__ = "Marco Gasparri"
__description__ = "AI-powered code diff tool with Git integration"

# Public API
from .diff_engine import DiffEngine, DiffBlock, LineDiff, ChangeType
from .formatter import DiffFormatter
from .llm_manager import LLMManager
from .git_manager import GitManager
from .exceptions import (
    DiffSenseError,
    ModelError,
    DiffError,
    FormattingError,
    GitError
)

__all__ = [
    # Version
    "__version__",

    # Main classes
    "DiffEngine",
    "DiffFormatter",
    "LLMManager",
    "GitManager",

    # Data classes
    "DiffBlock",
    "LineDiff",
    "ChangeType",

    # Exceptions
    "DiffSenseError",
    "ModelError",
    "DiffError",
    "FormattingError",
    "GitError",
]
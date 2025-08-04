__version__ = "0.5.0"

# Package metadata
__author__ = "Marco Gasparri"
__description__ = "AI-powered code diff & merge-conflict resolution tool with Git integration"

# Public API
from .diff_engine import DiffEngine, DiffBlock, LineDiff, ChangeType
from .formatter import DiffFormatter
from .llm_manager import LLMManager
from .git_manager import GitManager
from .model_providers import ModelProvider, create_provider
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
    "ConflictResolver",
    "ConflictParser",

    # Provider system
    "ModelProvider",
    "create_provider",

    # Data classes
    "DiffBlock",
    "LineDiff",
    "ChangeType",
    "ConflictResolution",
    "ConflictSection",

    # Exceptions
    "DiffSenseError",
    "ModelError",
    "DiffError",
    "FormattingError",
    "GitError"
]
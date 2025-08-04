"""
Git merge conflict resolution with AI assistance
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from enum import Enum

from .exceptions import DiffSenseError

logger = logging.getLogger(__name__)


class ConflictResolutionConfidence(Enum):
    """Confidence levels for conflict resolution"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConflictSection:
    """
    Represents a single conflict section in a file

    Attributes:
        start_line: Line number where conflict starts (1-based)
        end_line: Line number where conflict ends (1-based)
        current_content: Content from current branch (between <<<<<<< and =======)
        incoming_content: Content from incoming branch (between ======= and >>>>>>>)
        current_marker: The full <<<<<<< marker line with branch info
        incoming_marker: The full >>>>>>> marker line with branch info
        separator_line: Line number of ======= separator
    """
    start_line: int
    end_line: int
    current_content: str
    incoming_content: str
    current_marker: str
    incoming_marker: str
    separator_line: int

    @property
    def context_before_line(self) -> int:
        """Get line number for context before conflict"""
        return max(1, self.start_line - 1)

    @property
    def context_after_line(self) -> int:
        """Get line number for context after conflict"""
        return self.end_line + 1


@dataclass
class ConflictResolution:
    """
    Represents a resolution for a conflict section

    Attributes:
        section: The conflict section being resolved
        resolved_content: The resolved content (without markers)
        explanation: Explanation of why this resolution was chosen
        confidence: Confidence level of the resolution
        alternative_resolutions: Optional alternative resolutions if confidence is not high
    """
    section: ConflictSection
    resolved_content: str
    explanation: str
    confidence: ConflictResolutionConfidence
    alternative_resolutions: Optional[List[Dict[str, str]]] = None


class ConflictParser:
    """
    Parser for Git merge conflict files
    """

    # Regex patterns for conflict markers
    CONFLICT_START_PATTERN = re.compile(r'^<{7}(.*)$', re.MULTILINE)
    CONFLICT_SEPARATOR_PATTERN = re.compile(r'^={7}$', re.MULTILINE)
    CONFLICT_END_PATTERN = re.compile(r'^>{7}(.*)$', re.MULTILINE)

    def __init__(self):
        """Initialize the conflict parser"""
        pass

    def has_conflicts(self, content: str) -> bool:
        """
        Check if content contains merge conflicts
        Returns True if conflicts are present, False otherwise
        """
        return (
            bool(self.CONFLICT_START_PATTERN.search(content)) and
            bool(self.CONFLICT_SEPARATOR_PATTERN.search(content)) and
            bool(self.CONFLICT_END_PATTERN.search(content))
        )

    def parse_conflicts(self, content: str) -> List[ConflictSection]:
        """
        Parse all conflict sections from file content
        Returns a List of ConflictSection objects
        """
        lines = content.splitlines(keepends=True)
        conflicts = []
        i = 0

        while i < len(lines):
            # Look for conflict start
            line = lines[i]
            start_match = self.CONFLICT_START_PATTERN.match(line.rstrip('\n'))

            if start_match:
                conflict_start = i
                current_marker = line.rstrip('\n')
                current_content_lines = []

                # Find separator
                i += 1
                separator_line = -1
                while i < len(lines):
                    if self.CONFLICT_SEPARATOR_PATTERN.match(lines[i].rstrip('\n')):
                        separator_line = i
                        break
                    current_content_lines.append(lines[i])
                    i += 1

                # Check if separator was found
                if separator_line == -1:
                    raise DiffSenseError(f"Malformed conflict: no separator found after line {conflict_start + 1}")

                # Find end marker
                incoming_content_lines = []
                i += 1
                conflict_end = -1
                incoming_marker = None
                while i < len(lines):
                    end_match = self.CONFLICT_END_PATTERN.match(lines[i].rstrip('\n'))
                    if end_match:
                        conflict_end = i
                        incoming_marker = lines[i].rstrip('\n')
                        break
                    incoming_content_lines.append(lines[i])
                    i += 1

                # Check if end marker was found
                if conflict_end == -1 or incoming_marker is None:
                    raise DiffSenseError(f"Malformed conflict: no end marker found after line {separator_line + 1}")

                # Create conflict section
                conflict = ConflictSection(
                    start_line=conflict_start + 1,  # Convert to 1-based
                    end_line=conflict_end + 1,      # Convert to 1-based
                    current_content=''.join(current_content_lines).rstrip('\n'),
                    incoming_content=''.join(incoming_content_lines).rstrip('\n'),
                    current_marker=current_marker,
                    incoming_marker=incoming_marker,
                    separator_line=separator_line + 1  # Convert to 1-based
                )
                conflicts.append(conflict)

            i += 1

        return conflicts

    def extract_file_context(self, content: str, conflicts: List[ConflictSection]) -> Dict[str, Any]:
        """
        Extract context around conflicts for better analysis
        Returns a Dictionary with context information
        """
        lines = content.splitlines()
        context = {
            "total_lines": len(lines),
            "total_conflicts": len(conflicts),
            "file_language": self._detect_language(content),
            "conflicts_context": []
        }

        for conflict in conflicts:
            # Get context before and after
            context_before_start = max(0, conflict.start_line - 11)  # 10 lines before
            context_before = lines[context_before_start:conflict.start_line - 1]

            context_after_end = min(len(lines), conflict.end_line + 10)  # 10 lines after
            context_after = lines[conflict.end_line:context_after_end]

            context["conflicts_context"].append({
                "conflict_lines": f"{conflict.start_line}-{conflict.end_line}",
                "context_before": '\n'.join(context_before),
                "context_after": '\n'.join(context_after),
                "current_branch_info": conflict.current_marker.replace('<' * 7, '').strip(),
                "incoming_branch_info": conflict.incoming_marker.replace('>' * 7, '').strip()
            })

        return context

    def _detect_language(self, content: str) -> str:
        """
        Simple language detection based on content patterns
        Returns the detected language or 'unknown'
        """
        # Simple heuristics - can be expanded
        if 'import ' in content or 'from ' in content or 'def ' in content:
            return 'python'
        elif 'function ' in content or 'const ' in content or '=>' in content:
            return 'javascript'
        elif '#include' in content or 'int main' in content:
            return 'c/c++'
        elif 'public class' in content or 'private ' in content:
            return 'java'
        else:
            return 'unknown'


class ConflictResolver:
    """
    Orchestrates conflict resolution using AI
    """

    def __init__(self, llm_manager):
        """
        Initialize the conflict resolver
        """
        self.llm_manager = llm_manager
        self.parser = ConflictParser()

    def resolve_conflicts(
        self,
        file_path: Path,
        context_files: Optional[List[Path]] = None
    ) -> Tuple[List[ConflictResolution], int]:
        """
        Resolve conflicts in a file

        Args:
            file_path: Path to file with conflicts
            context_files: Optional additional files for context

        Returns: a tuple of (list of resolutions, total tokens used)
        """
        # Read main file
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise DiffSenseError(f"Cannot read file {file_path}: {e}")

        # Check for conflicts
        if not self.parser.has_conflicts(content):
            raise DiffSenseError(f"No merge conflicts found in {file_path}")

        # Parse conflicts
        conflicts = self.parser.parse_conflicts(content)
        logger.info(f"Found {len(conflicts)} conflict(s) in {file_path}")

        # Extract file context
        file_context = self.parser.extract_file_context(content, conflicts)

        # Read additional context files if provided
        additional_context = self._read_context_files(context_files) if context_files else None

        # Resolve each conflict
        resolutions = []
        total_tokens = 0

        for i, conflict in enumerate(conflicts):
            logger.debug(f"Resolving conflict {i + 1}/{len(conflicts)}")
            resolution, tokens = self.llm_manager.resolve_conflict(
                conflict=conflict,
                file_context=file_context,
                additional_context=additional_context,
                file_path=str(file_path)
            )
            resolutions.append(resolution)
            total_tokens += tokens

        return resolutions, total_tokens

    def _read_context_files(self, context_files: List[Path]) -> Dict[str, str]:
        """
        Read additional context files

        Args:
            context_files: List of paths to context files

        Returns a Dictionary mapping file paths to content
        """
        context = {}

        for file_path in context_files:
            try:
                if file_path.exists() and file_path.is_file():
                    content = file_path.read_text(encoding='utf-8')
                    # Limit context file size to avoid token overflow
                    if len(content) > 10000:
                        logger.warning(f"Context file {file_path} is large, truncating to 10000 chars")
                        content = content[:10000] + "\n... (truncated)"
                    context[str(file_path)] = content
                else:
                    logger.warning(f"Context file not found: {file_path}")
            except Exception as e:
                logger.warning(f"Error reading context file {file_path}: {e}")

        return context

    def apply_resolutions(
        self,
        file_path: Path,
        resolutions: List[ConflictResolution],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Apply resolutions to create resolved file content

        Args:
            file_path: Original file with conflicts
            resolutions: List of resolutions to apply
            output_path: Optional path to write resolved content

        Returns the resolved file content
        """
        content = file_path.read_text(encoding='utf-8')
        lines = content.splitlines(keepends=True)

        # Apply resolutions in reverse order to maintain line numbers
        for resolution in sorted(resolutions, key=lambda r: r.section.start_line, reverse=True):
            # Replace conflict section with resolved content
            start_idx = resolution.section.start_line - 1
            end_idx = resolution.section.end_line

            # Add resolution with explanation comment
            resolved_lines = resolution.resolved_content.splitlines(keepends=True)
            if not resolved_lines[-1].endswith('\n'):
                resolved_lines[-1] += '\n'

            # Add explanation as comment (language-aware)
            comment = self._format_explanation_comment(
                resolution.explanation,
                self.parser._detect_language(content)
            )
            if comment:
                resolved_lines.append(comment + '\n')

            # Replace lines
            lines[start_idx:end_idx] = resolved_lines

        resolved_content = ''.join(lines)

        # Write to output if specified
        if output_path:
            output_path.write_text(resolved_content, encoding='utf-8')
            logger.info(f"Resolved content written to {output_path}")

        return resolved_content

    def _format_explanation_comment(self, explanation: str, language: str) -> str:
        """
        Format explanation as a comment based on language

        Args:
            explanation: Resolution explanation
            language: Detected programming language

        Returns a formatted comment or empty string
        """
        if not explanation:
            return ""

        # Format based on language
        if language in ['python', 'shell', 'bash']:
            return f"# DIFFSENSE: {explanation}"
        elif language in ['javascript', 'java', 'c/c++', 'c', 'cpp']:
            return f"// DIFFSENSE: {explanation}"
        elif language in ['html', 'xml']:
            return f"<!-- DIFFSENSE: {explanation} -->"
        else:
            # Default to // style
            return f"// DIFFSENSE: {explanation}"
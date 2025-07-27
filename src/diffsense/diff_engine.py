"""
Core diff computation engine
"""

import difflib
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

from .exceptions import DiffError


class ChangeType(Enum):
    """Types of changes that can occur in a diff"""
    EQUAL = "equal"
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass
class LineDiff:
    """
    Represents a single line difference between two files

    Attributes:
        change_type: type of change (insert, delete, replace, equal)
        old_line: content of the line in the original file
        new_line: content of the line in the modified file
        old_line_num: line number in the original file (1-based, -1 if not applicable)
        new_line_num: line number in the modified file (1-based, -1 if not applicable)
        inline_changes: list of character spans where inline changes occurred
    """
    change_type: ChangeType
    old_line: str
    new_line: str
    old_line_num: int
    new_line_num: int
    inline_changes: List[Tuple[int, int]]

    def has_inline_changes(self) -> bool:
        """Check if this line has character-level changes highlighted"""
        return len(self.inline_changes) > 0


@dataclass
class DiffBlock:
    """
    Represents a block of related changes in a diff

    Attributes:
        old_start: starting line number in original file
        old_count: number of lines in original file for this block
        new_start: starting line number in modified file
        new_count: number of lines in modified file for this block
        lines: List of LineDiff objects representing the changes
    """
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[LineDiff]

    @property
    def header(self) -> str:
        """Generate the standard diff header for this block"""
        return f"-{self.old_start},{self.old_count} +{self.new_start},{self.new_count}"

    def get_changes_summary(self) -> dict:
        """
        Get a summary of changes in this block
        Returns a Dictionary with counts of different change types
        """
        summary = {change_type: 0 for change_type in ChangeType}
        for line in self.lines:
            summary[line.change_type] += 1
        return summary


class DiffEngine:
    """
    This engine provides diff computation with support for
    context lines, inline change detection and various output formats
    """

    def __init__(self, context_lines: int = 3):
        """
        Initialize the diff engine
        context_lines: Number of unchanged lines to show around changes
        """
        self.context_lines = max(0, context_lines)

    def compute_diff(self, original: str, modified: str) -> List[DiffBlock]:
        """
        Compute the differences between two text strings
        """
        try:
            original_lines = original.splitlines(keepends=False)
            modified_lines = modified.splitlines(keepends=False)

            return self._compute_unified_diff(original_lines, modified_lines)

        except Exception as e:
            raise DiffError(f"Failed to compute diff: {e}") from e

    def _compute_unified_diff(
        self,
        original_lines: List[str],
        modified_lines: List[str]
    ) -> List[DiffBlock]:
        """
        Compute unified diff format between two lists of lines
        """
        differ = difflib.SequenceMatcher(None, original_lines, modified_lines)
        blocks = []

        for group in self._get_grouped_opcodes(differ.get_opcodes()):
            block = self._create_diff_block(group, original_lines, modified_lines)
            if block.lines:  # Add only non-empty blocks
                blocks.append(block)

        return blocks

    def _get_grouped_opcodes(self, opcodes: List[Tuple]) -> List[List[Tuple]]:
        """
        Group opcodes with context lines for unified diff format
        Returns grouped opcodes with context
        """
        if not opcodes:
            return []

        return self._manual_group_opcodes(opcodes)

    def _manual_group_opcodes(self, opcodes: List[Tuple]) -> List[List[Tuple]]:
        """
        Manually group opcodes when difflib grouping is unavailable
        Returns manually grouped opcodes
        """
        # Filter out non-change opcodes for zero context or all-equal diffs
        non_equal_ops = [op for op in opcodes if op[0] != 'equal']

        # If no changes, return empty list
        if not non_equal_ops:
            return []

        # If no context lines, return only the changes
        if self.context_lines == 0:
            return [non_equal_ops]

        groups = []
        current_group = []

        for opcode in opcodes:
            tag, i1, i2, j1, j2 = opcode

            if tag == 'equal':
                # For equal blocks, only include context lines
                if current_group:
                    # Add trailing context
                    size = min(self.context_lines, i2 - i1)
                    if size > 0:
                        current_group.append(('equal', i1, i1 + size, j1, j1 + size))
                    groups.append(current_group)
                    current_group = []

                # Start new group with leading context if there are more changes
                remaining_ops = opcodes[opcodes.index(opcode) + 1:]
                if any(op[0] != 'equal' for op in remaining_ops):
                    size = min(self.context_lines, i2 - i1)
                    if size > 0:
                        start_i = max(i1, i2 - size)
                        start_j = max(j1, j2 - size)
                        current_group.append(('equal', start_i, i2, start_j, j2))
            else:
                current_group.append(opcode)

        if current_group:
            groups.append(current_group)

        return groups

    def _create_diff_block(
        self,
        opcodes: List[Tuple],
        original_lines: List[str],
        modified_lines: List[str]
    ) -> DiffBlock:
        """
        Create a DiffBlock from a group of opcodes
        """
        if not opcodes:
            return DiffBlock(0, 0, 0, 0, [])

        # Calculate block boundaries
        first_op = opcodes[0]
        last_op = opcodes[-1]

        old_start = first_op[1] + 1  # Convert to 1-based
        old_end = last_op[2]
        new_start = first_op[3] + 1  # Convert to 1-based
        new_end = last_op[4]

        old_count = old_end - (old_start - 1)
        new_count = new_end - (new_start - 1)

        # Process lines
        lines = []
        for tag, i1, i2, j1, j2 in opcodes:
            change_type = ChangeType(tag)

            if tag == 'equal':
                for i in range(i2 - i1):
                    old_idx = i1 + i
                    new_idx = j1 + i
                    lines.append(LineDiff(
                        change_type=change_type,
                        old_line=original_lines[old_idx],
                        new_line=modified_lines[new_idx],
                        old_line_num=old_idx + 1,
                        new_line_num=new_idx + 1,
                        inline_changes=[]
                    ))

            elif tag == 'delete':
                for i in range(i2 - i1):
                    old_idx = i1 + i
                    lines.append(LineDiff(
                        change_type=change_type,
                        old_line=original_lines[old_idx],
                        new_line="",
                        old_line_num=old_idx + 1,
                        new_line_num=-1,
                        inline_changes=[]
                    ))

            elif tag == 'insert':
                for i in range(j2 - j1):
                    new_idx = j1 + i
                    lines.append(LineDiff(
                        change_type=change_type,
                        old_line="",
                        new_line=modified_lines[new_idx],
                        old_line_num=-1,
                        new_line_num=new_idx + 1,
                        inline_changes=[]
                    ))

            elif tag == 'replace':
                # Handle replace as paired delete/insert with inline detection
                old_block = original_lines[i1:i2]
                new_block = modified_lines[j1:j2]

                # For simple 1:1 replacements, compute inline changes
                if len(old_block) == 1 and len(new_block) == 1:
                    inline_changes = self._compute_inline_changes(
                        old_block[0], new_block[0]
                    )
                    lines.append(LineDiff(
                        change_type=change_type,
                        old_line=old_block[0],
                        new_line=new_block[0],
                        old_line_num=i1 + 1,
                        new_line_num=j1 + 1,
                        inline_changes=inline_changes
                    ))
                else:
                    # For complex replacements, show as separate delete/insert
                    for i, old_line in enumerate(old_block):
                        lines.append(LineDiff(
                            change_type=ChangeType.DELETE,
                            old_line=old_line,
                            new_line="",
                            old_line_num=i1 + i + 1,
                            new_line_num=-1,
                            inline_changes=[]
                        ))

                    for i, new_line in enumerate(new_block):
                        lines.append(LineDiff(
                            change_type=ChangeType.INSERT,
                            old_line="",
                            new_line=new_line,
                            old_line_num=-1,
                            new_line_num=j1 + i + 1,
                            inline_changes=[]
                        ))

        return DiffBlock(old_start, old_count, new_start, new_count, lines)

    def _compute_inline_changes(self, old_line: str, new_line: str) -> List[Tuple[int, int]]:
        """
        Compute character-level changes within a line
        Returns a List of (start, end) tuples indicating changed character ranges in the new line
        """
        if old_line == new_line:
            return []

        matcher = difflib.SequenceMatcher(None, old_line, new_line)
        changes = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                changes.append((j1, j2))

        return changes
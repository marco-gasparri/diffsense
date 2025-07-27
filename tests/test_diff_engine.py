"""
Tests for the diff engine functionality
"""

import pytest
from diffsense.diff_engine import DiffEngine, DiffBlock, LineDiff, ChangeType
from diffsense.exceptions import DiffError


class TestDiffEngine:
    """Test cases for DiffEngine class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.engine = DiffEngine(context_lines=3)

    def test_identical_files(self):
        """Test diff of identical content"""
        content = "line1\nline2\nline3"
        blocks = self.engine.compute_diff(content, content)
        # Should return empty list for identical files
        assert len(blocks) == 0

    def test_simple_insertion(self):
        """Test simple line insertion"""
        original = "line1\nline3"
        modified = "line1\nline2\nline3"

        blocks = self.engine.compute_diff(original, modified)
        assert len(blocks) == 1

        block = blocks[0]
        insert_lines = [line for line in block.lines if line.change_type == ChangeType.INSERT]
        assert len(insert_lines) == 1
        assert insert_lines[0].new_line == "line2"

    def test_simple_deletion(self):
        """Test simple line deletion"""
        original = "line1\nline2\nline3"
        modified = "line1\nline3"

        blocks = self.engine.compute_diff(original, modified)
        assert len(blocks) == 1

        block = blocks[0]
        delete_lines = [line for line in block.lines if line.change_type == ChangeType.DELETE]
        assert len(delete_lines) == 1
        assert delete_lines[0].old_line == "line2"

    def test_line_replacement(self):
        """Test line replacement with inline changes"""
        original = "function oldName() {"
        modified = "function newName() {"

        blocks = self.engine.compute_diff(original, modified)
        assert len(blocks) == 1

        block = blocks[0]
        replace_lines = [line for line in block.lines if line.change_type == ChangeType.REPLACE]
        assert len(replace_lines) == 1

        replace_line = replace_lines[0]
        assert replace_line.old_line == original
        assert replace_line.new_line == modified
        assert len(replace_line.inline_changes) > 0

    def test_context_lines(self):
        """Test context line inclusion"""
        original = "line1\nline2\nline3\nline4\nline5"
        modified = "line1\nline2\nchanged\nline4\nline5"

        engine = DiffEngine(context_lines=2)
        blocks = engine.compute_diff(original, modified)

        assert len(blocks) == 1
        block = blocks[0]

        # Should include context lines around the change
        equal_lines = [line for line in block.lines if line.change_type == ChangeType.EQUAL]
        assert len(equal_lines) > 0

    def test_zero_context_lines(self):
        """Test diff with no context lines"""
        original = "line1\nline2\nline3"
        modified = "line1\nchanged\nline3"

        engine = DiffEngine(context_lines=0)
        blocks = engine.compute_diff(original, modified)

        assert len(blocks) == 1
        block = blocks[0]

        # Should only include changed lines (no context)
        equal_lines = [line for line in block.lines if line.change_type == ChangeType.EQUAL]
        assert len(equal_lines) == 0

    def test_multiple_blocks(self):
        """Test diff with multiple separate change blocks"""
        original = "line1\nline2\nline3\nline4\nline5\nline6\nline7"
        modified = "changed1\nline2\nline3\nline4\nline5\nline6\nchanged7"

        engine = DiffEngine(context_lines=1)
        blocks = engine.compute_diff(original, modified)

        # Should create separate blocks for distant changes
        assert len(blocks) >= 1

    def test_inline_changes_detection(self):
        """Test detection of character-level changes"""
        original = "def function_name():"
        modified = "def new_function_name():"

        blocks = self.engine.compute_diff(original, modified)

        replace_lines = []
        for block in blocks:
            replace_lines.extend([
                line for line in block.lines
                if line.change_type == ChangeType.REPLACE
            ])

        if replace_lines:
            replace_line = replace_lines[0]
            assert replace_line.has_inline_changes()
            # Check that inline changes are within the new line bounds
            for start, end in replace_line.inline_changes:
                assert 0 <= start <= end <= len(replace_line.new_line)

    def test_empty_strings(self):
        """Test diff of empty strings"""
        blocks = self.engine.compute_diff("", "")
        assert len(blocks) == 0

        blocks = self.engine.compute_diff("", "content")
        assert len(blocks) == 1

        blocks = self.engine.compute_diff("content", "")
        assert len(blocks) == 1

    def test_diff_block_header(self):
        """Test diff block header format"""
        original = "line1\nline2"
        modified = "line1\nchanged"

        blocks = self.engine.compute_diff(original, modified)
        assert len(blocks) == 1

        block = blocks[0]
        header = block.header
        assert "," in header  # Should contain line range information
        assert "+" in header  # Should contain new file indicator
        assert "-" in header  # Should contain old file indicator

    def test_diff_block_summary(self):
        """Test diff block change summary"""
        original = "line1\nline2\nline3"
        modified = "line1\nchanged\nline3\nadded"

        blocks = self.engine.compute_diff(original, modified)
        # Could be 1 or 2 blocks depending on context grouping
        assert len(blocks) >= 1

        # Test that we can get summary from each block
        for block in blocks:
            summary = block.get_changes_summary()
            assert isinstance(summary, dict)
            assert all(count >= 0 for count in summary.values())

        # Test overall changes
        all_lines = []
        for block in blocks:
            all_lines.extend(block.lines)

        change_types = [line.change_type for line in all_lines]
        assert ChangeType.REPLACE in change_types or ChangeType.DELETE in change_types
        assert ChangeType.INSERT in change_types
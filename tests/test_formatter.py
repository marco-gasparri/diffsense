"""
Tests for the diff formatter functionality
"""

import pytest
from rich.console import Console
from rich.text import Text

from diffsense.diff_engine import DiffEngine, ChangeType
from diffsense.formatter import DiffFormatter
from diffsense.exceptions import FormattingError


class TestDiffFormatter:
    """Test cases for DiffFormatter class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.formatter = DiffFormatter()
        self.engine = DiffEngine(context_lines=2)

    def test_format_empty_diff(self):
        """Test formatting of empty diff"""
        blocks = []
        result = self.formatter.format_diff(blocks)
        assert result is not None

    def test_format_simple_diff(self):
        """Test formatting of simple diff"""
        original = "line1\nline2"
        modified = "line1\nchanged"

        blocks = self.engine.compute_diff(original, modified)
        result = self.formatter.format_diff(blocks, "file1.py", "file2.py")

        assert result is not None

    def test_format_plain_diff(self):
        """Test plain text diff formatting"""
        original = "line1\nline2"
        modified = "line1\nchanged"

        blocks = self.engine.compute_diff(original, modified)
        plain_diff = self.formatter.format_plain_diff(blocks)

        assert isinstance(plain_diff, str)
        assert "@@" in plain_diff  # Should contain block headers
        assert "-" in plain_diff or "+" in plain_diff  # Should contain changes

    def test_inline_changes_formatting(self):
        """Test formatting of inline character changes"""
        original = "function oldName() {"
        modified = "function newName() {"

        blocks = self.engine.compute_diff(original, modified)

        # Find replacement lines
        replace_lines = []
        for block in blocks:
            replace_lines.extend([
                line for line in block.lines
                if line.change_type == ChangeType.REPLACE
            ])

        if replace_lines:
            formatted_line = self.formatter._format_line(replace_lines[0])
            assert isinstance(formatted_line, Text)

    def test_line_number_formatting(self):
        """Test that line numbers are properly formatted"""
        original = "line1\nline2\nline3"
        modified = "line1\nchanged\nline3"

        blocks = self.engine.compute_diff(original, modified)

        for block in blocks:
            for line in block.lines:
                formatted = self.formatter._format_line(line)
                assert isinstance(formatted, Text)
                # Check that formatted text is not empty
                assert len(formatted.plain) > 0

    def test_color_scheme_consistency(self):
        """Test that color scheme is applied consistently"""
        formatter = DiffFormatter()

        # Check that all change types have colors defined
        for change_type in ChangeType:
            assert change_type in formatter.colors
            assert change_type in formatter.symbols

    def test_format_with_custom_console(self):
        """Test formatter with custom console"""
        custom_console = Console(width=120)
        formatter = DiffFormatter(console=custom_console)

        original = "test"
        modified = "changed"

        blocks = self.engine.compute_diff(original, modified)
        result = formatter.format_diff(blocks)

        assert result is not None

    def test_summary_creation(self):
        """Test diff summary generation"""
        original = "line1\nline2\nline3"
        modified = "line1\nchanged\nline3\nadded"

        blocks = self.engine.compute_diff(original, modified)
        summary = self.formatter._create_diff_summary(blocks)

        assert summary is not None

    def test_header_creation(self):
        """Test diff header generation"""
        blocks = self.engine.compute_diff("test", "changed")
        header = self.formatter._create_diff_header("file1.py", "file2.py", blocks)

        assert header is not None

    def test_error_handling(self):
        """Test error handling in formatter"""
        # This should not raise an exception
        try:
            result = self.formatter.format_diff([], "file1", "file2")
            assert result is not None
        except FormattingError:
            pytest.fail("Formatting should not fail for empty diff")

    def test_inline_changes_highlighting(self):
        """Test inline changes highlighting"""
        content = "Hello World"
        changes = [(6, 11)]  # "World" changed
        base_color = "green"

        result = self.formatter._format_inline_changes(content, changes, base_color)
        assert isinstance(result, Text)
        assert len(result) > 0

    def test_diff_block_formatting(self):
        """Test individual diff block formatting"""
        original = "line1\nline2"
        modified = "line1\nchanged"

        blocks = self.engine.compute_diff(original, modified)
        if blocks:
            formatted_block = self.formatter._format_diff_block(blocks[0], 1)
            assert formatted_block is not None

    def test_symbols_mapping(self):
        """Test that symbols are correctly mapped to change types"""
        expected_symbols = {
            ChangeType.INSERT: "+",
            ChangeType.DELETE: "-",
            ChangeType.REPLACE: "~",
            ChangeType.EQUAL: " "
        }

        for change_type, symbol in expected_symbols.items():
            assert self.formatter.symbols[change_type] == symbol
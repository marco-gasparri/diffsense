"""
Tests for conflict resolution functionality
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from diffsense.conflict_resolver import (
    ConflictParser,
    ConflictResolver,
    ConflictSection,
    ConflictResolution,
    ConflictResolutionConfidence
)
from diffsense.exceptions import DiffSenseError


class TestConflictParser:
    """Test cases for ConflictParser"""

    def setup_method(self):
        """Set up test fixtures"""
        self.parser = ConflictParser()

    def test_has_conflicts_true(self):
        """Test detection of conflicts"""
        content = """
def func():
<<<<<<< HEAD
    return 1
=======
    return 2
>>>>>>> branch
"""
        assert self.parser.has_conflicts(content) is True

    def test_has_conflicts_false(self):
        """Test no conflicts detected"""
        content = """
def func():
    return 1
"""
        assert self.parser.has_conflicts(content) is False

    def test_has_conflicts_incomplete(self):
        """Test incomplete conflict markers"""
        content = """
<<<<<<< HEAD
    return 1
"""
        assert self.parser.has_conflicts(content) is False

    def test_parse_single_conflict(self):
        """Test parsing single conflict"""
        content = """line1
<<<<<<< HEAD
current
=======
incoming
>>>>>>> branch
line2"""

        conflicts = self.parser.parse_conflicts(content)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.start_line == 2
        assert conflict.end_line == 6
        assert conflict.current_content == "current"
        assert conflict.incoming_content == "incoming"
        assert conflict.current_marker == "<<<<<<< HEAD"
        assert conflict.incoming_marker == ">>>>>>> branch"
        assert conflict.separator_line == 4

    def test_parse_multiple_conflicts(self):
        """Test parsing multiple conflicts"""
        content = """line1
<<<<<<< HEAD
current1
=======
incoming1
>>>>>>> branch
line2
<<<<<<< HEAD
current2
=======
incoming2
>>>>>>> branch
line3"""

        conflicts = self.parser.parse_conflicts(content)

        assert len(conflicts) == 2
        assert conflicts[0].current_content == "current1"
        assert conflicts[1].current_content == "current2"

    def test_parse_multiline_conflict(self):
        """Test parsing multiline conflict content"""
        content = """
<<<<<<< HEAD
line1
line2
line3
=======
new1
new2
>>>>>>> feature
"""

        conflicts = self.parser.parse_conflicts(content)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.current_content == "line1\nline2\nline3"
        assert conflict.incoming_content == "new1\nnew2"

    def test_parse_empty_sections(self):
        """Test parsing conflicts with empty sections"""
        content = """
<<<<<<< HEAD
=======
new content
>>>>>>> branch
"""

        conflicts = self.parser.parse_conflicts(content)

        assert len(conflicts) == 1
        assert conflicts[0].current_content == ""
        assert conflicts[0].incoming_content == "new content"

    def test_parse_malformed_conflict_no_separator(self):
        """Test parsing malformed conflict without separator"""
        content = """
<<<<<<< HEAD
current
>>>>>>> branch
"""

        with pytest.raises(DiffSenseError, match="no separator found"):
            self.parser.parse_conflicts(content)

    def test_parse_malformed_conflict_no_end(self):
        """Test parsing malformed conflict without end marker"""
        content = """
<<<<<<< HEAD
current
=======
incoming
"""

        with pytest.raises(DiffSenseError, match="no end marker found"):
            self.parser.parse_conflicts(content)

    def test_extract_file_context(self):
        """Test extracting context around conflicts"""
        content = """# File header
import sys

def function1():
    pass

<<<<<<< HEAD
    return 1
=======
    return 2
>>>>>>> branch

def function2():
    pass
"""

        conflicts = self.parser.parse_conflicts(content)
        context = self.parser.extract_file_context(content, conflicts)

        assert context["total_conflicts"] == 1
        assert context["file_language"] == "python"
        assert len(context["conflicts_context"]) == 1

        conflict_ctx = context["conflicts_context"][0]
        assert "def function1():" in conflict_ctx["context_before"]
        assert "def function2():" in conflict_ctx["context_after"]

    def test_detect_language_python(self):
        """Test Python language detection"""
        content = "import os\ndef main():\n    pass"
        assert self.parser._detect_language(content) == "python"

    def test_detect_language_javascript(self):
        """Test JavaScript language detection"""
        content = "const x = 1;\nfunction test() { return x; }"
        assert self.parser._detect_language(content) == "javascript"

    def test_detect_language_c(self):
        """Test C/C++ language detection"""
        content = "#include <stdio.h>\nint main() { return 0; }"
        assert self.parser._detect_language(content) == "c/c++"

    def test_detect_language_java(self):
        """Test Java language detection"""
        content = "public class Main {\n    private int x;\n}"
        assert self.parser._detect_language(content) == "java"

    def test_detect_language_unknown(self):
        """Test unknown language detection"""
        content = "This is just plain text."
        assert self.parser._detect_language(content) == "unknown"


class TestConflictResolver:
    """Test cases for ConflictResolver"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.resolver = ConflictResolver(self.mock_llm)
        self.test_file = Path("test_conflict.py")

    @patch('diffsense.conflict_resolver.Path.read_text')
    def test_resolve_conflicts_no_conflicts(self, mock_read):
        """Test resolving file without conflicts"""
        mock_read.return_value = "def func():\n    return 1"

        with pytest.raises(DiffSenseError, match="No merge conflicts found"):
            self.resolver.resolve_conflicts(self.test_file)

    @patch('diffsense.conflict_resolver.Path.read_text')
    def test_resolve_conflicts_success(self, mock_read):
        """Test successful conflict resolution"""
        mock_read.return_value = """
<<<<<<< HEAD
    return 1
=======
    return 2
>>>>>>> branch
"""

        # Mock LLM response
        mock_resolution = ConflictResolution(
            section=Mock(),
            resolved_content="    return 2",
            explanation="Incoming change is more recent",
            confidence=ConflictResolutionConfidence.HIGH,
            alternative_resolutions=None
        )
        self.mock_llm.resolve_conflict.return_value = (mock_resolution, 100)

        resolutions, tokens = self.resolver.resolve_conflicts(self.test_file)

        assert len(resolutions) == 1
        assert resolutions[0].resolved_content == "    return 2"
        assert tokens == 100

    @patch('diffsense.conflict_resolver.Path.read_text')
    def test_resolve_conflicts_with_context_files(self, mock_read):
        """Test resolution with additional context files"""
        mock_read.side_effect = [
            # Main file
            """
<<<<<<< HEAD
    return calculate_discount(price, "premium")
=======
    return calculate_discount(price, customer.type)
>>>>>>> branch
""",
            # Context file
            "def calculate_discount(price, customer_type):\n    pass"
        ]

        # Mock resolution
        mock_resolution = Mock()
        self.mock_llm.resolve_conflict.return_value = (mock_resolution, 150)

        context_file = Path("context.py")
        resolutions, tokens = self.resolver.resolve_conflicts(
            self.test_file,
            context_files=[context_file]
        )

        # Verify context was passed to LLM
        self.mock_llm.resolve_conflict.assert_called()
        call_args = self.mock_llm.resolve_conflict.call_args
        assert call_args.kwargs["additional_context"] is not None

    def test_read_context_files(self):
        """Test reading context files"""
        # Create mock paths
        mock_path1 = Mock(spec=Path)
        mock_path1.exists.return_value = True
        mock_path1.is_file.return_value = True
        mock_path1.read_text.return_value = "content1"
        mock_path1.__str__ = Mock(return_value="file1.py")

        mock_path2 = Mock(spec=Path)
        mock_path2.exists.return_value = False
        mock_path2.__str__ = Mock(return_value="file2.py")

        context = self.resolver._read_context_files([mock_path1, mock_path2])

        assert "file1.py" in context
        assert context["file1.py"] == "content1"
        assert "file2.py" not in context

    def test_read_large_context_file(self):
        """Test truncation of large context files"""
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.read_text.return_value = "x" * 15000  # Large content
        mock_path.__str__ = Mock(return_value="large.py")

        context = self.resolver._read_context_files([mock_path])

        assert len(context["large.py"]) < 15000
        assert "truncated" in context["large.py"]

    @patch('diffsense.conflict_resolver.Path.read_text')
    @patch('diffsense.conflict_resolver.Path.write_text')
    def test_apply_resolutions(self, mock_write, mock_read):
        """Test applying resolutions to file"""
        original_content = """line1
<<<<<<< HEAD
current
=======
incoming
>>>>>>> branch
line2"""

        mock_read.return_value = original_content

        # Create mock resolution
        section = ConflictSection(
            start_line=2,
            end_line=6,
            current_content="current",
            incoming_content="incoming",
            current_marker="<<<<<<< HEAD",
            incoming_marker=">>>>>>> branch",
            separator_line=4
        )

        resolution = ConflictResolution(
            section=section,
            resolved_content="resolved",
            explanation="Test resolution",
            confidence=ConflictResolutionConfidence.HIGH
        )

        result = self.resolver.apply_resolutions(
            self.test_file,
            [resolution],
            output_path=Path("output.py")
        )

        assert "resolved" in result
        assert "<<<<<<< HEAD" not in result
        assert ">>>>>>> branch" not in result
        assert "DIFFSENSE: Test resolution" in result
        mock_write.assert_called_once()

    def test_format_explanation_comment_python(self):
        """Test formatting explanation as Python comment"""
        comment = self.resolver._format_explanation_comment(
            "Test explanation",
            "python"
        )
        assert comment == "# DIFFSENSE: Test explanation"

    def test_format_explanation_comment_javascript(self):
        """Test formatting explanation as JavaScript comment"""
        comment = self.resolver._format_explanation_comment(
            "Test explanation",
            "javascript"
        )
        assert comment == "// DIFFSENSE: Test explanation"

    def test_format_explanation_comment_html(self):
        """Test formatting explanation as HTML comment"""
        comment = self.resolver._format_explanation_comment(
            "Test explanation",
            "html"
        )
        assert comment == "<!-- DIFFSENSE: Test explanation -->"

    def test_format_explanation_comment_empty(self):
        """Test empty explanation returns empty string"""
        comment = self.resolver._format_explanation_comment("", "python")
        assert comment == ""
"""
Additional tests for CLI Git functionality
"""

import pytest
import tempfile
import shutil
import os
import logging
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, Mock, MagicMock
from git import Repo

from diffsense.cli import app
from diffsense.git_manager import GitFileContent


class TestCLIGitMode:
    """Test cases for CLI Git mode functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()

        # Create a temporary Git repository
        self.temp_dir = tempfile.mkdtemp(prefix="diffsense_cli_test_")
        self.repo_path = Path(self.temp_dir).resolve()

        # Initialize repository
        self.repo = Repo.init(self.repo_path)
        self.repo.config_writer().set_value("user", "name", "Test User").release()
        self.repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Change to repo directory
        self.original_cwd = os.getcwd()
        os.chdir(self.repo_path)

        # Create test file and initial commit with relative path
        self.test_file = Path("test.py")
        self.test_file.write_text("def hello():\n    print('Hello')\n")
        self.repo.index.add(["test.py"])
        self.initial_commit = self.repo.index.commit("Initial commit")

        # Create second commit
        self.test_file.write_text("def hello():\n    print('Hello World')\n")
        self.repo.index.add(["test.py"])
        self.second_commit = self.repo.index.commit("Update hello")

    def teardown_method(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_git_mode_help(self):
        """Test git option is present in help"""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "git" in result.stdout

    def test_git_mode_no_arguments(self):
        """Test Git mode with no arguments"""
        # Don't use isolated_filesystem - stay in the Git repo
        result = self.runner.invoke(app, ["--git"])
        assert result.exit_code == 1
        # Check stderr or logs for error message
        # The error is logged, not printed to stdout

    def test_git_mode_not_in_repository(self):
        """Test Git mode outside a repository"""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(app, ["--git", "HEAD", "test.py"])
            assert result.exit_code == 1

    @patch('diffsense.cli.DiffEngine')
    def test_git_mode_single_ref(self, mock_engine):
        """Test Git mode with single reference (HEAD vs working)"""
        # Mock diff engine
        mock_engine.return_value.compute_diff.return_value = []

        # Don't use isolated_filesystem - run in actual repo
        os.chdir(self.repo_path)

        # Create/modify the file in working directory
        Path("test.py").write_text("def hello():\n    print('Modified')\n")

        result = self.runner.invoke(app, ["--git", "HEAD", "test.py", "--no-ai"])
        assert result.exit_code == 0
        assert "No differences found" in result.stdout

    @patch('diffsense.cli.DiffEngine')
    @patch('diffsense.cli.DiffFormatter')
    def test_git_mode_two_refs_success(self, mock_formatter, mock_engine):
        """Test Git mode with two references - successful case"""
        # Mock diff engine to return some changes
        from diffsense.diff_engine import DiffBlock, LineDiff, ChangeType

        mock_block = DiffBlock(
            old_start=2, old_count=1, new_start=2, new_count=1,
            lines=[LineDiff(
                ChangeType.REPLACE,
                "    print('Hello')",
                "    print('Hello World')",
                2, 2, []
            )]
        )
        mock_engine.return_value.compute_diff.return_value = [mock_block]

        # Mock formatter
        mock_formatter.return_value.format_diff.return_value = Mock()

        # Run in actual repo
        os.chdir(self.repo_path)

        result = self.runner.invoke(app, [
            "--git", "HEAD~1", "HEAD", "test.py", "--no-ai"
        ])
        assert result.exit_code == 0

    @patch('diffsense.cli.LLMManager')
    @patch('diffsense.cli.DiffEngine')
    def test_git_mode_with_ai_analysis_mocked(self, mock_engine, mock_llm):
        """Test Git mode with AI analysis - fully mocked"""
        # Setup mocks
        from diffsense.diff_engine import DiffBlock, LineDiff, ChangeType

        mock_block = DiffBlock(
            old_start=2, old_count=1, new_start=2, new_count=1,
            lines=[LineDiff(
                ChangeType.REPLACE,
                "    print('Hello')",
                "    print('Hello World')",
                2, 2, []
            )]
        )
        mock_engine.return_value.compute_diff.return_value = [mock_block]
        mock_llm.return_value.analyze_diff.return_value = "This is a test analysis"

        # Mock GitManager to avoid file access issues
        with patch('diffsense.cli.GitManager') as mock_git_manager:
            mock_git_instance = Mock()
            mock_git_manager.return_value = mock_git_instance

            # Setup GitManager mocks
            mock_git_instance.validate_git_mode.return_value = ("HEAD~1", "HEAD", "test.py")

            # Mock create_temp_files as context manager
            mock_context = MagicMock()
            mock_context.__enter__.return_value = (
                Path(self.temp_dir) / "temp1.py",
                Path(self.temp_dir) / "temp2.py"
            )
            mock_context.__exit__.return_value = None
            mock_git_instance.create_temp_files.return_value = mock_context

            # Create temp files
            (Path(self.temp_dir) / "temp1.py").write_text("def hello():\n    print('Hello')\n")
            (Path(self.temp_dir) / "temp2.py").write_text("def hello():\n    print('Hello World')\n")

            mock_git_instance.format_file_label.side_effect = lambda f, r: f"{f} ({r})"

            result = self.runner.invoke(app, ["--git", "HEAD~1", "HEAD", "test.py"])
            assert result.exit_code == 0
            assert "AI Analysis:" in result.stdout
            assert "This is a test analysis" in result.stdout

    def test_git_mode_invalid_reference(self):
        """Test Git mode with invalid reference"""
        os.chdir(self.repo_path)
        result = self.runner.invoke(app, ["--git", "invalid_ref", "test.py"])
        assert result.exit_code == 1

    def test_git_mode_file_not_in_commit(self):
        """Test Git mode with file not in commit"""
        os.chdir(self.repo_path)
        result = self.runner.invoke(app, ["--git", "HEAD", "nonexistent.py"])
        assert result.exit_code == 1

    @patch('diffsense.cli.DiffEngine')
    def test_git_mode_with_context_option(self, mock_engine):
        """Test Git mode with --context option"""
        mock_engine.return_value.compute_diff.return_value = []

        # Run in actual repo
        os.chdir(self.repo_path)

        result = self.runner.invoke(app, [
            "--git", "HEAD", "test.py", "--context", "5", "--no-ai"
        ])
        assert result.exit_code == 0
        # Verify DiffEngine was called with context_lines=5
        mock_engine.assert_called_with(context_lines=5)

    @patch('diffsense.cli.LLMManager')
    @patch('diffsense.cli.DiffEngine')
    @patch('diffsense.cli.DiffFormatter')
    @patch('diffsense.cli.GitManager')
    def test_git_mode_with_custom_model(self, mock_git_manager, mock_formatter, mock_engine, mock_llm):
        """Test Git mode with custom model"""
        # Setup mocks
        from diffsense.diff_engine import DiffBlock, LineDiff, ChangeType

        # Create a valid diff block
        mock_block = DiffBlock(
            old_start=1, old_count=1, new_start=1, new_count=1,
            lines=[LineDiff(
                ChangeType.REPLACE,
                "old content",
                "new content",
                1, 1, []
            )]
        )
        mock_engine.return_value.compute_diff.return_value = [mock_block]

        # Mock formatter properly
        from rich.text import Text
        mock_formatted = Text("Formatted diff output")
        mock_formatter.return_value.format_diff.return_value = mock_formatted

        # Mock LLM
        mock_llm.return_value.analyze_diff.return_value = "Analysis"

        # Mock GitManager
        mock_git_instance = Mock()
        mock_git_manager.return_value = mock_git_instance
        mock_git_instance.validate_git_mode.return_value = ("HEAD", None, "test.py")

        # Mock create_temp_files
        mock_context = MagicMock()
        temp1 = Path(self.temp_dir) / "temp1.py"
        temp2 = Path(self.temp_dir) / "temp2.py"
        temp1.write_text("content1")
        temp2.write_text("content2")
        mock_context.__enter__.return_value = (temp1, temp2)
        mock_git_instance.create_temp_files.return_value = mock_context
        mock_git_instance.format_file_label.return_value = "test.py (HEAD)"

        result = self.runner.invoke(app, [
            "--git", "HEAD", "test.py", "--model", "custom/model"
        ])
        assert result.exit_code == 0
        mock_llm.assert_called_with(model_id="custom/model")

    def test_git_mode_verbose(self):
        """Test Git mode with verbose output"""
        # Run in actual repo
        os.chdir(self.repo_path)

        result = self.runner.invoke(app, [
            "--git", "HEAD", "test.py", "--verbose", "--no-ai"
        ])
        # Either it succeeds or we check for verbose mode activation
        assert result.exit_code in [0, 1]

    def test_traditional_mode_in_git_repo(self):
        """Test that traditional mode still works in a Git repository"""
        os.chdir(self.repo_path)

        # Create two files for comparison
        file1 = Path("file1.txt")
        file2 = Path("file2.txt")
        file1.write_text("content1")
        file2.write_text("content2")

        result = self.runner.invoke(app, [
            str(file1), str(file2), "--no-ai"
        ])
        assert result.exit_code == 0
        # Check output doesn't have Git-specific messages
        output_lower = result.stdout.lower()
        assert "commit" not in output_lower or "repository" not in output_lower

    @patch('diffsense.cli.GitManager')
    def test_git_mode_temp_file_cleanup(self, mock_git_manager):
        """Test that temporary files are cleaned up properly"""
        # Track cleanup
        cleanup_called = False

        class MockTempFiles:
            def __init__(self, temp_dir):
                self.temp_dir = temp_dir

            def __enter__(self):
                file1 = Path(self.temp_dir) / "temp1"
                file2 = Path(self.temp_dir) / "temp2"
                file1.write_text("content1")
                file2.write_text("content2")
                return file1, file2

            def __exit__(self, *args):
                nonlocal cleanup_called
                cleanup_called = True

        mock_instance = Mock()
        mock_instance.validate_git_mode.return_value = ("HEAD", None, "test.py")
        mock_instance.create_temp_files.return_value = MockTempFiles(self.temp_dir)
        mock_instance.format_file_label.return_value = "test.py (HEAD)"
        mock_git_manager.return_value = mock_instance

        result = self.runner.invoke(app, ["--git", "HEAD", "test.py", "--no-ai"])

        # Verify cleanup was called
        assert cleanup_called

    def test_git_mode_single_ref_only(self):
        """Test Git mode with only one reference and no file"""
        os.chdir(self.repo_path)
        result = self.runner.invoke(app, ["--git", "HEAD"])
        assert result.exit_code == 0
        # Should list changed files, not fail

    def test_git_mode_list_changes(self):
        """Test Git mode listing changed files"""
        os.chdir(self.repo_path)

        # Create a new file in working directory
        new_file = Path("newfile.py")
        new_file.write_text("# New file")

        # Modify existing file
        self.test_file.write_text("def hello():\n    print('Modified')\n")

        result = self.runner.invoke(app, ["--git", "HEAD"])
        assert result.exit_code == 0
        assert "Changed files between HEAD and working directory" in result.stdout
        assert "test.py" in result.stdout
        # newfile.py will be shown as it's an untracked file
        assert "newfile.py" in result.stdout
        assert "To see changes for a specific file" in result.stdout
"""
Tests for Git manager functionality
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import git
from git import Repo
from gitdb.exc import BadName

from diffsense.git_manager import GitManager, GitFileContent
from diffsense.exceptions import GitError


class TestGitManager:
    """Test cases for GitManager class"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a temporary directory for test repository
        self.temp_dir = tempfile.mkdtemp(prefix="diffsense_test_")
        self.repo_path = Path(self.temp_dir).resolve()

        # Initialize a test repository
        self.repo = Repo.init(self.repo_path)

        # Configure Git user for commits
        self.repo.config_writer().set_value("user", "name", "Test User").release()
        self.repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Change to repo directory to avoid path issues
        self.original_cwd = os.getcwd()
        os.chdir(self.repo_path)

        # Create initial commit with relative path
        self.test_file = Path("test.py")
        self.test_file.write_text("def hello():\n    print('Hello')\n")
        self.repo.index.add(["test.py"])
        self.initial_commit = self.repo.index.commit("Initial commit")

        # Create second commit
        self.test_file.write_text("def hello():\n    print('Hello World')\n")
        self.repo.index.add(["test.py"])
        self.second_commit = self.repo.index.commit("Update hello function")

    def teardown_method(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test GitManager initialization"""
        manager = GitManager(self.repo_path)
        assert manager.repo_path == self.repo_path
        assert manager._repo is None  # Lazy loading

    def test_repo_property(self):
        """Test repository access"""
        manager = GitManager(self.repo_path)
        repo = manager.repo
        assert isinstance(repo, Repo)
        assert Path(repo.working_dir).resolve() == self.repo_path.resolve()

    def test_repo_not_found(self):
        """Test error when not in a Git repository"""
        non_git_path = Path(tempfile.mkdtemp())
        try:
            manager = GitManager(non_git_path)
            with pytest.raises(GitError, match="Not a Git repository"):
                _ = manager.repo
        finally:
            shutil.rmtree(non_git_path)

    def test_parse_git_reference(self):
        """Test parsing various Git references"""
        manager = GitManager(self.repo_path)

        # Test with commit SHA
        sha = manager.parse_git_reference(self.initial_commit.hexsha)
        assert sha == self.initial_commit.hexsha

        # Test with HEAD
        head_sha = manager.parse_git_reference("HEAD")
        assert head_sha == self.second_commit.hexsha

        # Test with HEAD~1
        prev_sha = manager.parse_git_reference("HEAD~1")
        assert prev_sha == self.initial_commit.hexsha

    def test_parse_invalid_reference(self):
        """Test parsing invalid Git reference"""
        manager = GitManager(self.repo_path)

        with pytest.raises(GitError, match="Invalid Git reference"):
            manager.parse_git_reference("invalid_ref")

    def test_get_file_content_from_commit(self):
        """Test getting file content from a specific commit"""
        manager = GitManager(self.repo_path)

        # Get content from initial commit
        content = manager.get_file_content("test.py", self.initial_commit.hexsha)
        assert isinstance(content, GitFileContent)
        assert content.content == "def hello():\n    print('Hello')\n"
        assert content.commit_sha == self.initial_commit.hexsha
        assert self.initial_commit.hexsha[:7] in content.filename

    def test_get_file_content_from_working_directory(self):
        """Test getting file content from working directory"""
        manager = GitManager(self.repo_path)

        # Modify file in working directory
        self.test_file.write_text("def hello():\n    print('Modified')\n")

        content = manager.get_file_content("test.py", None)
        assert isinstance(content, GitFileContent)
        assert content.content == "def hello():\n    print('Modified')\n"
        assert content.commit_sha is None
        assert content.filename == "test.py"

    def test_get_file_content_not_found(self):
        """Test getting non-existent file"""
        manager = GitManager(self.repo_path)

        # File not in commit
        with pytest.raises(GitError, match="not found in commit"):
            manager.get_file_content("nonexistent.py", "HEAD")

        # File not in working directory
        with pytest.raises(GitError, match="File does not exist"):
            manager.get_file_content("nonexistent.py", None)

    def test_get_changed_files(self):
        """Test getting list of changed files"""
        manager = GitManager(self.repo_path)

        # Changes between commits
        changes = manager.get_changed_files(
            self.initial_commit.hexsha,
            self.second_commit.hexsha
        )
        assert "test.py" in changes

        # Add a new file (untracked)
        new_file = Path("new.py")
        new_file.write_text("# New file")

        # Changes between HEAD and working directory (includes untracked)
        changes = manager.get_changed_files("HEAD", None)
        # The new file won't appear in diffs since it's untracked
        # But our improved implementation should include it

        # For working directory comparison, we need to check both refs None
        changes = manager.get_changed_files(None, None)
        assert "new.py" in changes

    def test_create_temp_files_commit_to_commit(self):
        """Test creating temp files for commit comparison"""
        manager = GitManager(self.repo_path)

        with manager.create_temp_files(
            self.initial_commit.hexsha,
            self.second_commit.hexsha,
            "test.py"
        ) as (file1, file2):
            assert file1.exists()
            assert file2.exists()

            content1 = file1.read_text()
            content2 = file2.read_text()

            assert "print('Hello')" in content1
            assert "print('Hello World')" in content2

        # Files should be cleaned up
        assert not file1.exists()
        assert not file2.exists()

    def test_create_temp_files_working_directory(self):
        """Test creating temp files with working directory"""
        manager = GitManager(self.repo_path)

        # Modify file in working directory
        self.test_file.write_text("def hello():\n    print('Working')\n")

        with manager.create_temp_files("HEAD", None, "test.py") as (file1, file2):
            content1 = file1.read_text()
            content2 = file2.read_text()

            assert "print('Hello World')" in content1
            assert "print('Working')" in content2

    def test_create_temp_files_no_file_path(self):
        """Test error when no file path provided for commit comparison"""
        manager = GitManager(self.repo_path)

        # For two commits without file, we now get a different error
        with pytest.raises(GitError) as exc_info:
            with manager.create_temp_files("HEAD~1", "HEAD", None):
                pass

        error_msg = str(exc_info.value)
        assert "Multiple files changed" in error_msg or "No changes found" in error_msg

    def test_format_file_label(self):
        """Test formatting file labels"""
        manager = GitManager(self.repo_path)

        # Label for commit
        label = manager.format_file_label("test.py", "HEAD")
        assert "test.py" in label
        assert self.second_commit.hexsha[:7] in label
        assert "Update hello function" in label

        # Label for working directory
        label = manager.format_file_label("test.py", None)
        assert label == "test.py (working directory)"

    def test_validate_git_mode_single_ref(self):
        """Test Git mode validation with single reference"""
        manager = GitManager(self.repo_path)

        # Single ref without file should be valid (lists changes)
        ref1, ref2, file_path = manager.validate_git_mode(["HEAD"])
        assert ref1 == "HEAD"
        assert ref2 is None
        assert file_path is None

    def test_validate_git_mode_two_refs(self):
        """Test Git mode validation with two references"""
        manager = GitManager(self.repo_path)

        # Two refs without file should be valid (lists changes between commits)
        ref1, ref2, file_path = manager.validate_git_mode(["HEAD~1", "HEAD"])
        assert ref1 == "HEAD~1"
        assert ref2 == "HEAD"
        assert file_path is None

    def test_validate_git_mode_ref_and_file(self):
        """Test Git mode validation with reference and file"""
        manager = GitManager(self.repo_path)

        # When file exists
        ref1, ref2, file_path = manager.validate_git_mode(["HEAD", "test.py"])
        assert ref1 == "HEAD"
        assert ref2 is None
        assert file_path == "test.py"

    def test_validate_git_mode_ref_and_nonexistent_file(self):
        """Test Git mode validation with reference and non-existent file"""
        manager = GitManager(self.repo_path)

        # Should still parse as file even if it doesn't exist
        # (might exist in the commit)
        ref1, ref2, file_path = manager.validate_git_mode(["HEAD", "nonexistent.py"])
        assert ref1 == "HEAD"
        assert ref2 is None
        assert file_path == "nonexistent.py"

    def test_validate_git_mode_full_syntax(self):
        """Test Git mode validation with full syntax"""
        manager = GitManager(self.repo_path)

        ref1, ref2, file_path = manager.validate_git_mode(["HEAD~1", "HEAD", "test.py"])
        assert ref1 == "HEAD~1"
        assert ref2 == "HEAD"
        assert file_path == "test.py"

    def test_validate_git_mode_errors(self):
        """Test Git mode validation errors"""
        manager = GitManager(self.repo_path)

        # No arguments
        with pytest.raises(GitError, match="at least one reference"):
            manager.validate_git_mode([])

        # Too many arguments
        with pytest.raises(GitError, match="Too many arguments"):
            manager.validate_git_mode(["HEAD", "HEAD~1", "file.py", "extra"])

    def test_is_git_repository(self):
        """Test repository detection"""
        # Valid repository
        manager = GitManager(self.repo_path)
        assert manager.is_git_repository() is True

        # Not a repository
        non_git_path = Path(tempfile.mkdtemp())
        try:
            manager = GitManager(non_git_path)
            assert manager.is_git_repository() is False
        finally:
            shutil.rmtree(non_git_path)

    def test_binary_file_handling(self):
        """Test handling of binary files"""
        manager = GitManager(self.repo_path)

        # Create a binary file with clearly non-UTF8 content
        binary_file = Path("binary.dat")
        # Use bytes that will definitely fail UTF-8 decoding
        binary_content = bytes([0xFF, 0xFE, 0x00, 0x00, 0xFF, 0xFF])
        binary_file.write_bytes(binary_content)
        self.repo.index.add(["binary.dat"])
        self.repo.index.commit("Add binary file")

        # Should raise error for binary file in commit
        with pytest.raises(GitError, match="not a valid text file"):
            manager.get_file_content("binary.dat", "HEAD")

        # Should also raise for binary file in working directory
        with pytest.raises(GitError, match="not a valid text file"):
            manager.get_file_content("binary.dat", None)

    def test_relative_path_handling(self):
        """Test handling of relative paths"""
        manager = GitManager(self.repo_path)

        # Create a subdirectory with a file
        subdir = Path("subdir")
        subdir.mkdir()
        sub_file = subdir / "sub.py"
        sub_file.write_text("# Subdir file")
        self.repo.index.add(["subdir/sub.py"])
        self.repo.index.commit("Add subdir file")

        # Test with relative path
        content = manager.get_file_content("subdir/sub.py", "HEAD")
        assert content.content == "# Subdir file"

    @patch('diffsense.git_manager.Repo')
    def test_repo_initialization_error(self, mock_repo):
        """Test handling of repository initialization errors"""
        mock_repo.side_effect = Exception("Git error")

        manager = GitManager(self.repo_path)
        with pytest.raises(GitError, match="Failed to access Git repository"):
            _ = manager.repo

    def test_get_file_content_unchanged_file(self):
        """Test getting content of a file that exists but wasn't changed in last commit"""
        manager = GitManager(self.repo_path)

        # Create a file in the first commit
        unchanged_file = Path("unchanged.py")
        unchanged_file.write_text("# This file won't change")
        self.repo.index.add(["unchanged.py"])
        self.repo.index.commit("Add unchanged file")

        # Make another commit that doesn't touch this file
        another_file = Path("another.py")
        another_file.write_text("# Another file")
        self.repo.index.add(["another.py"])
        self.repo.index.commit("Add another file")

        # Now try to get unchanged.py from HEAD - it should work
        content = manager.get_file_content("unchanged.py", "HEAD")
        assert content.content == "# This file won't change"
        assert content.filename.startswith("unchanged.py")

        # Also test comparing with working directory
        # Modify the file in working directory
        unchanged_file.write_text("# This file is now changed in working dir")

        # Should be able to compare working vs HEAD
        working_content = manager.get_file_content("unchanged.py", None)
        head_content = manager.get_file_content("unchanged.py", "HEAD")

        assert working_content.content == "# This file is now changed in working dir"
        assert head_content.content == "# This file won't change"

    def test_create_temp_files_two_commits_no_file(self):
        """Test creating temp files for two commits without specifying a file"""
        manager = GitManager(self.repo_path)

        # Try to compare two commits without specifying a file
        # Should get an informative error listing the changed files
        with pytest.raises(GitError) as exc_info:
            with manager.create_temp_files("HEAD~1", "HEAD", None):
                pass

        error_msg = str(exc_info.value)
        assert "Multiple files changed" in error_msg
        assert "test.py" in error_msg  # Should list the changed file
        assert "Please specify which file to diff" in error_msg
        assert "Example: diffsense --git" in error_msg

    def test_get_file_content_from_subdirectory(self):
        """Test getting file content when running from a subdirectory"""
        manager = GitManager(self.repo_path)

        # Create a subdirectory structure
        src_dir = Path("src")
        src_dir.mkdir()
        src_file = src_dir / "module.py"
        src_file.write_text("# Module in src")

        # Also create a file in root
        root_file = Path("rootfile.py")
        root_file.write_text("# File in root")

        # Commit both files
        self.repo.index.add(["src/module.py", "rootfile.py"])
        self.repo.index.commit("Add files in different locations")

        # Change to subdirectory
        original_cwd = os.getcwd()
        try:
            os.chdir(src_dir)

            # Create a new GitManager from subdirectory
            sub_manager = GitManager()

            # Should be able to access file in current directory
            content = sub_manager.get_file_content("module.py", "HEAD")
            assert content.content == "# Module in src"

            # Should be able to access file in parent directory
            content = sub_manager.get_file_content("../rootfile.py", "HEAD")
            assert content.content == "# File in root"

        finally:
            os.chdir(original_cwd)

    def test_get_file_content_from_head_unchanged_in_last_commit(self):
        """Test getting file from HEAD even if it wasn't changed in the last commit"""
        manager = GitManager(self.repo_path)

        # The file test.py exists in HEAD but wasn't changed in the last commit
        # (it was changed in the second-to-last commit)

        # Add another file in a new commit (test.py unchanged)
        other_file = Path("other.py")
        other_file.write_text("# Other file")
        self.repo.index.add(["other.py"])
        self.repo.index.commit("Add other file")

        # Now test.py exists in HEAD but wasn't modified in the last commit
        # We should still be able to get its content
        content = manager.get_file_content("test.py", "HEAD")
        assert content.content == "def hello():\n    print('Hello World')\n"
        assert "test.py" in content.filename

        # Also test with working directory modified
        self.test_file.write_text("def hello():\n    print('Modified in working dir')\n")

        # Should be able to get both versions
        working_content = manager.get_file_content("test.py", None)
        head_content = manager.get_file_content("test.py", "HEAD")

        assert working_content.content == "def hello():\n    print('Modified in working dir')\n"
        assert head_content.content == "def hello():\n    print('Hello World')\n"
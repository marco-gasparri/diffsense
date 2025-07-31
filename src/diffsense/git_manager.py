"""
Git repository management for diff operations
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, List, NamedTuple
from contextlib import contextmanager

import git
from git import Repo, GitCommandError
from gitdb.exc import BadName

from .exceptions import GitError

logger = logging.getLogger(__name__)


class GitFileContent(NamedTuple):
    """
    Container for file content from Git

    Attributes:
        content: The file content as string
        filename: The filename (possibly with commit info)
        commit_sha: The commit SHA if applicable
    """
    content: str
    filename: str
    commit_sha: Optional[str] = None


class GitManager:
    """
    Manages Git repository operations for diff analysis
    """

    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize Git manager

        Args:
            repo_path: Path to Git repository (uses current directory if None)
        """
        self.repo_path = repo_path or Path.cwd()
        self._repo: Optional[Repo] = None

    @property
    def repo(self) -> Repo:
        """
        Get the Git repository instance
        Returns GitPython Repo instance
        """
        if self._repo is None:
            try:
                self._repo = Repo(self.repo_path, search_parent_directories=True)
            except git.InvalidGitRepositoryError:
                raise GitError(f"Not a Git repository: {self.repo_path}")
            except Exception as e:
                raise GitError(f"Failed to access Git repository: {e}")

        return self._repo

    def parse_git_reference(self, ref: str) -> str:
        """
        Parse and validate a Git reference
        Returns resolved commit SHA
        """
        try:
            commit = self.repo.commit(ref)
            return str(commit.hexsha)
        except (GitCommandError, ValueError, BadName) as e:
            raise GitError(f"Invalid Git reference '{ref}': {e}")

    def get_file_content(
        self,
        file_path: str,
        ref: Optional[str] = None
    ) -> GitFileContent:
        """
        Get file content from Git
        Returns GitFileContent with file content and metadata
        """
        try:
            # Normalize file path relative to repo root
            repo_root = Path(self.repo.working_dir)

            # Handle both absolute and relative paths
            if Path(file_path).is_absolute():
                try:
                    rel_path = Path(file_path).relative_to(repo_root)
                except ValueError:
                    raise GitError(f"File path {file_path} is outside repository")
            else:
                # If the path is relative, resolve it relative to current directory
                # then make it relative to repo root
                current_dir = Path.cwd()
                try:
                    abs_path = (current_dir / file_path).resolve()
                    rel_path = abs_path.relative_to(repo_root)
                except ValueError:
                    # If that doesn't work, use the path as-is
                    rel_path = Path(file_path)

            if ref is None:
                # Get content from working directory
                full_path = repo_root / rel_path
                if not full_path.exists():
                    raise GitError(f"File does not exist in working directory: {file_path}")
                if not full_path.is_file():
                    raise GitError(f"Path is not a file: {file_path}")

                try:
                    content = full_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    raise GitError(f"File is not a valid text file: {file_path}")

                return GitFileContent(
                    content=content,
                    filename=str(rel_path),
                    commit_sha=None
                )

            else:
                # Get content from specific commit
                commit = self.repo.commit(ref)

                # Try to get the file from the commit's tree
                # Use forward slashes for Git paths
                git_path = str(rel_path).replace('\\', '/')

                try:
                    # Navigate through the tree to find the file
                    # This handles both files in root and in subdirectories
                    tree = commit.tree

                    # Split the path and navigate through directories
                    path_parts = git_path.split('/')

                    # Navigate to the parent directory if needed
                    for part in path_parts[:-1]:
                        try:
                            tree = tree[part]
                        except KeyError:
                            raise GitError(f"Directory '{part}' not found in commit {ref}")

                    # Get the final file
                    try:
                        item = tree[path_parts[-1]]
                    except KeyError:
                        raise GitError(f"File '{file_path}' not found in commit {ref}")

                    if item.type != 'blob':
                        raise GitError(f"Path '{file_path}' is not a file in commit {ref}")

                    try:
                        content = item.data_stream.read().decode('utf-8')
                    except UnicodeDecodeError:
                        raise GitError(f"File is not a valid text file: {file_path}")

                    # Format filename with commit info
                    short_sha = commit.hexsha[:7]
                    filename = f"{rel_path} ({short_sha})"

                    return GitFileContent(
                        content=content,
                        filename=filename,
                        commit_sha=commit.hexsha
                    )
                except GitError:
                    raise
                except Exception as e:
                    raise GitError(f"Error accessing file '{file_path}' in commit {ref}: {e}")

        except GitError:
            raise
        except Exception as e:
            raise GitError(f"Failed to get file content: {e}")

    def get_changed_files(
        self,
        ref1: Optional[str] = None,
        ref2: Optional[str] = None
    ) -> List[str]:
        """
        Get list of changed files between two references
        Returns a List of changed file paths
        """
        try:
            if ref1 is None and ref2 is None:
                # Both working directory - check for unstaged changes
                # Get both staged and unstaged changes
                staged_files = [item.a_path for item in self.repo.index.diff('HEAD')]
                unstaged_files = [item.a_path for item in self.repo.index.diff(None)]
                untracked_files = self.repo.untracked_files

                all_files = set(staged_files + unstaged_files + untracked_files)
                return sorted(all_files)

            elif ref1 is not None and ref2 is None:
                # Commit vs working directory - list all modified/added/deleted files
                commit = self.repo.commit(ref1)

                # Get staged changes
                staged_files = [item.a_path for item in self.repo.index.diff(commit)]

                # Get unstaged changes
                unstaged_files = [item.a_path for item in self.repo.index.diff(None)]

                # Get untracked files
                untracked_files = self.repo.untracked_files

                # Combine all changes
                all_files = set()
                for item in staged_files:
                    if item:
                        all_files.add(item)
                for item in unstaged_files:
                    if item:
                        all_files.add(item)
                for item in untracked_files:
                    if item:
                        all_files.add(item)

                return sorted(all_files)

            elif ref1 is None and ref2 is not None:
                # Working directory vs commit - same as above but reversed
                return self.get_changed_files(ref2, None)

            else:
                # Commit vs commit
                commit1 = self.repo.commit(ref1)
                commit2 = self.repo.commit(ref2)
                diffs = commit1.diff(commit2)

                # Extract file paths from diffs
                changed_files = []
                for diff in diffs:
                    if diff.a_path:
                        changed_files.append(diff.a_path)
                    if diff.b_path and diff.b_path != diff.a_path:
                        changed_files.append(diff.b_path)

                return sorted(set(changed_files))

        except GitError:
            raise
        except Exception as e:
            raise GitError(f"Failed to get changed files: {e}")

    @contextmanager
    def create_temp_files(
        self,
        ref1: Optional[str],
        ref2: Optional[str],
        file_path: Optional[str] = None
    ) -> Tuple[Path, Path]:
        """
        Create temporary files for diff comparison

        Args:
            ref1: First Git reference (None for working directory)
            ref2: Second Git reference (None for working directory)
            file_path: Specific file to diff (None for all changes)

        Returns the Tuple of (file1_path, file2_path) for comparison
        """
        temp_dir = None
        try:
            # Handle the case where both are Git references
            if ref1 is not None and ref2 is not None:
                if file_path is None:
                    # Show all changes between commits
                    changed_files = self.get_changed_files(ref1, ref2)
                    if not changed_files:
                        raise GitError(f"No changes found between {ref1} and {ref2}")

                    # For now, inform user to specify a file
                    # In future, we could create a combined diff
                    files_str = "\n  ".join(changed_files[:10])
                    if len(changed_files) > 10:
                        files_str += f"\n  ... and {len(changed_files) - 10} more files"

                    raise GitError(
                        f"Multiple files changed between {ref1} and {ref2}. "
                        f"Please specify which file to diff:\n  {files_str}\n\n"
                        f"Example: diffsense --git {ref1} {ref2} {changed_files[0]}"
                    )

                temp_dir = tempfile.mkdtemp(prefix="diffsense_")
                temp_path = Path(temp_dir)

                # Get content from both commits
                content1 = self.get_file_content(file_path, ref1)
                content2 = self.get_file_content(file_path, ref2)

                # Create temp files
                file1 = temp_path / f"file1_{content1.commit_sha[:7]}"
                file2 = temp_path / f"file2_{content2.commit_sha[:7]}"

                file1.write_text(content1.content, encoding='utf-8')
                file2.write_text(content2.content, encoding='utf-8')

                yield file1, file2

            # Handle working directory comparisons
            elif ref1 is None and ref2 is not None:
                # Working directory vs commit
                if file_path is None:
                    raise GitError("File path required for Git diff")

                temp_dir = tempfile.mkdtemp(prefix="diffsense_")
                temp_path = Path(temp_dir)

                # Working directory file
                repo_root = Path(self.repo.working_dir)
                working_file = repo_root / file_path

                if not working_file.exists():
                    raise GitError(f"File does not exist in working directory: {file_path}")

                # Get content from commit
                content2 = self.get_file_content(file_path, ref2)
                file2 = temp_path / f"file_{content2.commit_sha[:7]}"
                file2.write_text(content2.content, encoding='utf-8')

                yield working_file, file2

            elif ref1 is not None and ref2 is None:
                # Commit vs working directory
                if file_path is None:
                    raise GitError("File path required for Git diff")

                temp_dir = tempfile.mkdtemp(prefix="diffsense_")
                temp_path = Path(temp_dir)

                # Get content from commit
                content1 = self.get_file_content(file_path, ref1)
                file1 = temp_path / f"file_{content1.commit_sha[:7]}"
                file1.write_text(content1.content, encoding='utf-8')

                # Working directory file
                repo_root = Path(self.repo.working_dir)
                working_file = repo_root / file_path

                if not working_file.exists():
                    raise GitError(f"File does not exist in working directory: {file_path}")

                yield file1, working_file

            else:
                # Both are None - invalid
                raise GitError("At least one Git reference must be specified")

        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

    def format_file_label(
        self,
        file_path: str,
        ref: Optional[str] = None
    ) -> str:
        """
        Format a file label for diff display

        Args:
            file_path: Path to file
            ref: Git reference (None for working directory)

        Returns a formatted label string
        """
        if ref is None:
            return f"{file_path} (working directory)"

        try:
            commit = self.repo.commit(ref)
            short_sha = commit.hexsha[:7]

            # Include commit message summary if available
            msg_summary = commit.summary[:50]
            if len(commit.summary) > 50:
                msg_summary += "..."

            return f"{file_path} ({short_sha}: {msg_summary})"

        except Exception:
            # Fallback to simple format
            return f"{file_path} ({ref})"

    def validate_git_mode(self, args: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Validate and parse Git mode arguments

        Args:
            args: List of arguments after --git flag

        Returns a Tuple of (ref1, ref2, file_path)
        """
        if not args:
            raise GitError("Git mode requires at least one reference")

        if len(args) == 1:
            # diffsense --git HEAD
            # Compare working directory with HEAD (all changes)
            return args[0], None, None

        elif len(args) == 2:
            # Could be:
            # 1. diffsense --git HEAD~1 HEAD (compare two commits)
            # 2. diffsense --git HEAD file.py (compare file with HEAD)

            # Try to parse second argument as Git reference
            try:
                self.parse_git_reference(args[1])
                # It's a valid reference - compare two commits
                return args[0], args[1], None
            except GitError:
                # Not a valid reference, assume it's a file path
                return args[0], None, args[1]

        elif len(args) == 3:
            # diffsense --git HEAD~1 HEAD file.py
            return args[0], args[1], args[2]

        else:
            raise GitError("Too many arguments for Git mode")

    def is_git_repository(self) -> bool:
        """
        Check if current directory is within a Git repository
        Returns True if in a Git repository, False otherwise
        """
        try:
            _ = self.repo
            return True
        except GitError:
            return False
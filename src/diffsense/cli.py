"""
Command Line Interface for DiffSense
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.logging import RichHandler

from .diff_engine import DiffEngine
from .llm_manager import LLMManager
from .formatter import DiffFormatter
from .git_manager import GitManager
from .exceptions import DiffSenseError, GitError
from .conflict_resolver import ConflictResolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("diffsense")

app = typer.Typer(
    name="diffsense",
    help="AI-powered code diff tool with Git integration",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Display version information."""
    if value:
        from . import __version__
        console.print(f"DiffSense version {__version__}")
        raise typer.Exit()


@app.command()
def main(
    file1: Optional[Path] = typer.Argument(None, help="First file to compare (or Git reference with --git)"),
    file2: Optional[Path] = typer.Argument(None, help="Second file to compare (or Git reference/file with --git)"),
    git_args: Optional[List[str]] = typer.Argument(None, help="Additional Git arguments"),
    git: bool = typer.Option(
        False,
        "--git",
        help="Git mode: compare commits, branches, or working directory"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override default LLM model"
    ),
    no_ai: bool = typer.Option(
        False,
        "--no-ai",
        help="Skip AI analysis and show just the diff"
    ),
    full_context: bool = typer.Option(
        False,
        "--full-context",
        help="Include complete file content in AI analysis"
    ),
    context: int = typer.Option(
        3,
        "--context",
        "-c",
        help="Number of context lines to show around changes"
    ),
    resolve_conflicts: bool = typer.Option(
        False,
        "--resolve-conflicts",
        help="Resolve merge conflicts in the file using AI"
    ),
    context_files: Optional[List[Path]] = typer.Option(
        None,
        "--context-file",
        "-f",
        help="Additional files to provide context for conflict resolution (can be used multiple times)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version information"
    ),
) -> None:

    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    try:
        if resolve_conflicts:
            _handle_conflict_resolution_mode(file1, model, context_files, verbose)
        elif git:
            _handle_git_mode(file1, file2, git_args, model, no_ai, full_context, context, verbose)
        else:
            _handle_file_mode(file1, file2, model, no_ai, full_context, context, verbose)

    except DiffSenseError as e:
        logger.error(f"DiffSense error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(code=1)


def _handle_conflict_resolution_mode(
    file_path: Optional[Path],
    model: Optional[str],
    context_files: Optional[List[Path]],
    verbose: bool
) -> None:
    """
    Handle conflict resolution mode

    Args:
        file_path: Path to file with conflicts
        model: Optional model override
        context_files: Optional context files
        verbose: Enable verbose output
    """
    if not file_path:
        raise DiffSenseError("File path required for conflict resolution mode")

    if not file_path.exists():
        raise DiffSenseError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise DiffSenseError(f"Path is not a file: {file_path}")

    logger.debug(f"Resolving conflicts in: {file_path}")

    # Initialize LLM manager
    try:
        llm_manager = LLMManager(model_id=model)
    except Exception as e:
        raise DiffSenseError(f"Failed to initialize AI model: {e}")

    # Initialize conflict resolver
    resolver = ConflictResolver(llm_manager)

    # Resolve conflicts
    console.print(f"\n[bold]Analyzing merge conflicts in {file_path}...[/bold]")

    try:
        resolutions, total_tokens = resolver.resolve_conflicts(file_path, context_files)

        if not resolutions:
            console.print("[green]No conflicts found in the file.[/green]")
            return

        # Display resolutions
        console.print(f"\n[bold]Found {len(resolutions)} conflict(s)[/bold]")
        console.print(f"[dim]Total tokens used: {total_tokens}[/dim]\n")

        for i, resolution in enumerate(resolutions, 1):
            _display_conflict_resolution(i, resolution)

        # Ask user if they want to apply resolutions
        if len(resolutions) == 1:
            prompt_text = "\nApply this resolution? [Y/n]: "
        else:
            prompt_text = "\nApply all resolutions? [Y/n]: "

        apply = typer.confirm(prompt_text, default=True)

        if apply:
            # Create backup
            backup_path = Path(f"{file_path}.backup")
            import shutil
            shutil.copy2(file_path, backup_path)
            console.print(f"[dim]Created backup: {backup_path}[/dim]")

            # Apply resolutions
            resolved_content = resolver.apply_resolutions(file_path, resolutions)
            file_path.write_text(resolved_content, encoding='utf-8')
            console.print(f"[green]✓ Resolutions applied to {file_path}[/green]")
        else:
            console.print("[yellow]Resolutions not applied.[/yellow]")

    except Exception as e:
        logger.error(f"Conflict resolution failed: {e}")
        raise DiffSenseError(f"Failed to resolve conflicts: {e}")


def _display_conflict_resolution(index: int, resolution) -> None:
    """
    Display a single conflict resolution

    Args:
        index: Conflict number
        resolution: ConflictResolution object
    """
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table

    # Create a table for the conflict info
    info_table = Table.grid(padding=1)
    info_table.add_column(style="bold")
    info_table.add_column()

    info_table.add_row(
        "Conflict Location:",
        f"Lines {resolution.section.start_line}-{resolution.section.end_line}"
    )
    info_table.add_row(
        "Confidence:",
        _format_confidence(resolution.confidence)
    )

    # Display current vs incoming
    console.print(Panel(
        info_table,
        title=f"[bold]Conflict #{index}[/bold]",
        border_style="blue"
    ))

    # Show the conflict versions side by side
    conflict_table = Table(show_header=True, header_style="bold", show_lines=True)
    conflict_table.add_column("Current Version (ours)", style="red")
    conflict_table.add_column("Incoming Version (theirs)", style="green")

    conflict_table.add_row(
        resolution.section.current_content or "(empty)",
        resolution.section.incoming_content or "(empty)"
    )

    console.print(conflict_table)

    # Show resolution
    console.print("\n[bold yellow]Proposed Resolution:[/bold yellow]")

    # Try to detect language for syntax highlighting
    lang = "python"  # Default, could be improved
    if resolution.resolved_content:
        syntax = Syntax(
            resolution.resolved_content,
            lang,
            theme="monokai",
            line_numbers=False
        )
        console.print(syntax)
    else:
        console.print("[dim](empty resolution)[/dim]")

    # Show explanation
    console.print(f"\n[bold]Explanation:[/bold] {resolution.explanation}")

    # Show alternatives if any
    if resolution.alternative_resolutions:
        console.print("\n[bold yellow]Alternative Resolutions:[/bold yellow]")
        for i, alt in enumerate(resolution.alternative_resolutions, 1):
            console.print(f"\n[dim]Alternative {i}:[/dim]")
            alt_syntax = Syntax(
                alt["resolution"],
                lang,
                theme="monokai",
                line_numbers=False
            )
            console.print(alt_syntax)
            console.print(f"[dim]Rationale: {alt['explanation']}[/dim]")

    console.print("─" * 80)


def _format_confidence(confidence) -> str:
    """
    Format confidence level with color

    Args:
        confidence: ConflictResolutionConfidence enum

    Returns the formatted string with color
    """
    from .conflict_resolver import ConflictResolutionConfidence

    if confidence == ConflictResolutionConfidence.HIGH:
        return "[bold green]HIGH[/bold green]"
    elif confidence == ConflictResolutionConfidence.MEDIUM:
        return "[bold yellow]MEDIUM[/bold yellow]"
    else:
        return "[bold red]LOW[/bold red]"


def _handle_git_mode(
    file1: Optional[Path],
    file2: Optional[Path],
    git_args: Optional[List[str]],
    model: Optional[str],
    no_ai: bool,
    full_context: bool,
    context: int,
    verbose: bool
) -> None:
    """
    Handle Git mode operations
    """
    logger.debug("Git mode enabled")

    # Collect all arguments for Git mode
    args = []
    if file1:
        args.append(str(file1))
    if file2:
        args.append(str(file2))
    if git_args:
        args.extend(git_args)

    if not args:
        raise DiffSenseError("Git mode requires at least one reference (e.g., --git HEAD)")

    # Initialize Git manager
    try:
        git_manager = GitManager()
    except GitError as e:
        raise DiffSenseError(f"Git initialization failed: {e}")

    # Validate and parse Git arguments
    try:
        ref1, ref2, file_path = git_manager.validate_git_mode(args)
        logger.debug(f"Git mode parsed: ref1={ref1}, ref2={ref2}, file_path={file_path}")
    except GitError as e:
        raise DiffSenseError(f"Invalid Git arguments: {e}")

    # Handle case where no file is specified
    if file_path is None and ref1 is not None and ref2 is None:
        # Show changed files for single ref
        try:
            changed_files = git_manager.get_changed_files(ref1, None)
            if not changed_files:
                console.print(f"[green]No changes found between {ref1} and working directory.[/green]")
                return

            console.print(f"\n[bold]Changed files between {ref1} and working directory:[/bold]")
            for file in changed_files:
                console.print(f"  • {file}")
            console.print(f"\n[dim]To see changes for a specific file, use: diffsense --git {ref1} <filename>[/dim]")
            return
        except GitError as e:
            raise DiffSenseError(f"Failed to get changed files: {e}")

    # Create temporary files and run diff
    try:
        with git_manager.create_temp_files(ref1, ref2, file_path) as (temp_file1, temp_file2):
            # Format file labels for display
            if file_path:
                label1 = git_manager.format_file_label(file_path, ref1)
                label2 = git_manager.format_file_label(file_path, ref2)
            else:
                # For full repository diffs (future enhancement)
                label1 = f"Commit {ref1}" if ref1 else "Working Directory"
                label2 = f"Commit {ref2}" if ref2 else "Working Directory"

            logger.debug(f"Comparing: {label1} vs {label2}")

            # Read file contents
            original_content = temp_file1.read_text(encoding='utf-8')
            modified_content = temp_file2.read_text(encoding='utf-8')

            # Run the diff analysis
            _run_diff_analysis(
                original_content=original_content,
                modified_content=modified_content,
                file1_name=label1,
                file2_name=label2,
                context_lines=context,
                model=model,
                no_ai=no_ai,
                full_context=full_context,
                verbose=verbose
            )

    except GitError as e:
        raise DiffSenseError(f"Git operation failed: {e}")


def _handle_file_mode(
    file1: Optional[Path],
    file2: Optional[Path],
    model: Optional[str],
    no_ai: bool,
    full_context: bool,
    context: int,
    verbose: bool
) -> None:
    """
    Handle traditional file mode operations
    """
    if not file1 or not file2:
        raise DiffSenseError("Two file paths are required in file mode")

    # Validate input files
    _validate_files(file1, file2)

    # Read file contents
    logger.debug(f"Reading files: {file1} and {file2}")
    original_content = file1.read_text(encoding='utf-8')
    modified_content = file2.read_text(encoding='utf-8')

    # Run the diff analysis
    _run_diff_analysis(
        original_content=original_content,
        modified_content=modified_content,
        file1_name=file1.name,
        file2_name=file2.name,
        context_lines=context,
        model=model,
        no_ai=no_ai,
        full_context=full_context,
        verbose=verbose
    )


def _run_diff_analysis(
    original_content: str,
    modified_content: str,
    file1_name: str,
    file2_name: str,
    context_lines: int,
    model: Optional[str],
    no_ai: bool,
    full_context: bool,
    verbose: bool
) -> None:
    """
    Run the core diff analysis and display results

    This is the common logic shared between file and Git modes
    """
    # Compute diff
    logger.debug("Computing diff...")
    diff_engine = DiffEngine(context_lines=context_lines)
    diff_blocks = diff_engine.compute_diff(original_content, modified_content)

    if not diff_blocks:
        console.print("[green]No differences found between files.[/green]")
        return

    # Format and display diff
    formatter = DiffFormatter()
    formatted_diff = formatter.format_diff(diff_blocks, file1_name, file2_name)
    console.print(formatted_diff)

    # Generate AI analysis if requested
    if not no_ai:
        logger.debug("Generating AI analysis...")
        try:
            llm_manager = LLMManager(model_id=model)

            # Decide whether to include full context
            should_include_context = full_context or _should_include_full_context_from_content(
                original_content, modified_content
            )

            if should_include_context:
                logger.debug("Attempting AI analysis with full file context")
                analysis = llm_manager.analyze_diff(
                    diff_blocks,
                    original_content=original_content,
                    modified_content=modified_content,
                    file1_name=file1_name,
                    file2_name=file2_name
                )
            else:
                if full_context:
                    logger.info("Files too large for full context analysis, using diff-only mode")
                analysis = llm_manager.analyze_diff(diff_blocks)

            console.print("\n" + "─" * 80)
            console.print("[bold yellow]AI Analysis:[/bold yellow]")
            console.print(analysis)

        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            console.print(
                "\n[yellow]Warning: AI analysis unavailable. "
                "Run with --no-ai to suppress this message.[/yellow]"
            )


def _should_include_full_context(file1: Path, file2: Path) -> bool:
    """
    Decide if files are small enough to include full context in AI analysis.
    """
    try:
        content1 = file1.read_text(encoding='utf-8')
        content2 = file2.read_text(encoding='utf-8')
        return _should_include_full_context_from_content(content1, content2)
    except Exception as e:
        logger.debug(f"Could not determine file sizes: {e}")
        return False


def _should_include_full_context_from_content(content1: str, content2: str) -> bool:
    """
    Decide if content is small enough to include full context in AI analysis.
    """
    max_lines = 100
    max_chars = 5000  # Additional character limit

    lines1 = len(content1.splitlines())
    lines2 = len(content2.splitlines())
    chars1 = len(content1)
    chars2 = len(content2)

    max_file_lines = max(lines1, lines2)
    max_file_chars = max(chars1, chars2)

    logger.debug(f"Content sizes: {lines1} lines ({chars1} chars), {lines2} lines ({chars2} chars)")
    logger.debug(f"Limits: {max_lines} lines, {max_chars} chars")

    # Both conditions must be met
    size_ok = max_file_lines <= max_lines and max_file_chars <= max_chars
    logger.debug(f"Decision: {size_ok}")

    return size_ok


def _validate_files(file1: Path, file2: Path) -> None:
    """
    Validate that input files exist and are readable.
    """
    for file_path in [file1, file2]:
        if not file_path.exists():
            raise DiffSenseError(f"File does not exist: {file_path}")
        if not file_path.is_file():
            raise DiffSenseError(f"Path is not a file: {file_path}")
        try:
            content = file_path.read_text(encoding='utf-8')
            # Check for binary content
            if '\x00' in content:
                raise DiffSenseError(f"Binary file detected: {file_path}")
        except UnicodeDecodeError:
            raise DiffSenseError(f"File is not a valid text file: {file_path}")
        except PermissionError:
            raise DiffSenseError(f"Permission denied reading file: {file_path}")


if __name__ == "__main__":
    typer.run(main)
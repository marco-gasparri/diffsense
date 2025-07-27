"""
Command Line Interface for DiffSense
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .diff_engine import DiffEngine
from .llm_manager import LLMManager
from .formatter import DiffFormatter
from .exceptions import DiffSenseError

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
    help="AI-powered code diff tool",
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
    file1: Path = typer.Argument(..., help="First file to compare"),
    file2: Path = typer.Argument(..., help="Second file to compare"),
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
        # Validate input files
        _validate_files(file1, file2)

        # Read file contents
        logger.debug(f"Reading files: {file1} and {file2}")
        original_content = file1.read_text(encoding='utf-8')
        modified_content = file2.read_text(encoding='utf-8')

        # Compute diff
        logger.debug("Computing diff...")
        diff_engine = DiffEngine(context_lines=context)
        diff_blocks = diff_engine.compute_diff(original_content, modified_content)

        if not diff_blocks:
            console.print("[green]No differences found between files.[/green]")
            return

        # Format and display diff
        formatter = DiffFormatter()
        formatted_diff = formatter.format_diff(diff_blocks, file1.name, file2.name)
        console.print(formatted_diff)

        # Generate AI analysis if requested
        if not no_ai:
            logger.debug("Generating AI analysis...")
            try:
                llm_manager = LLMManager(model_id=model)

                # Decide whether to include full context
                should_include_context = full_context or _should_include_full_context(file1, file2)

                if should_include_context:
                    logger.debug("Attempting AI analysis with full file context")
                    analysis = llm_manager.analyze_diff(
                        diff_blocks,
                        original_content=original_content,
                        modified_content=modified_content,
                        file1_name=file1.name,
                        file2_name=file2.name
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

    except DiffSenseError as e:
        logger.error(f"DiffSense error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(code=1)


def _should_include_full_context(file1: Path, file2: Path) -> bool:
    """
    Decide if files are small enough to include full context in AI analysis.
    """
    max_lines = 80
    max_chars = 3000  # Additional character limit

    try:
        content1 = file1.read_text(encoding='utf-8')
        content2 = file2.read_text(encoding='utf-8')

        lines1 = len(content1.splitlines())
        lines2 = len(content2.splitlines())
        chars1 = len(content1)
        chars2 = len(content2)

        max_file_lines = max(lines1, lines2)
        max_file_chars = max(chars1, chars2)

        logger.debug(f"File sizes: {file1.name}={lines1} lines ({chars1} chars), {file2.name}={lines2} lines ({chars2} chars)")
        logger.debug(f"Limits: {max_lines} lines, {max_chars} chars")

        # Both conditions must be met
        size_ok = max_file_lines <= max_lines and max_file_chars <= max_chars
        logger.debug(f"Decision: {size_ok}")

        return size_ok
    except Exception as e:
        logger.debug(f"Could not determine file sizes: {e}")
        return False


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
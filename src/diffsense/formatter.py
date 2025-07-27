"""
Rich text formatter for displaying diffs with syntax highlighting
"""

from typing import List, Optional
from rich.text import Text
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from .diff_engine import DiffBlock, LineDiff, ChangeType
from .exceptions import FormattingError


class DiffFormatter:
    """
    Formatter for diff output with rich text features
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the formatter
        """
        self.console = console or Console()

        # Color scheme for different change types
        self.colors = {
            ChangeType.INSERT: "green",
            ChangeType.DELETE: "red",
            ChangeType.REPLACE: "yellow",
            ChangeType.EQUAL: "dim white"
        }

        # Symbols for different change types
        self.symbols = {
            ChangeType.INSERT: "+",
            ChangeType.DELETE: "-",
            ChangeType.REPLACE: "~",
            ChangeType.EQUAL: " "
        }

    def format_diff(
        self,
        blocks: List[DiffBlock],
        file1_name: str = "file1",
        file2_name: str = "file2"
    ) -> Group:
        """
        Format a complete diff with header and all blocks
        Returns Rich Group object containing the formatted diff
        """
        try:
            if not blocks:
                # Don't create panels for identical files - return simple text
                return Group(Text("No differences found", style="green"))

            # Create header
            header = self._create_diff_header(file1_name, file2_name, blocks)

            # Format each block
            formatted_blocks = []
            for i, block in enumerate(blocks):
                formatted_block = self._format_diff_block(block, i + 1)
                formatted_blocks.append(formatted_block)

            # Create summary
            summary = self._create_diff_summary(blocks)

            return Group(header, *formatted_blocks, summary)

        except Exception as e:
            raise FormattingError(f"Failed to format diff: {e}") from e

    def _create_diff_header(
        self,
        file1_name: str,
        file2_name: str,
        blocks: List[DiffBlock]
    ) -> Panel:
        """
        Create a fancy header for the diff output
        Returns a Rich Panel containing the header
        """
        # Calculate statistics
        total_changes = sum(len(block.lines) for block in blocks)
        total_blocks = len(blocks)

        # Create header content
        header_table = Table.grid(padding=1)
        header_table.add_column(style="bold red")
        header_table.add_column(style="bold green")

        header_table.add_row(f"--- {file1_name}", f"+++ {file2_name}")

        stats_text = Text()
        stats_text.append(f"Blocks: {total_blocks} | ", style="dim")
        stats_text.append(f"Changes: {total_changes}", style="dim")

        content = Group(header_table, stats_text)

        return Panel(
            content,
            title="[bold]Diff Analysis[/bold]",
            border_style="blue"
        )

    def _format_diff_block(self, block: DiffBlock, block_number: int) -> Panel:
        """
        Format a single diff block with syntax highlighting
        Returns a Rich Panel containing the formatted block
        """
        # Create block header
        header_text = Text()
        header_text.append(f"@@ {block.header} @@", style="bold cyan")

        # Format lines
        line_texts = []
        for line in block.lines:
            formatted_line = self._format_line(line)
            line_texts.append(formatted_line)

        # Combine content
        content = Group(header_text, *line_texts)

        # Get block statistics for title
        summary = block.get_changes_summary()
        changes_info = []
        if summary[ChangeType.INSERT] > 0:
            changes_info.append(f"+{summary[ChangeType.INSERT]}")
        if summary[ChangeType.DELETE] > 0:
            changes_info.append(f"-{summary[ChangeType.DELETE]}")
        if summary[ChangeType.REPLACE] > 0:
            changes_info.append(f"~{summary[ChangeType.REPLACE]}")

        title = f"Block {block_number}"
        if changes_info:
            title += f" ({', '.join(changes_info)})"

        return Panel(
            content,
            title=title,
            border_style="dim",
            padding=(0, 1)
        )

    def _format_line(self, line: LineDiff) -> Text:
        """
        Format a single line with appropriate styling and inline changes
        Returns Rich Text object with formatted line
        """
        symbol = self.symbols[line.change_type]
        color = self.colors[line.change_type]

        # Choose content based on change type
        if line.change_type == ChangeType.DELETE:
            content = line.old_line
            line_num = f"{line.old_line_num:4d}" if line.old_line_num > 0 else "    "
        elif line.change_type == ChangeType.INSERT:
            content = line.new_line
            line_num = f"{line.new_line_num:4d}" if line.new_line_num > 0 else "    "
        elif line.change_type == ChangeType.EQUAL:
            content = line.old_line  # Same as new_line for equal lines
            line_num = f"{line.old_line_num:4d}"
        else:  # Replace
            content = line.new_line
            line_num = f"{line.new_line_num:4d}" if line.new_line_num > 0 else "    "

        # Create the formatted line
        formatted = Text()

        # Add line number (dimmed)
        formatted.append(f"{line_num} ", style="dim")

        # Add change symbol
        formatted.append(f"{symbol} ", style=color)

        # Add content with inline highlighting for replacements
        if line.change_type == ChangeType.REPLACE and line.has_inline_changes():
            formatted.append(self._format_inline_changes(content, line.inline_changes, color))
        else:
            formatted.append(content, style=color)

        return formatted

    def _format_inline_changes(
        self,
        content: str,
        changes: List[tuple],
        base_color: str
    ) -> Text:
        """
        Format a line with inline character-level changes highlighted.
        Args:
            content: Line content
            changes: List of (start, end) tuples for changed regions
            base_color: Base color for the line
        Returns a Rich Text object with inline highlighting
        """
        if not changes:
            return Text(content, style=base_color)

        formatted = Text()
        last_pos = 0

        for start, end in changes:
            # Add unchanged part
            if start > last_pos:
                formatted.append(content[last_pos:start], style=base_color)

            # Add changed part with highlighting
            changed_text = content[start:end]
            formatted.append(changed_text, style=f"bold {base_color} on dim")

            last_pos = end

        # Add remaining unchanged part
        if last_pos < len(content):
            formatted.append(content[last_pos:], style=base_color)

        return formatted

    def _create_diff_summary(self, blocks: List[DiffBlock]) -> Panel:
        """
        Create a summary panel with diff statistics
        Returns a Rich Panel containing the summary
        """
        # Compute overall statistics
        total_insertions = 0
        total_deletions = 0
        total_replacements = 0

        for block in blocks:
            summary = block.get_changes_summary()
            total_insertions += summary[ChangeType.INSERT]
            total_deletions += summary[ChangeType.DELETE]
            total_replacements += summary[ChangeType.REPLACE]

        # Create summary table
        summary_table = Table.grid(padding=1)
        summary_table.add_column(style="green")
        summary_table.add_column(style="red")
        summary_table.add_column(style="yellow")

        summary_table.add_row(
            f"+ {total_insertions} insertions",
            f"- {total_deletions} deletions",
            f"~ {total_replacements} modifications"
        )

        return Panel(
            summary_table,
            title="[bold]Summary[/bold]",
            border_style="dim"
        )

    def format_plain_diff(self, blocks: List[DiffBlock]) -> str:
        """
        Format diff blocks as plain text (for AI processing)
        """
        lines = []

        for block in blocks:
            lines.append(f"@@ {block.header} @@")

            for line in block.lines:
                symbol = self.symbols[line.change_type]

                if line.change_type == ChangeType.DELETE:
                    content = line.old_line
                elif line.change_type == ChangeType.INSERT:
                    content = line.new_line
                elif line.change_type == ChangeType.EQUAL:
                    content = line.old_line
                else:  # Replace
                    content = line.new_line

                lines.append(f"{symbol} {content}")

        return "\n".join(lines)
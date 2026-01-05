"""
Output formatting module for extracted data
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


class OutputFormatter:
    """Handles formatting and exporting extracted data"""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize the output formatter

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def format_as_json(
        self,
        data: str,
        pretty: bool = True
    ) -> str:
        """
        Format data as JSON

        Args:
            data: Raw extracted data (should be JSON string)
            pretty: Whether to pretty-print the JSON

        Returns:
            Formatted JSON string
        """
        try:
            # Try to parse as JSON
            parsed = json.loads(data)
            if pretty:
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            # If not valid JSON, wrap it
            console.print("[yellow]Data is not valid JSON, wrapping in object[/yellow]")
            result = {"extracted_data": data}
            if pretty:
                return json.dumps(result, indent=2, ensure_ascii=False)
            return json.dumps(result, ensure_ascii=False)

    def save_json(
        self,
        data: str,
        filename: str,
        pretty: bool = True
    ) -> Path:
        """
        Save data as JSON file

        Args:
            data: Data to save
            filename: Output filename (without extension)
            pretty: Whether to pretty-print

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.json"
        formatted_data = self.format_as_json(data, pretty=pretty)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_data)

        console.print(f"[green]✓ Saved JSON to {output_path}[/green]")
        return output_path

    def save_text(
        self,
        data: str,
        filename: str
    ) -> Path:
        """
        Save data as plain text file

        Args:
            data: Data to save
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data)

        console.print(f"[green]✓ Saved text to {output_path}[/green]")
        return output_path

    def save_csv(
        self,
        data: str,
        filename: str
    ) -> Path:
        """
        Save data as CSV file (attempts to parse structured data)

        Args:
            data: Data to save (should be JSON with tabular structure)
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.csv"

        try:
            # Try to parse as JSON
            parsed = json.loads(data)

            # Convert to DataFrame
            df = self._json_to_dataframe(parsed)

            # Save as CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            console.print(f"[green]✓ Saved CSV to {output_path}[/green]")

        except Exception as e:
            console.print(f"[yellow]Could not convert to CSV: {str(e)}[/yellow]")
            console.print("[yellow]Saving as raw text in CSV format[/yellow]")

            # Fallback: save as single-column CSV
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['extracted_data'])
                writer.writerow([data])

            console.print(f"[green]✓ Saved fallback CSV to {output_path}[/green]")

        return output_path

    def save_markdown(
        self,
        data: str,
        filename: str,
        title: Optional[str] = None
    ) -> Path:
        """
        Save data as Markdown file

        Args:
            data: Data to save
            filename: Output filename (without extension)
            title: Optional title for the document

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.md"

        content = ""
        if title:
            content += f"# {title}\n\n"

        # Try to format as code block if it's JSON
        try:
            parsed = json.loads(data)
            content += "```json\n"
            content += json.dumps(parsed, indent=2, ensure_ascii=False)
            content += "\n```\n"
        except json.JSONDecodeError:
            content += data

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        console.print(f"[green]✓ Saved Markdown to {output_path}[/green]")
        return output_path

    def _json_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert JSON data to pandas DataFrame

        Args:
            data: Parsed JSON data

        Returns:
            pandas DataFrame
        """
        # Handle different JSON structures
        if isinstance(data, list):
            # List of objects -> direct conversion
            return pd.DataFrame(data)

        elif isinstance(data, dict):
            # Look for common table-like structures
            if "line_items" in data:
                # Financial statement format
                return pd.DataFrame(data["line_items"])
            elif "rows" in data:
                return pd.DataFrame(data["rows"])
            elif "data" in data:
                return pd.DataFrame(data["data"])
            else:
                # Try to convert dict to single-row DataFrame
                return pd.DataFrame([data])

        else:
            # Fallback: create single-cell DataFrame
            return pd.DataFrame([{"data": str(data)}])

    def display_preview(
        self,
        data: str,
        max_length: int = 500
    ):
        """
        Display a preview of extracted data in the console

        Args:
            data: Data to preview
            max_length: Maximum length to display
        """
        console.print("\n[bold cyan]Extracted Data Preview:[/bold cyan]")
        console.print("─" * 80)

        if len(data) > max_length:
            preview = data[:max_length] + "..."
            console.print(preview)
            console.print(f"\n[dim](Showing first {max_length} characters of {len(data)} total)[/dim]")
        else:
            console.print(data)

        console.print("─" * 80)

    def display_table_preview(
        self,
        data: str,
        max_rows: int = 10
    ):
        """
        Display extracted data as a formatted table (if possible)

        Args:
            data: Data to display (should be JSON with tabular structure)
            max_rows: Maximum number of rows to display
        """
        try:
            parsed = json.loads(data)
            df = self._json_to_dataframe(parsed)

            # Create Rich table
            table = Table(show_header=True, header_style="bold cyan")

            # Add columns
            for col in df.columns:
                table.add_column(str(col))

            # Add rows (up to max_rows)
            for idx, row in df.head(max_rows).iterrows():
                table.add_row(*[str(val) for val in row])

            console.print(table)

            if len(df) > max_rows:
                console.print(f"\n[dim](Showing {max_rows} of {len(df)} rows)[/dim]")

        except Exception as e:
            console.print(f"[yellow]Could not display as table: {str(e)}[/yellow]")
            self.display_preview(data)

    def save_multiple_formats(
        self,
        data: str,
        base_filename: str,
        formats: List[str] = ["json", "txt"]
    ) -> Dict[str, Path]:
        """
        Save data in multiple formats

        Args:
            data: Data to save
            base_filename: Base filename (without extension)
            formats: List of formats to save (json, txt, csv, md)

        Returns:
            Dictionary mapping format to file path
        """
        saved_files = {}

        for fmt in formats:
            if fmt == "json":
                saved_files["json"] = self.save_json(data, base_filename)
            elif fmt == "txt":
                saved_files["txt"] = self.save_text(data, base_filename)
            elif fmt == "csv":
                saved_files["csv"] = self.save_csv(data, base_filename)
            elif fmt == "md":
                saved_files["md"] = self.save_markdown(data, base_filename)
            else:
                console.print(f"[yellow]Unknown format: {fmt}[/yellow]")

        return saved_files

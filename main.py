#!/usr/bin/env python3
"""
PDF Extractor - CLI application for extracting data from PDFs using OpenAI's API
"""

import os
from pathlib import Path
from typing import Optional, List
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

from src.pdf_uploader import PDFUploader
from src.extractor import ExtractionEngine
from src.validator import ExtractionValidator
from src.formatter import OutputFormatter
from src.prompts import PromptTemplates

# Load environment variables
load_dotenv()

# Initialize CLI app
app = typer.Typer(
    name="pdf-extractor",
    help="Extract structured data from PDFs using OpenAI's API",
    add_completion=False
)

console = Console()


@app.command()
def extract(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Custom extraction prompt"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Use a prompt template (table, financial, custom)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output filename (without extension)"),
    format: List[str] = typer.Option(["json"], "--format", "-f", help="Output format(s): json, txt, csv, md"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate extracted data"),
    model: str = typer.Option("gpt-5.2", "--model", "-m", help="Model to use (gpt-4, gpt-5.2)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show preview of extracted data")
):
    """
    Extract data from a PDF file

    Examples:

        # Extract with custom prompt
        python main.py extract document.pdf -p "Extract all tables"

        # Extract balance sheet using template
        python main.py extract financials.pdf -t financial --validate

        # Save in multiple formats
        python main.py extract data.pdf -p "Extract summary" -f json -f csv -f md
    """
    try:
        # Display header
        console.print(Panel.fit(
            "[bold cyan]PDF Extractor[/bold cyan]\n"
            "Powered by OpenAI API",
            border_style="cyan"
        ))

        # Get API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OpenAI API key not found.[/red]")
            console.print("Set OPENAI_API_KEY environment variable or use --api-key option")
            raise typer.Exit(1)

        # Initialize components
        uploader = PDFUploader(api_key=api_key)
        extractor = ExtractionEngine(api_key=api_key, model=model)
        formatter = OutputFormatter()
        prompt_templates = PromptTemplates()

        # Upload PDF
        console.print(f"\n[bold]Step 1: Uploading PDF[/bold]")
        file_id = uploader.upload_pdf(pdf_path)

        # Generate prompt
        console.print(f"\n[bold]Step 2: Preparing extraction prompt[/bold]")

        if prompt:
            # Use custom prompt
            extraction_prompt = prompt_templates.custom_prompt(prompt)
            console.print(f"[dim]Using custom prompt[/dim]")
        elif template == "table":
            extraction_prompt = prompt_templates.table_extraction_prompt("table")
            console.print(f"[dim]Using table extraction template[/dim]")
        elif template == "financial":
            extraction_prompt = prompt_templates.financial_statement_prompt()
            console.print(f"[dim]Using financial statement template[/dim]")
        else:
            # Default prompt
            extraction_prompt = "Extract all relevant information from this PDF in a structured format (JSON preferred)."
            console.print(f"[dim]Using default extraction prompt[/dim]")

        # Extract data
        console.print(f"\n[bold]Step 3: Extracting data with {model}[/bold]")
        result = extractor.multi_page_extract(file_id, extraction_prompt)

        if not result["success"]:
            console.print(f"[red]Extraction failed: {result['error']}[/red]")
            raise typer.Exit(1)

        extracted_data = result["output_text"]

        # Show preview
        if preview:
            formatter.display_preview(extracted_data)

        # Validate if requested
        validation_result = None
        if validate:
            console.print(f"\n[bold]Step 4: Validating extracted data[/bold]")
            validator = ExtractionValidator(api_key=api_key, model=model)
            validation_result = validator.validate(
                file_id=file_id,
                data_type="data",
                extracted_data=extracted_data
            )

            if not validation_result["is_valid"]:
                console.print("[yellow]Warning: Validation found potential issues[/yellow]")
                if validation_result["errors"]:
                    for error in validation_result["errors"]:
                        console.print(f"  [yellow]• {error}[/yellow]")

        # Save output
        step_num = 5 if validate else 4
        console.print(f"\n[bold]Step {step_num}: Saving output[/bold]")

        # Determine output filename
        if output:
            base_filename = output
        else:
            pdf_name = Path(pdf_path).stem
            base_filename = f"{pdf_name}_extracted"

        # Save in requested formats
        saved_files = formatter.save_multiple_formats(extracted_data, base_filename, format)

        # Summary
        console.print(f"\n[bold green]✓ Extraction complete![/bold green]")
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Model: {model}")
        console.print(f"  Input tokens: {result['input_tokens']}")
        console.print(f"  Output tokens: {result['output_tokens']}")

        if validation_result:
            console.print(f"  Validation: {'✓ Passed' if validation_result['is_valid'] else '⚠ Issues found'}")
            console.print(f"  Confidence: {validation_result['confidence']}%")

        console.print(f"\n[bold]Saved files:[/bold]")
        for fmt, path in saved_files.items():
            console.print(f"  {fmt.upper()}: {path}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def batch(
    pdf_dir: str = typer.Argument(..., help="Directory containing PDF files"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Extraction prompt for all PDFs"),
    output_dir: str = typer.Option("output", "--output-dir", "-o", help="Output directory"),
    format: str = typer.Option("json", "--format", "-f", help="Output format"),
    model: str = typer.Option("gpt-5.2", "--model", "-m", help="Model to use"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key")
):
    """
    Extract data from multiple PDF files in a directory

    Example:
        python main.py batch ./pdfs -p "Extract summary" -o results
    """
    try:
        # Get API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OpenAI API key not found[/red]")
            raise typer.Exit(1)

        # Find all PDFs
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))

        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {pdf_dir}[/yellow]")
            raise typer.Exit(1)

        console.print(f"[cyan]Found {len(pdf_files)} PDF files[/cyan]")

        # Initialize components
        uploader = PDFUploader(api_key=api_key)
        extractor = ExtractionEngine(api_key=api_key, model=model)
        formatter = OutputFormatter(output_dir=output_dir)
        prompt_templates = PromptTemplates()

        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            console.print(f"\n[bold cyan]Processing {i}/{len(pdf_files)}: {pdf_path.name}[/bold cyan]")

            try:
                # Upload
                file_id = uploader.upload_pdf(str(pdf_path))

                # Extract
                extraction_prompt = prompt_templates.custom_prompt(prompt)
                result = extractor.extract(file_id, extraction_prompt)

                if result["success"]:
                    # Save
                    base_filename = pdf_path.stem
                    if format == "json":
                        formatter.save_json(result["output_text"], base_filename)
                    elif format == "txt":
                        formatter.save_text(result["output_text"], base_filename)
                    elif format == "csv":
                        formatter.save_csv(result["output_text"], base_filename)

                    console.print(f"[green]✓ Completed {pdf_path.name}[/green]")
                else:
                    console.print(f"[red]✗ Failed {pdf_path.name}: {result['error']}[/red]")

            except Exception as e:
                console.print(f"[red]✗ Error processing {pdf_path.name}: {str(e)}[/red]")
                continue

        console.print(f"\n[bold green]Batch processing complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key")
):
    """
    Display information about uploaded files and configuration
    """
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OpenAI API key not found[/red]")
            raise typer.Exit(1)

        uploader = PDFUploader(api_key=api_key)

        console.print("[bold cyan]Uploaded Files:[/bold cyan]")
        files = uploader.list_uploaded_files()

        if not files:
            console.print("[dim]No files uploaded yet[/dim]")
        else:
            for f in files:
                console.print(f"\n  ID: {f['id']}")
                console.print(f"  Name: {f['filename']}")
                console.print(f"  Size: {f['bytes']} bytes")
                console.print(f"  Created: {f['created_at']}")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Display version information"""
    from src import __version__
    console.print(f"PDF Extractor v{__version__}")
    console.print("Powered by OpenAI API")


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""
PDF Extractor - CLI application for extracting data from PDFs using OpenAI's API
"""

import os
import re
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

def _to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[’']", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "statement"

def _parse_missing_statements(validation_output: str) -> List[str]:
    if not validation_output:
        return []
    lines = validation_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("MISSING_STATEMENTS:"):
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("NONE") or rest.startswith("[NONE"):
                return []
            # Expected: "LIST - heading1, heading2" possibly continuing until "STATEMENTS:"
            collected = [rest]
            for j in range(i + 1, len(lines)):
                nxt = lines[j].strip()
                if not nxt:
                    break
                if nxt.startswith("STATEMENTS:") or nxt.startswith("OVERALL:"):
                    break
                collected.append(nxt)
            blob = " ".join(collected).strip()
            blob = re.sub(r"^\[?LIST\]?\s*-\s*", "", blob, flags=re.IGNORECASE)
            blob = re.sub(r"^LIST\s*-\s*", "", blob, flags=re.IGNORECASE)
            parts = re.split(r"\s*[,;]\s*|\s+\|\s+", blob)
            headings = [p.strip(" -\t") for p in parts if p.strip(" -\t")]
            return headings
    return []

def _parse_incomplete_statement_keys(validation_output: str) -> List[str]:
    """
    Extract statement keys that the structure validator flags as incomplete.

    We look for an "If INVALID" / "specific corrections needed" section and parse
    bullet lines like: "- changes_in_shareholders_equity — ..."
    """
    if not validation_output:
        return []
    keys: List[str] = []
    in_corrections = False
    for raw in validation_output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("if invalid"):
            in_corrections = True
            continue
        if line.startswith("MISSING_STATEMENTS:"):
            # Not the section we need; keep scanning
            continue
        if in_corrections:
            m = re.match(r"^-\s+([a-z0-9_]+)\s*[—:-]\s*", line)
            if m:
                keys.append(m.group(1))
    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out

def _parse_required_rows_for_statement(validation_output: str, statement_key: str, max_rows: int = 30) -> List[str]:
    """
    Pull a short list of row labels that must be present as separate lines for a given statement.
    This is best-effort parsing of the bullet list under the statement's correction block.
    """
    if not validation_output or not statement_key:
        return []

    lines = validation_output.splitlines()
    start_idx = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if re.match(rf"^-\s+{re.escape(statement_key)}\s*[—:-]\s*", line):
            start_idx = i
            break
    if start_idx is None:
        return []

    required: List[str] = []
    for j in range(start_idx + 1, len(lines)):
        line = lines[j].rstrip()
        if not line.strip():
            continue
        if line.strip().startswith("- "):
            break
        m = re.match(r"^\s*-\s+(.+)$", line)
        if not m:
            continue
        label = m.group(1).strip()
        # Skip meta bullets
        if label.lower().startswith("missing"):
            continue
        required.append(label)
        if len(required) >= max_rows:
            break
    return required

def _note_key(note_id: str) -> str:
    return f"note_{note_id.replace('.', '_')}"

def _note_needs_retry(note_obj: dict) -> bool:
    """
    Heuristic: retry notes where the model produced empty tables or explicitly said it skipped/deferred extraction.
    """
    if not isinstance(note_obj, dict):
        return True
    tables = note_obj.get("tables")
    if not isinstance(tables, list) or len(tables) == 0:
        # If there are truly no tables, this is valid, but we can't know; retry once if notes imply skipping.
        notes = note_obj.get("extraction_notes", [])
        if not isinstance(notes, list):
            return True
        text = " ".join([n for n in notes if isinstance(n, str)]).lower()
        skip_markers = ["deferred", "omitted", "skipped", "not extracted", "not parsed", "empty tables", "constraints", "time constraints", "not completed"]
        return any(m in text for m in skip_markers) or len(tables) == 0
    return False


@app.command()
def extract(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Custom extraction prompt"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Use a prompt template (table, financial, custom)"),
    statement_type: str = typer.Option("balance sheet", "--statement-type", "-s", help="Type(s) of financial statement. Examples: 'balance sheet', 'balance sheet and cash flow', 'all'"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output filename (without extension)"),
    format: List[str] = typer.Option(["json"], "--format", "-f", help="Output format(s): json, txt, csv, md"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate extracted data"),
    validate_schema: bool = typer.Option(True, "--validate-schema/--no-validate-schema", help="Validate JSON schema compliance"),
    enforce_json: bool = typer.Option(True, "--enforce-json/--no-enforce-json", help="Enforce strict JSON-only output"),
    extract_notes: bool = typer.Option(False, "--extract-notes", help="Extract referenced notes from financial statements"),
    model: str = typer.Option("gpt-5-mini", "--model", "-m", help="Model to use (gpt-5-mini)"),
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
            extraction_prompt = prompt_templates.financial_statement_prompt(
                statement_type=statement_type
            )
            console.print(f"[dim]Using financial statement template for: {statement_type}[/dim]")
            if enforce_json:
                console.print(f"[dim]JSON-only mode enabled[/dim]")
        else:
            # Default prompt
            extraction_prompt = "Extract all relevant information from this PDF in a structured format (JSON preferred)."
            console.print(f"[dim]Using default extraction prompt[/dim]")

        # Extract data
        console.print(f"\n[bold]Step 3: Extracting data with {model}[/bold]")
        result = extractor.multi_page_extract(
            file_id,
            extraction_prompt,
            enforce_json=enforce_json
        )

        if not result["success"]:
            console.print(f"[red]Extraction failed: {result['error']}[/red]")
            raise typer.Exit(1)

        extracted_data = result["output_text"]

        # Normalize common JSON drift (financial template) before any further steps
        if template == "financial":
            try:
                import json
                parsed_data = json.loads(extracted_data)
                validator = ExtractionValidator(api_key=api_key, model=model)
                normalized_data, fixes = validator.normalize_financial_json(parsed_data)
                if fixes:
                    extracted_data = json.dumps(normalized_data, indent=2)
                    console.print(f"[dim]Normalized extracted JSON ({len(fixes)} fixes)[/dim]")
            except json.JSONDecodeError:
                pass

        # Extract notes if requested (only for financial template)
        if extract_notes and template == "financial":
            console.print(f"\n[bold]Step 3c: Extracting Referenced Notes[/bold]")
            try:
                import json
                from src.notes_extractor import NotesExtractor

                parsed_data = json.loads(extracted_data)

                # Collect note references from extracted statements
                note_refs = NotesExtractor.collect_note_references(parsed_data)

                if note_refs:
                    console.print(f"[dim]Found {len(note_refs)} unique note references: {', '.join(note_refs)}[/dim]")

                    merged_notes: dict = {}

                    def _extract_notes_batch(batch_ids: List[str]) -> dict:
                        batch_prompt = prompt_templates.notes_extraction_prompt(batch_ids)
                        notes_result = extractor.multi_page_extract(
                            file_id,
                            batch_prompt,
                            enforce_json=enforce_json
                        )
                        if not notes_result["success"]:
                            raise RuntimeError(notes_result.get("error") or "Unknown notes extraction error")
                        return json.loads(notes_result["output_text"])

                    def _extract_notes_batched(note_ids: List[str], batch_size: int = 5) -> None:
                        """
                        Extract notes in batches, splitting batches on failures/timeouts.
                        This is more reliable than attempting all notes in one request.
                        """
                        if not note_ids:
                            return
                        batch_size = max(1, int(batch_size))
                        queue: List[List[str]] = [note_ids[i:i + batch_size] for i in range(0, len(note_ids), batch_size)]
                        while queue:
                            batch = queue.pop(0)
                            try:
                                merged_notes.update(_extract_notes_batch(batch))
                            except Exception as e:
                                if len(batch) > 1:
                                    mid = max(1, len(batch) // 2)
                                    left, right = batch[:mid], batch[mid:]
                                    queue.insert(0, right)
                                    queue.insert(0, left)
                                    console.print(f"[yellow]Notes batch failed ({', '.join(batch)}): {e}. Splitting batch...[/yellow]")
                                else:
                                    console.print(f"[yellow]Notes batch failed ({', '.join(batch)}): {e}[/yellow]")

                    # Always batch notes extraction (avoid all-notes single request timeouts)
                    batch_size = 5  # up to 5 notes per request (as requested)
                    console.print(f"[dim]Extracting notes in batches (up to {batch_size} notes/request)...[/dim]")
                    _extract_notes_batched(note_refs, batch_size=batch_size)

                    # Retry: batch only the notes that came back empty/omitted
                    max_retry_rounds = 3
                    for round_idx in range(max_retry_rounds):
                        remaining: List[str] = []
                        for note_id in note_refs:
                            key = _note_key(note_id)
                            obj = merged_notes.get(key)
                            if obj is None:
                                remaining.append(note_id)
                                continue
                            if _note_needs_retry(obj):
                                remaining.append(note_id)

                        if not remaining:
                            break

                        console.print(f"[yellow]Retrying {len(remaining)} notes in batches (round {round_idx + 1}/{max_retry_rounds})...[/yellow]")
                        _extract_notes_batched(remaining, batch_size=batch_size)

                    if merged_notes:
                        # Merge notes into main data (same level as statements)
                        parsed_data.update(merged_notes)
                        extracted_data = json.dumps(parsed_data, indent=2)
                        console.print(f"[green]✓ Extracted {len(merged_notes)} notes[/green]")

                        # Summarize any notes that still have empty/omitted tables (best-effort signal)
                        still_empty: List[str] = []
                        for note_id in note_refs:
                            key = _note_key(note_id)
                            obj = merged_notes.get(key)
                            if obj is None:
                                continue
                            if _note_needs_retry(obj):
                                still_empty.append(note_id)
                        if still_empty:
                            console.print(f"[yellow]⚠ Notes with empty/omitted tables after retries: {', '.join(still_empty[:20])}[/yellow]")
                            if len(still_empty) > 20:
                                console.print(f"[yellow]  ...and {len(still_empty) - 20} more[/yellow]")

                        # Normalize notes using same pipeline
                        normalized_data, notes_fixes = validator.normalize_financial_json(parsed_data)
                        if notes_fixes:
                            extracted_data = json.dumps(normalized_data, indent=2)
                            console.print(f"[dim]Normalized notes JSON ({len(notes_fixes)} additional fixes)[/dim]")

                        # If schema issues remain for notes, retry those notes once (batched)
                        try:
                            schema_check = validator.validate_financial_json(normalized_data, verbose=False)
                            if not schema_check.get("is_valid"):
                                to_repair: List[str] = []
                                for err in schema_check.get("errors") or []:
                                    if not isinstance(err, str) or ":" not in err:
                                        continue
                                    prefix = err.split(":", 1)[0].strip()
                                    if prefix.startswith("note_"):
                                        to_repair.append(prefix.replace("note_", "").replace("_", "."))
                                # De-duplicate while preserving order
                                seen = set()
                                repair_ids: List[str] = []
                                for nid in to_repair:
                                    if nid in seen:
                                        continue
                                    seen.add(nid)
                                    repair_ids.append(nid)

                                if repair_ids:
                                    console.print(f"[yellow]Schema issues remain in {len(repair_ids)} notes; retrying those notes (batched)...[/yellow]")
                                    merged_notes.update({})
                                    _extract_notes_batched(repair_ids, batch_size=batch_size)
                                    # Merge repaired notes and normalize again
                                    parsed_data.update(merged_notes)
                                    normalized_data, more_fixes = validator.normalize_financial_json(parsed_data)
                                    extracted_data = json.dumps(normalized_data, indent=2)
                                    if more_fixes:
                                        console.print(f"[dim]Normalized retried notes JSON ({len(more_fixes)} fixes)[/dim]")
                        except Exception:
                            pass
                    else:
                        console.print("[yellow]Notes extraction produced no note objects[/yellow]")
                else:
                    console.print("[dim]No note references found in extracted statements[/dim]")
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Cannot extract notes (invalid JSON): {e}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Notes extraction error: {e}[/yellow]")

        # Show extraction notes if available
        console.print(f"\n[bold]Step 3b: Extraction Notes (QA Review)[/bold]")
        try:
            import json
            parsed_data = json.loads(extracted_data)
            notes_printed = False

            # Single-statement format
            if isinstance(parsed_data, dict) and parsed_data.get("extraction_notes"):
                console.print("─" * 80)
                console.print("[bold cyan]EXTRACTION NOTES:[/bold cyan]")
                console.print("─" * 80)
                for note in parsed_data["extraction_notes"]:
                    console.print(f"  [cyan]•[/cyan] {note}")
                console.print("─" * 80)
                notes_printed = True

            # Multi-statement format (statement keys at top-level)
            if isinstance(parsed_data, dict) and not notes_printed:
                for statement_key, statement_data in parsed_data.items():
                    if not isinstance(statement_data, dict):
                        continue
                    statement_notes = statement_data.get("extraction_notes")
                    if not statement_notes:
                        continue
                    console.print("─" * 80)
                    console.print(f"[bold cyan]EXTRACTION NOTES: {statement_key}[/bold cyan]")
                    console.print("─" * 80)
                    for note in statement_notes:
                        console.print(f"  [cyan]•[/cyan] {note}")
                    notes_printed = True

            if not notes_printed:
                console.print("[dim]No extraction notes logged by model[/dim]")
        except json.JSONDecodeError:
            console.print("[yellow]Cannot parse extraction notes (invalid JSON)[/yellow]")
        except Exception as e:
            console.print(f"[dim]Could not display extraction notes: {e}[/dim]")

        # Show preview
        if preview:
            formatter.display_preview(extracted_data)

        # Validate if requested
        validation_result = None
        schema_validation_result = None

        # Schema validation (for financial templates)
        if validate_schema and template == "financial":
            console.print(f"\n[bold]Step 4a: Validating JSON schema[/bold]")
            try:
                import json
                parsed_data = json.loads(extracted_data)
                validator = ExtractionValidator(api_key=api_key, model=model)
                schema_validation_result = validator.validate_financial_json(parsed_data, verbose=True)

                if not schema_validation_result["is_valid"]:
                    console.print("[yellow]⚠ Schema validation found issues - see details above[/yellow]")
            except json.JSONDecodeError:
                console.print("[red]Cannot validate schema - data is not valid JSON[/red]")

        # Content validation - section structure (for financial templates)
        structure_validation_result = None
        if validate and template == "financial":
            console.print(f"\n[bold]Step 4b: Validating section structure against PDF[/bold]")
            try:
                import json
                parsed_data = json.loads(extracted_data)
                validator = ExtractionValidator(api_key=api_key, model=model)
                structure_validation_result = validator.validate_section_structure(
                    file_id=file_id,
                    data=parsed_data,
                    statement_type=statement_type,
                    verbose=True
                )

                if not structure_validation_result["is_valid"]:
                    console.print("[red]⚠ Section structure validation found issues - see details above[/red]")

                    # Attempt targeted re-extraction of missing statements (only when --validate and template financial)
                    missing_headings = _parse_missing_statements(structure_validation_result.get("validation_output") or "")
                    if missing_headings:
                        console.print(f"\n[yellow]Attempting to re-extract missing statements ({len(missing_headings)})...[/yellow]")
                        merged = parsed_data
                        for heading in missing_headings:
                            statement_key = _to_snake_case(heading)
                            repair_prompt = prompt_templates.financial_statement_repair_prompt(
                                statement_heading=heading,
                                statement_key=statement_key
                            )
                            repair_result = extractor.extract(file_id, repair_prompt, enforce_json=True)
                            if not repair_result.get("success"):
                                console.print(f"[yellow]⚠ Could not re-extract '{heading}': {repair_result.get('error')}[/yellow]")
                                continue
                            try:
                                repair_json = json.loads(repair_result["output_text"])
                                if isinstance(repair_json, dict):
                                    merged.update(repair_json)
                            except Exception:
                                console.print(f"[yellow]⚠ Re-extraction returned invalid JSON for '{heading}'[/yellow]")

                        # Normalize again after merge
                        normalized_data, fixes = validator.normalize_financial_json(merged)
                        extracted_data = json.dumps(normalized_data, indent=2)
                        if fixes:
                            console.print(f"[dim]Normalized merged JSON ({len(fixes)} fixes)[/dim]")

                        # Re-run structure validation once after repairs
                        console.print(f"\n[bold]Step 4b (re-check): Validating section structure against PDF[/bold]")
                        structure_validation_result = validator.validate_section_structure(
                            file_id=file_id,
                            data=normalized_data,
                            statement_type=statement_type,
                            verbose=True
                        )
                    # Attempt targeted re-extraction of incomplete statements (missing/aggregated rows)
                    if not structure_validation_result["is_valid"]:
                        validation_output = structure_validation_result.get("validation_output") or ""
                        incomplete_keys = _parse_incomplete_statement_keys(validation_output)
                        if incomplete_keys:
                            console.print(f"\n[yellow]Attempting to re-extract incomplete statements ({len(incomplete_keys)})...[/yellow]")
                            merged = parsed_data
                            for statement_key in incomplete_keys:
                                current = merged.get(statement_key) if isinstance(merged, dict) else None
                                heading = None
                                if isinstance(current, dict):
                                    heading = (current.get("metadata") or {}).get("statement_type")
                                heading = heading or statement_key.replace("_", " ")

                                required_rows = _parse_required_rows_for_statement(validation_output, statement_key)
                                repair_prompt = prompt_templates.financial_statement_repair_prompt(
                                    statement_heading=str(heading),
                                    statement_key=statement_key,
                                    required_row_labels=required_rows
                                )
                                repair_result = extractor.extract(file_id, repair_prompt, enforce_json=True)
                                if not repair_result.get("success"):
                                    console.print(f"[yellow]⚠ Could not re-extract '{statement_key}': {repair_result.get('error')}[/yellow]")
                                    continue
                                try:
                                    repair_json = json.loads(repair_result["output_text"])
                                    if isinstance(repair_json, dict) and statement_key in repair_json:
                                        merged[statement_key] = repair_json[statement_key]
                                except Exception:
                                    console.print(f"[yellow]⚠ Re-extraction returned invalid JSON for '{statement_key}'[/yellow]")

                            # Normalize again after merge
                            normalized_data, fixes = validator.normalize_financial_json(merged)
                            extracted_data = json.dumps(normalized_data, indent=2)
                            if fixes:
                                console.print(f"[dim]Normalized merged JSON ({len(fixes)} fixes)[/dim]")

                            # Re-run structure validation once after repairs
                            console.print(f"\n[bold]Step 4b (re-check): Validating section structure against PDF[/bold]")
                            structure_validation_result = validator.validate_section_structure(
                                file_id=file_id,
                                data=normalized_data,
                                statement_type=statement_type,
                                verbose=True
                            )
            except json.JSONDecodeError:
                console.print("[red]Cannot validate section structure - data is not valid JSON[/red]")

        # Content validation (general re-check against PDF)
        if validate:
            console.print(f"\n[bold]Step 4c: Validating extracted data against PDF[/bold]")
            validator = ExtractionValidator(api_key=api_key, model=model)
            validation_target_data = extracted_data
            notes_keys: List[str] = []
            full_json = None
            if template == "financial":
                try:
                    import json
                    full_json = json.loads(extracted_data)
                    if isinstance(full_json, dict):
                        for k, v in full_json.items():
                            if isinstance(k, str) and k.startswith("note_"):
                                notes_keys.append(k)
                                continue
                            if isinstance(v, dict) and isinstance(v.get("tables"), list):
                                notes_keys.append(k)
                                continue
                            md = v.get("metadata") if isinstance(v, dict) else None
                            if isinstance(md, dict) and md.get("statement_type") == "note":
                                notes_keys.append(k)
                    # If notes exist, validate statements first (avoid context overflows)
                    if notes_keys and isinstance(full_json, dict):
                        statements_only = {k: v for k, v in full_json.items() if k not in set(notes_keys)}
                        validation_target_data = json.dumps(statements_only, indent=2)
                except Exception:
                    full_json = None

            validation_result = validator.validate(
                file_id=file_id,
                data_type=statement_type if template == "financial" else "data",
                extracted_data=validation_target_data
            )

            # Apply corrections if validation found issues (statements-only if notes exist)
            if not validation_result["is_valid"] and validation_result.get("validation_output"):
                console.print("\n[yellow]Attempting to apply corrections from validation...[/yellow]")
                try:
                    import json
                    data_for_correction = json.loads(validation_target_data)
                except Exception:
                    data_for_correction = None

                if isinstance(data_for_correction, dict):
                    corrected_data, corrections = validator.apply_corrections_from_validation(
                        data=data_for_correction,
                        validation_output=validation_result["validation_output"],
                        verbose=True
                    )
                else:
                    corrected_data, corrections = (None, [])

                if corrections:
                    # Update extracted_data with corrected JSON for saving and any subsequent steps
                    console.print(f"[green]✓ Applied {len(corrections)} corrections[/green]")

                    # Re-run validation to confirm corrections
                    console.print("\n[dim]Re-validating after corrections...[/dim]")
                    revalidation_result = validator.validate(
                        file_id=file_id,
                        data_type=statement_type if template == "financial" else "data",
                        extracted_data=json.dumps(corrected_data, indent=2)
                    )

                    # Display updated validation results
                    if revalidation_result["is_valid"]:
                        console.print(f"[green]✓ Validation passed after corrections (confidence: {revalidation_result.get('confidence', 'N/A')}%)[/green]")
                    else:
                        console.print(f"[yellow]⚠ Some issues remain (confidence: {revalidation_result.get('confidence', 'N/A')}%)[/yellow]")

                    # Merge corrected statements back into full JSON if notes were excluded
                    if isinstance(full_json, dict) and notes_keys:
                        for k, v in corrected_data.items():
                            full_json[k] = v
                        extracted_data = json.dumps(full_json, indent=2)
                    else:
                        extracted_data = json.dumps(corrected_data, indent=2)
                else:
                    console.print("[yellow]⚠ No corrections could be automatically applied[/yellow]")

            if not validation_result["is_valid"]:
                console.print("[yellow]Warning: Validation found potential issues[/yellow]")
                if validation_result["errors"]:
                    for error in validation_result["errors"]:
                        console.print(f"  [yellow]• {error}[/yellow]")

            # Notes validation (batched, manifest-based) to avoid context overruns
            if template == "financial" and isinstance(full_json, dict) and notes_keys:
                console.print("\n[bold]Step 4c (notes): Validating note tables structure (batched)[/bold]")

                def _chunk(items: List[str], size: int) -> List[List[str]]:
                    return [items[i:i + size] for i in range(0, len(items), size)]

                issues_by_note: dict = {}
                for batch in _chunk(sorted(notes_keys), 5):
                    subset = {k: full_json.get(k) for k in batch if k in full_json}
                    batch_result = validator.validate_notes_tables_structure(file_id=file_id, notes=subset, verbose=True)
                    for issue in batch_result.get("issues") or []:
                        if not isinstance(issue, dict):
                            continue
                        note_id = issue.get("note_id")
                        table_id = issue.get("table_id")
                        if not isinstance(note_id, str) or not note_id.strip():
                            continue
                        if not isinstance(table_id, str) or not table_id.strip():
                            continue
                        issues_by_note.setdefault(note_id.strip(), set()).add(table_id.strip())

                # Attempt targeted re-extraction of problematic tables (one repair round)
                if issues_by_note:
                    console.print(f"\n[yellow]Attempting targeted re-extraction for {len(issues_by_note)} notes with table issues...[/yellow]")
                    try:
                        import json
                        # Batch repairs (up to 5 notes/request)
                        note_ids_sorted = sorted(issues_by_note.keys(), key=lambda x: [int(p) for p in x.split('.') if p.isdigit()])
                        for i in range(0, len(note_ids_sorted), 5):
                            batch_note_ids = note_ids_sorted[i:i + 5]
                            mapping = {nid: sorted(list(issues_by_note[nid])) for nid in batch_note_ids}
                            repair_prompt = prompt_templates.notes_tables_repair_prompt(mapping)
                            repair_result = extractor.multi_page_extract(file_id, repair_prompt, enforce_json=True)
                            if not repair_result.get("success"):
                                console.print(f"[yellow]⚠ Notes repair batch failed ({', '.join(batch_note_ids)}): {repair_result.get('error')}[/yellow]")
                                continue
                            repair_json = json.loads(repair_result["output_text"])
                            if not isinstance(repair_json, dict):
                                continue

                            # Merge repaired tables into existing notes
                            for nid in batch_note_ids:
                                nk = _note_key(nid)
                                new_note = repair_json.get(nk)
                                if not isinstance(new_note, dict):
                                    continue
                                existing_note = full_json.get(nk)
                                if not isinstance(existing_note, dict):
                                    full_json[nk] = new_note
                                    continue
                                new_tables = new_note.get("tables")
                                if not isinstance(new_tables, list):
                                    continue
                                existing_tables = existing_note.get("tables")
                                if not isinstance(existing_tables, list):
                                    existing_tables = []
                                    existing_note["tables"] = existing_tables

                                by_id = {}
                                for t in existing_tables:
                                    if isinstance(t, dict) and isinstance(t.get("table_id"), str):
                                        by_id[t["table_id"]] = t
                                for t in new_tables:
                                    if not isinstance(t, dict) or not isinstance(t.get("table_id"), str):
                                        continue
                                    by_id[t["table_id"]] = t
                                # Preserve order where possible
                                merged_tables = list(by_id.values())
                                existing_note["tables"] = merged_tables
                                # Prefer refreshed note metadata/title if provided
                                if isinstance(new_note.get("metadata"), dict):
                                    existing_note["metadata"] = new_note["metadata"]
                                if isinstance(new_note.get("extraction_notes"), list):
                                    existing_note["extraction_notes"] = new_note["extraction_notes"]

                        # Normalize after repairs
                        normalized_after, fixes_after = validator.normalize_financial_json(full_json)
                        extracted_data = json.dumps(normalized_after, indent=2)
                        full_json = normalized_after
                        if fixes_after:
                            console.print(f"[dim]Normalized repaired notes JSON ({len(fixes_after)} fixes)[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]⚠ Notes repair failed: {e}[/yellow]")

        # Save output
        step_num = 5
        if validate:
            step_num += 1
        if validate_schema and template == "financial":
            step_num += 1

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
        console.print(f"  Statement type: {statement_type if template == 'financial' else 'N/A'}")
        console.print(f"  JSON enforcement: {'Enabled' if enforce_json else 'Disabled'}")
        console.print(f"  Input tokens: {result['input_tokens']}")
        console.print(f"  Output tokens: {result['output_tokens']}")

        if schema_validation_result:
            console.print(f"  Schema validation: {'✓ Passed' if schema_validation_result['is_valid'] else '⚠ Issues found'}")
            console.print(f"  Schema confidence: {schema_validation_result['confidence']}%")

        if validation_result:
            console.print(f"  Content validation: {'✓ Passed' if validation_result['is_valid'] else '⚠ Issues found'}")
            console.print(f"  Content confidence: {validation_result['confidence']}%")

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
    model: str = typer.Option("gpt-5-mini", "--model", "-m", help="Model to use"),
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

"""
Validation module for verifying extraction accuracy
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
from rich.console import Console
from .prompts import PromptTemplates

console = Console()


class ExtractionValidator:
    """Validates extracted data against source PDF"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini"
    ):
        """
        Initialize the validator

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: Model to use for validation
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.prompt_templates = PromptTemplates()

    def _is_multi_statement_format(self, data: Dict[str, Any]) -> bool:
        """
        Detect if data is in multi-statement format.
        Multi-statement format has statement types as top-level keys,
        each containing their own metadata.

        Args:
            data: Parsed JSON dictionary

        Returns:
            True if multi-statement format, False otherwise
        """
        # Any top-level object (except metadata/extraction_notes) that contains metadata counts
        for key, value in data.items():
            if key in ["metadata", "extraction_notes"]:
                continue
            if isinstance(value, dict) and "metadata" in value:
                return True

        return False

    def validate(
        self,
        file_id: str,
        data_type: str,
        extracted_data: str
    ) -> Dict[str, Any]:
        """
        Validate extracted data against the source PDF

        Args:
            file_id: OpenAI file ID of the source PDF
            data_type: Type of data extracted (e.g., "balance sheet")
            extracted_data: The data that was extracted

        Returns:
            Dictionary containing:
                - is_valid: Boolean indicating if data is valid
                - confidence: Confidence score (0-100)
                - errors: List of errors found
                - validation_output: Raw validation response
                - success: Whether validation completed successfully
        """
        console.print(f"[blue]Validating extracted {data_type}...[/blue]")

        # Generate validation prompt
        validation_prompt = self.prompt_templates.validation_prompt(
            data_type=data_type,
            extracted_data=extracted_data
        )

        try:
            # Send validation request
            response = self.client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": file_id},
                        {"type": "input_text", "text": validation_prompt}
                    ]
                }]
            )

            validation_output = response.output_text if hasattr(response, 'output_text') else str(response)

            # Parse validation response
            result = self._parse_validation_response(validation_output)
            result["validation_output"] = validation_output
            result["success"] = True

            # Display results
            if result["is_valid"]:
                console.print(f"[green]✓ Validation passed (confidence: {result['confidence']}%)[/green]")
            else:
                console.print(f"[yellow]⚠ Validation issues found (confidence: {result['confidence']}%)[/yellow]")
                if result["errors"]:
                    console.print("[yellow]Errors:[/yellow]")
                    for error in result["errors"]:
                        console.print(f"  - {error}")

            return result

        except Exception as e:
            console.print(f"[red]✗ Validation failed: {str(e)}[/red]")
            return {
                "is_valid": False,
                "confidence": 0,
                "errors": [f"Validation error: {str(e)}"],
                "validation_output": None,
                "success": False
            }

    def _parse_validation_response(self, validation_output: str) -> Dict[str, Any]:
        """
        Parse the validation response to extract structured data

        Args:
            validation_output: Raw validation response text

        Returns:
            Dictionary with parsed validation results
        """
        # Check for validation status
        if "VALIDATED" in validation_output and "All data is accurate" in validation_output:
            return {
                "is_valid": True,
                "confidence": 100,
                "errors": []
            }

        # Check for errors
        if "ERRORS FOUND" in validation_output:
            errors = self._extract_errors(validation_output)
            return {
                "is_valid": False,
                "confidence": 0,
                "errors": errors
            }

        # Check for partial validation
        if "PARTIAL" in validation_output:
            confidence = self._extract_confidence(validation_output)
            errors = self._extract_errors(validation_output)
            return {
                "is_valid": confidence >= 90,  # Consider valid if 90%+ confidence
                "confidence": confidence,
                "errors": errors
            }

        # Default: assume validation passed if no errors explicitly mentioned
        return {
            "is_valid": True,
            "confidence": 95,
            "errors": []
        }

    def _extract_errors(self, text: str) -> List[str]:
        """
        Extract error messages from validation output

        Args:
            text: Validation output text

        Returns:
            List of error messages
        """
        errors = []

        # Split by common error indicators
        lines = text.split('\n')
        in_error_section = False

        for line in lines:
            line = line.strip()

            # Detect error section start
            if "ERRORS FOUND" in line or "Issues:" in line or "Errors:" in line:
                in_error_section = True
                continue

            # Detect error section end
            if in_error_section and line.startswith("PARTIAL"):
                in_error_section = False
                continue

            # Extract error lines
            if in_error_section and line:
                # Remove bullet points and numbering
                clean_line = re.sub(r'^[-*•]\s*', '', line)
                clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                if clean_line:
                    errors.append(clean_line)

        return errors

    def _extract_confidence(self, text: str) -> int:
        """
        Extract confidence percentage from validation output

        Args:
            text: Validation output text

        Returns:
            Confidence score (0-100)
        """
        # Look for percentage patterns
        patterns = [
            r'PARTIAL:\s*(\d+)%',
            r'confidence:\s*(\d+)%',
            r'(\d+)%\s*confident',
            r'accuracy:\s*(\d+)%'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return 75  # Default confidence if not specified

    def quick_validate(
        self,
        extracted_data: str,
        data_type: str = "data"
    ) -> Dict[str, Any]:
        """
        Perform quick validation without re-analyzing the PDF

        Args:
            extracted_data: The extracted data to validate
            data_type: Type of data

        Returns:
            Dictionary with validation results (basic checks only)
        """
        console.print(f"[blue]Running quick validation checks...[/blue]")

        errors = []
        confidence = 100

        # Basic checks
        if not extracted_data or len(extracted_data.strip()) == 0:
            errors.append("Extracted data is empty")
            confidence = 0

        # Check for common extraction issues
        if "[CONTINUE]" in extracted_data:
            errors.append("Data appears incomplete (contains continuation marker)")
            confidence -= 20

        if "ERROR" in extracted_data.upper() or "FAILED" in extracted_data.upper():
            errors.append("Data contains error indicators")
            confidence -= 30

        # JSON validation if it looks like JSON
        if extracted_data.strip().startswith('{'):
            import json
            try:
                json.loads(extracted_data)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON format: {str(e)}")
                confidence -= 40

        is_valid = len(errors) == 0 and confidence >= 70

        result = {
            "is_valid": is_valid,
            "confidence": max(0, confidence),
            "errors": errors,
            "validation_output": "Quick validation (no PDF re-analysis)",
            "success": True
        }

        if is_valid:
            console.print(f"[green]✓ Quick validation passed[/green]")
        else:
            console.print(f"[yellow]⚠ Quick validation issues found[/yellow]")

        return result

    def validate_financial_json(
        self,
        data: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate that extracted financial data meets schema requirements.
        This is a structural validation that checks JSON schema compliance.
        Supports both single and multi-statement extraction formats.

        Args:
            data: Parsed JSON dictionary from extraction
            verbose: Whether to print detailed validation results

        Returns:
            Dictionary containing:
                - is_valid: Boolean indicating if schema is valid
                - errors: List of specific validation errors
                - warnings: List of non-critical issues
                - confidence: Confidence score (0-100)
        """
        # Detect if multi-statement format
        if self._is_multi_statement_format(data):
            # Multi-statement validation
            if verbose:
                console.print("[blue]Validating multi-statement extraction...[/blue]")

            overall_valid = True
            all_errors = []
            all_warnings = []

            for statement_key, statement_data in data.items():
                if verbose:
                    console.print(f"\n[dim]Validating {statement_key}...[/dim]")

                # Validate each statement separately
                result = self._validate_single_statement(statement_data, verbose=False)

                if not result['is_valid']:
                    overall_valid = False
                    all_errors.extend([f"{statement_key}: {e}" for e in result['errors']])

                all_warnings.extend([f"{statement_key}: {w}" for w in result['warnings']])

            # Calculate overall confidence
            confidence = 100 if overall_valid else max(0, 100 - len(all_errors) * 10)

            # Display results
            if verbose:
                if overall_valid:
                    console.print(f"\n[green]✓ Multi-statement validation passed (confidence: {confidence}%)[/green]")
                    if all_warnings:
                        console.print(f"[yellow]Warnings ({len(all_warnings)}):[/yellow]")
                        for warning in all_warnings:
                            console.print(f"  [yellow]• {warning}[/yellow]")
                else:
                    console.print(f"\n[red]✗ Multi-statement validation failed (confidence: {confidence}%)[/red]")
                    console.print(f"[red]Errors ({len(all_errors)}):[/red]")
                    for error in all_errors:
                        console.print(f"  [red]• {error}[/red]")
                    if all_warnings:
                        console.print(f"[yellow]Warnings ({len(all_warnings)}):[/yellow]")
                        for warning in all_warnings:
                            console.print(f"  [yellow]• {warning}[/yellow]")

            return {
                'is_valid': overall_valid,
                'errors': all_errors,
                'warnings': all_warnings,
                'confidence': confidence
            }
        else:
            # Single statement validation (use existing logic)
            return self._validate_single_statement(data, verbose)

    def normalize_financial_json(
        self,
        data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Normalize common LLM schema/shape drift for financial extraction outputs.

        This is a deterministic, non-LLM post-processor that makes outputs
        more consistent and schema-compliant without changing numeric content.
        """
        from copy import deepcopy

        normalized = deepcopy(data)
        fixes: List[str] = []

        iso_date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        def _to_snake_case(text: str) -> str:
            text = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
            return re.sub(r"[\s-]+", "_", text)

        def _keys_look_like_iso_dates(values_obj: Dict[str, Any]) -> bool:
            keys = list(values_obj.keys())
            return bool(keys) and all(isinstance(k, str) and iso_date_pattern.match(k) for k in keys)

        def _detect_matrix_table(statement_data: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
            columns = metadata.get("columns")
            if not isinstance(columns, list) or len(columns) == 0:
                return False

            # If any line-item values keys are not ISO dates, treat this statement as a matrix table.
            for array_name, value in statement_data.items():
                if array_name in ["metadata", "extraction_notes"]:
                    continue
                if not isinstance(value, list):
                    continue
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    values_obj = item.get("values")
                    if isinstance(values_obj, dict) and values_obj:
                        return not _keys_look_like_iso_dates(values_obj)
            return False

        def normalize_statement(statement_data: Dict[str, Any], statement_key: str) -> Dict[str, Any]:
            if not isinstance(statement_data, dict):
                return statement_data

            metadata = statement_data.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                statement_data["metadata"] = metadata
                fixes.append(f"{statement_key}: added missing metadata object")

            # Move top-level column definitions into metadata.columns
            if "columns" in statement_data and isinstance(statement_data["columns"], list):
                if "columns" not in metadata or not isinstance(metadata.get("columns"), list) or len(metadata.get("columns") or []) == 0:
                    metadata["columns"] = statement_data["columns"]
                    fixes.append(f"{statement_key}: moved top-level columns -> metadata.columns")
                del statement_data["columns"]
                fixes.append(f"{statement_key}: removed forbidden top-level columns")

            if "metadata_columns" in statement_data and isinstance(statement_data["metadata_columns"], list):
                if "columns" not in metadata or not isinstance(metadata.get("columns"), list) or len(metadata.get("columns") or []) == 0:
                    # Some model variants use {header,label}; normalize to {key,label}
                    cols = []
                    for col in statement_data["metadata_columns"]:
                        if isinstance(col, dict):
                            cols.append({
                                "key": col.get("key") or col.get("snake_case") or col.get("id") or "",
                                "label": col.get("label") or col.get("header") or ""
                            })
                    metadata["columns"] = cols
                    fixes.append(f"{statement_key}: moved metadata_columns -> metadata.columns")
                del statement_data["metadata_columns"]
                fixes.append(f"{statement_key}: removed forbidden metadata_columns")

            # Normalize metadata.columns entries (accept drift: header/name -> label; derive missing key/label)
            if isinstance(metadata.get("columns"), list):
                normalized_cols: List[Dict[str, Any]] = []
                for idx, col in enumerate(metadata["columns"]):
                    if not isinstance(col, dict):
                        continue
                    label = col.get("label") or col.get("header") or col.get("name") or ""
                    key = col.get("key") or ""
                    if not isinstance(label, str):
                        label = ""
                    if not isinstance(key, str):
                        key = ""
                    label = label.strip()
                    key = key.strip()
                    if not key and label:
                        key = _to_snake_case(label)
                        fixes.append(f"{statement_key}: derived metadata.columns[{idx}].key from label")
                    if not label and key:
                        derived = key.replace("_", " ").strip()
                        label = derived[:1].upper() + derived[1:] if derived else ""
                        fixes.append(f"{statement_key}: derived metadata.columns[{idx}].label from key")
                    normalized_cols.append({"key": key, "label": label})
                if normalized_cols:
                    metadata["columns"] = normalized_cols
                    fixes.append(f"{statement_key}: normalized metadata.columns entries to {{key,label}}")

            # Decide axis robustly (matrix vs time-series) based on values keys
            is_matrix = _detect_matrix_table(statement_data, metadata)
            if is_matrix:
                if metadata.get("periods") != []:
                    metadata["periods"] = []
                    fixes.append(f"{statement_key}: forced metadata.periods = [] for matrix table")
            else:
                # If time-series, avoid axis ambiguity by removing metadata.columns when periods are present
                if isinstance(metadata.get("periods"), list) and len(metadata.get("periods") or []) > 0 and isinstance(metadata.get("columns"), list) and len(metadata.get("columns") or []) > 0:
                    del metadata["columns"]
                    fixes.append(f"{statement_key}: removed metadata.columns for time-series table (axis ambiguity)")

            # Matrix tables: normalize line-array naming (rows -> lines)
            if is_matrix and "lines" not in statement_data and isinstance(statement_data.get("rows"), list):
                statement_data["lines"] = statement_data["rows"]
                del statement_data["rows"]
                fixes.append(f"{statement_key}: renamed rows -> lines for matrix table")

            # Build column label->key mapping for matrix tables
            column_label_to_key: Dict[str, str] = {}
            column_keys = set()
            if isinstance(metadata.get("columns"), list):
                for col in metadata["columns"]:
                    if not isinstance(col, dict):
                        continue
                    key = col.get("key")
                    label = col.get("label")
                    if isinstance(key, str) and key:
                        column_keys.add(key)
                    if isinstance(label, str) and isinstance(key, str) and key:
                        column_label_to_key[label.strip().lower()] = key

            def normalize_line_item(item: Dict[str, Any], idx: int, array_name: str) -> Dict[str, Any]:
                if not isinstance(item, dict):
                    return item

                # Remove known drift keys
                if "line_kind" in item:
                    item.pop("line_kind", None)
                    fixes.append(f"{statement_key}: removed unsupported field {array_name}[{idx}].line_kind")

                # For non-position rows, row_as_of should be null (dates belong in row_period)
                if item.get("row_kind") in ["movement", "subtotal"] and isinstance(item.get("row_as_of"), str):
                    item["row_as_of"] = None
                    fixes.append(f"{statement_key}: set row_as_of=null for non-position {array_name}[{idx}]")

                # Ensure required fields
                if "line_number" not in item or not isinstance(item.get("line_number"), int):
                    item["line_number"] = idx + 1
                    fixes.append(f"{statement_key}: set missing/invalid line_number for {array_name}[{idx}]")

                if "label" not in item or not isinstance(item.get("label"), str) or not item.get("label"):
                    fallback = item.get("row_description") if isinstance(item.get("row_description"), str) and item.get("row_description") else f"unknown_line_{item['line_number']}"
                    item["label"] = str(fallback)
                    fixes.append(f"{statement_key}: set missing label for {array_name}[{idx}]")

                if "level" not in item or not isinstance(item.get("level"), int):
                    item["level"] = 0
                    fixes.append(f"{statement_key}: set missing level for {array_name}[{idx}]")

                # Position rows are not totals unless explicitly labeled as total/subtotal
                if isinstance(item.get("is_total"), bool) and item.get("row_kind") == "position":
                    label_lower = str(item.get("label", "")).lower()
                    if "total" not in label_lower and "sub-total" not in label_lower and "subtotal" not in label_lower:
                        if item["is_total"] is True:
                            item["is_total"] = False
                            fixes.append(f"{statement_key}: corrected is_total=false for position {array_name}[{idx}]")

                if "is_total" not in item or not isinstance(item.get("is_total"), bool):
                    label_lower = str(item.get("label", "")).lower()
                    row_kind = item.get("row_kind")
                    is_total = False
                    if row_kind == "subtotal":
                        is_total = True
                    elif "total" in label_lower or "sub-total" in label_lower or "subtotal" in label_lower:
                        is_total = True
                    item["is_total"] = is_total
                    fixes.append(f"{statement_key}: set missing is_total for {array_name}[{idx}]")

                if "notes_reference" not in item or not isinstance(item.get("notes_reference"), list):
                    item["notes_reference"] = []
                    fixes.append(f"{statement_key}: set missing notes_reference for {array_name}[{idx}]")

                # Normalize row_as_of type (must be ISO date string or null)
                if "row_as_of" in item and isinstance(item.get("row_as_of"), dict):
                    # Some outputs incorrectly copy row_period into row_as_of
                    if item.get("row_period") is None and set(item["row_as_of"].keys()) >= {"start", "end"}:
                        item["row_period"] = item["row_as_of"]
                        fixes.append(f"{statement_key}: moved dict row_as_of -> row_period for {array_name}[{idx}]")
                    item["row_as_of"] = None
                    fixes.append(f"{statement_key}: coerced dict row_as_of -> null for {array_name}[{idx}]")

                # Values normalization
                if "values" not in item or not isinstance(item.get("values"), dict):
                    item["values"] = {}
                    fixes.append(f"{statement_key}: set missing values object for {array_name}[{idx}]")

                # Matrix: remap values keys from human labels to metadata.columns keys if possible
                if is_matrix and item.get("values"):
                    values_obj = item["values"]
                    remapped: Dict[str, Any] = {}
                    did_remap = False
                    for k, v in values_obj.items():
                        if k in column_keys:
                            remapped[k] = v
                            continue
                        if isinstance(k, str):
                            mapped = column_label_to_key.get(k.strip().lower())
                            if mapped:
                                remapped[mapped] = v
                                did_remap = True
                            else:
                                remapped[k] = v
                        else:
                            remapped[str(k)] = v
                    if did_remap:
                        item["values"] = remapped
                        fixes.append(f"{statement_key}: remapped values keys to metadata.columns keys for {array_name}[{idx}]")

                return item

            # Normalize all arrays of line items (any list of dicts with a values object)
            for array_name, value in list(statement_data.items()):
                if array_name in ["metadata", "extraction_notes"]:
                    continue
                if not isinstance(value, list) or len(value) == 0:
                    continue
                if not all(isinstance(x, dict) for x in value):
                    continue
                if not any(isinstance(x.get("values"), dict) for x in value if isinstance(x, dict)):
                    continue

                statement_data[array_name] = [
                    normalize_line_item(item, idx, array_name) for idx, item in enumerate(value)
                ]

            return statement_data

        if self._is_multi_statement_format(normalized):
            for statement_key, statement_data in list(normalized.items()):
                if isinstance(statement_data, dict) and "metadata" in statement_data:
                    normalized[statement_key] = normalize_statement(statement_data, statement_key)
        else:
            normalized = normalize_statement(normalized, "root")

        return normalized, fixes

    def _validate_single_statement(
        self,
        data: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a single statement (extracted from existing validate_financial_json logic).
        This is the original validation logic, now reusable for both single and multi-statement modes.

        Args:
            data: Parsed JSON dictionary for a single statement
            verbose: Whether to print detailed validation results

        Returns:
            Dictionary containing validation results
        """
        errors = []
        warnings = []

        if verbose:
            console.print(f"[blue]Validating financial JSON schema...[/blue]")

        # === METADATA VALIDATION ===
        if "metadata" not in data:
            errors.append("Missing 'metadata' section")
        else:
            metadata = data["metadata"]

            # Required metadata fields
            required_metadata = [
                "company_name",
                "statement_type",
                "currency",
                "units_multiplier"
            ]

            for field in required_metadata:
                if field not in metadata:
                    errors.append(f"Missing metadata.{field}")
                elif metadata[field] is None or metadata[field] == "":
                    warnings.append(f"metadata.{field} is empty")

            # Recommended metadata fields
            if "reporting_date" not in metadata:
                warnings.append("Missing metadata.reporting_date (recommended)")
            if "periods" not in metadata:
                warnings.append("Missing metadata.periods (recommended)")
            if "dates_covered" not in metadata:
                warnings.append("Missing metadata.dates_covered (recommended)")
            elif isinstance(metadata["periods"], list) and len(metadata["periods"]) > 0:
                # Handle both period formats: objects or simple strings
                first_period = metadata["periods"][0]
                if isinstance(first_period, dict):
                    # New format: [{"label": "...", "iso_date": "...", "context": "..."}]
                    for i, period in enumerate(metadata["periods"]):
                        if not isinstance(period, dict):
                            errors.append(f"metadata.periods[{i}] should be object with 'label', 'iso_date', and optional 'context' fields")
                        elif "label" not in period:
                            errors.append(f"metadata.periods[{i}] missing 'label' field")
                        elif "iso_date" not in period:
                            errors.append(f"metadata.periods[{i}] missing 'iso_date' field")
                        else:
                            # Validate ISO date format
                            iso_date = period.get("iso_date")
                            if iso_date and not re.match(r'^\d{4}-\d{2}-\d{2}$', iso_date):
                                errors.append(f"metadata.periods[{i}].iso_date '{iso_date}' is not in valid ISO format (YYYY-MM-DD)")
                        # 'context' field is optional
                # else: Old format with simple strings ["2024", "2023"] - still valid for backward compatibility

            # Optional component columns for matrix-style tables (e.g., equity reconciliation)
            if "columns" in metadata:
                if not isinstance(metadata["columns"], list):
                    errors.append(f"metadata.columns must be an array, got {type(metadata['columns']).__name__}")
                else:
                    for i, col in enumerate(metadata["columns"]):
                        if not isinstance(col, dict):
                            errors.append(f"metadata.columns[{i}] must be an object with 'key' and 'label'")
                            continue
                        if "key" not in col or not isinstance(col.get("key"), str) or not col.get("key"):
                            errors.append(f"metadata.columns[{i}].key is required and must be a non-empty string")
                        if "label" not in col or not isinstance(col.get("label"), str) or not col.get("label"):
                            errors.append(f"metadata.columns[{i}].label is required and must be a non-empty string")

            # Validate units_multiplier value
            if "units_multiplier" in metadata:
                valid_multipliers = [1, 1000, 1000000, 1000000000]
                if metadata["units_multiplier"] not in valid_multipliers:
                    warnings.append(
                        f"Unusual units_multiplier: {metadata['units_multiplier']} "
                        f"(expected one of: {valid_multipliers})"
                    )

            # Warn if both periods and columns are populated (axis ambiguity)
            if isinstance(metadata.get("periods"), list) and len(metadata.get("periods")) > 0 and isinstance(metadata.get("columns"), list) and len(metadata.get("columns")) > 0:
                warnings.append("metadata.periods and metadata.columns are both populated; expected exactly one axis (periods for time-series OR columns for matrix tables)")

        # === STRUCTURE VALIDATION ===
        # Generic validation - check that there are data arrays (not specific field names)

        # Hard-stop on wrapper patterns that break reconstruction
        if "sections" in data:
            errors.append("Found 'sections' wrapper; expected top-level arrays of line items (no sections/line_items nesting)")
        if "metadata_columns" in data:
            errors.append("Found 'metadata_columns'; columns must be stored in metadata.columns (not as a top-level array)")
        if "columns" in data:
            errors.append("Found top-level 'columns'; columns must be stored in metadata.columns (not as a top-level array)")

        # Get all top-level arrays (excluding metadata and extraction_notes)
        top_level_arrays = [
            k for k in data.keys()
            if k not in ["metadata", "extraction_notes", "sections", "metadata_columns", "columns"] and isinstance(data[k], list)
        ]

        if len(top_level_arrays) == 0:
            errors.append("No data arrays found (expected at least one section with line items)")
        else:
            # Validate each array found
            for array_name in top_level_arrays:
                if not isinstance(data[array_name], list):
                    errors.append(f"'{array_name}' must be an array, got {type(data[array_name]).__name__}")
                elif len(data[array_name]) == 0:
                    warnings.append(f"Array '{array_name}' is empty")

        # === LINE ITEM VALIDATION ===
        def validate_line_items(items: List[Dict], section_name: str):
            """Helper to validate an array of line items"""
            for i, item in enumerate(items):
                item_ref = f"{section_name}[{i}]"

                # Check required fields
                required_fields = ["line_number", "label", "level", "is_total", "values"]
                for field in required_fields:
                    if field not in item:
                        errors.append(f"{item_ref}: Missing required field '{field}'")

                # Validate data types
                if "line_number" in item and not isinstance(item["line_number"], int):
                    errors.append(f"{item_ref}: 'line_number' must be integer, got {type(item['line_number']).__name__}")

                if "label" in item and not isinstance(item["label"], str):
                    errors.append(f"{item_ref}: 'label' must be string, got {type(item['label']).__name__}")

                if "level" in item and not isinstance(item["level"], int):
                    errors.append(f"{item_ref}: 'level' must be integer, got {type(item['level']).__name__}")

                if "is_total" in item and not isinstance(item["is_total"], bool):
                    errors.append(f"{item_ref}: 'is_total' must be boolean, got {type(item['is_total']).__name__}")

                if "notes_reference" in item:
                    if isinstance(item["notes_reference"], str):
                        warnings.append(f"{item_ref}: 'notes_reference' should be array of strings, got string")
                    elif item["notes_reference"] is not None and not isinstance(item["notes_reference"], list):
                        errors.append(f"{item_ref}: 'notes_reference' must be array or null, got {type(item['notes_reference']).__name__}")
                    elif isinstance(item["notes_reference"], list):
                        for note in item["notes_reference"]:
                            if not isinstance(note, str):
                                errors.append(f"{item_ref}: 'notes_reference' entries must be strings, got {type(note).__name__}")

                if "values" in item:
                    # Values should now be an object (dict) with period keys, not an array
                    if not isinstance(item["values"], dict):
                        errors.append(f"{item_ref}: 'values' must be object with period keys, got {type(item['values']).__name__}")
                    else:
                        # Validate that all values are numeric (or null)
                        for period, val in item["values"].items():
                            if val is not None and not isinstance(val, (int, float)):
                                errors.append(
                                    f"{item_ref}.values[\"{period}\"]: Value must be number or null, "
                                    f"got {type(val).__name__}: {val}"
                                )

                        # Check that value keys match the declared axis (periods vs component columns)
                        if "metadata" in data:
                            metadata_obj = data.get("metadata", {})
                            metadata_periods = metadata_obj.get("periods")
                            metadata_columns = metadata_obj.get("columns")
                            actual_keys = set(item["values"].keys())

                            iso_date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
                            all_keys_are_iso_dates = all(iso_date_pattern.match(k) for k in actual_keys)

                            # Axis B: component/measures as columns (matrix tables)
                            if isinstance(metadata_columns, list) and len(metadata_columns) > 0 and not all_keys_are_iso_dates:
                                expected_keys = set(
                                    c.get("key") for c in metadata_columns
                                    if isinstance(c, dict) and c.get("key")
                                )
                                if expected_keys and not actual_keys.issubset(expected_keys):
                                    extra = sorted(actual_keys - expected_keys)
                                    errors.append(
                                        f"{item_ref}: Value keys contain unknown column keys {extra} (not in metadata.columns)"
                                    )
                                if isinstance(metadata_periods, list) and len(metadata_periods) > 0:
                                    warnings.append(f"{item_ref}: Detected matrix-style value keys but metadata.periods is populated; expected metadata.periods to be [] for matrix tables")

                            # Axis A: time periods as columns
                            elif isinstance(metadata_periods, list) and len(metadata_periods) > 0:
                                if isinstance(metadata_periods[0], dict):
                                    expected_keys = set(
                                        p.get("iso_date") for p in metadata_periods
                                        if isinstance(p, dict) and p.get("iso_date")
                                    )
                                else:
                                    expected_keys = set(metadata_periods)

                                # Validate that all actual keys are ISO dates (YYYY-MM-DD format)
                                for key in actual_keys:
                                    if not iso_date_pattern.match(key):
                                        errors.append(
                                            f"{item_ref}: Period key '{key}' is not in ISO date format (YYYY-MM-DD)"
                                        )

                                if expected_keys != actual_keys:
                                    warnings.append(
                                        f"{item_ref}: Period keys {actual_keys} don't match metadata.periods iso_date values {expected_keys}"
                                    )

        # Validate all line item sections dynamically (no hardcoded field names)
        for array_name in top_level_arrays:
            if isinstance(data[array_name], list):
                validate_line_items(data[array_name], array_name)

        # === CALCULATE CONFIDENCE ===
        # Each error reduces confidence by 10%, each warning by 5%
        confidence = 100 - (len(errors) * 10) - (len(warnings) * 5)
        confidence = max(0, min(100, confidence))

        is_valid = len(errors) == 0

        # === DISPLAY RESULTS ===
        if verbose:
            if is_valid:
                console.print(f"[green]✓ Schema validation passed (confidence: {confidence}%)[/green]")
                if warnings:
                    console.print(f"[yellow]Warnings ({len(warnings)}):[/yellow]")
                    for warning in warnings:
                        console.print(f"  [yellow]• {warning}[/yellow]")
            else:
                console.print(f"[red]✗ Schema validation failed (confidence: {confidence}%)[/red]")
                console.print(f"[red]Errors ({len(errors)}):[/red]")
                for error in errors:
                    console.print(f"  [red]• {error}[/red]")
                if warnings:
                    console.print(f"[yellow]Warnings ({len(warnings)}):[/yellow]")
                    for warning in warnings:
                        console.print(f"  [yellow]• {warning}[/yellow]")

        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "errors": errors,
            "warnings": warnings
        }

    def validate_section_structure(
        self,
        file_id: str,
        data: Dict[str, Any],
        statement_type: str = "financial statement",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate that sections are correctly identified and labeled against the PDF.
        This performs rigorous content validation by re-reading the PDF.
        Supports both single and multi-statement formats.

        Args:
            file_id: OpenAI file ID of the source PDF
            data: Parsed JSON dictionary from extraction
            statement_type: Type of financial statement
            verbose: Whether to print detailed validation results

        Returns:
            Dictionary containing:
                - is_valid: Boolean indicating if content structure is correct
                - errors: List of specific structural errors
                - warnings: List of non-critical issues
                - confidence: Confidence score (0-100)
                - validation_output: Raw LLM validation response
        """
        def _collect_notes(statement_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
            notes = statement_data.get("extraction_notes", []) if isinstance(statement_data, dict) else []
            if not isinstance(notes, list):
                return [], []
            ambiguous_keywords = ["ambiguous", "unclear", "uncertain", "assumption", "assumed", "confidence", "estimate"]
            ambiguous_notes = [
                note for note in notes
                if isinstance(note, str) and any(k in note.lower() for k in ambiguous_keywords)
            ]
            return notes, ambiguous_notes

        # Detect if multi-statement format
        if self._is_multi_statement_format(data):
            if verbose:
                console.print(f"[blue]Validating multi-statement section structure against PDF (single request)...[/blue]")

            # Build combined summary for all statements
            statement_summaries = []
            for statement_key, statement_data in data.items():
                if not isinstance(statement_data, dict):
                    continue

                extracted_sections = [
                    k for k in statement_data.keys()
                    if k not in ["metadata", "extraction_notes"] and isinstance(statement_data[k], list)
                ]

                section_summary = f"Statement '{statement_key}': {len(extracted_sections)} sections\n"
                for i, section_name in enumerate(extracted_sections, 1):
                    section_data = statement_data[section_name]
                    line_count = len(section_data)
                    first_label = section_data[0].get("label", "Unknown") if section_data else "Empty"
                    last_label = section_data[-1].get("label", "Unknown") if section_data else "Empty"
                    section_summary += f"  {i}. {section_name} ({line_count} line items)\n"
                    section_summary += f"     - First item: {first_label}\n"
                    section_summary += f"     - Last item: {last_label}\n"

                notes, ambiguous_notes = _collect_notes(statement_data)
                if notes:
                    section_summary += "  Extraction notes:\n"
                    for note in notes[:6]:
                        section_summary += f"  - {note}\n"
                if ambiguous_notes:
                    section_summary += "  Ambiguities (focus here):\n"
                    for note in ambiguous_notes[:6]:
                        section_summary += f"  - {note}\n"

                statement_summaries.append(section_summary)

            validation_prompt = f"""You are validating multi-statement extraction structure against the source PDF.

TASK: Validate ALL extracted statements together against the PDF.
FOCUS: Prioritize sections or areas mentioned in extraction_notes, especially ambiguities.
Also verify overall section count, names, boundaries, and missing/extra sections.

STRICT COMPLETENESS RULE:
- If ANY statement has missing sections/rows/columns (missing != NONE) OR any YES/NO check is NO, then OVERALL MUST BE INVALID.
- Do NOT mark OVERALL as VALID if you list anything missing.
- Additionally, if the PDF contains a primary statement that is NOT PRESENT in the extracted JSON at all, OVERALL MUST BE INVALID.

EXTRACTED STRUCTURE:
{chr(10).join(statement_summaries)}

RESPOND IN THIS FORMAT:

OVERALL: [VALID/INVALID] - [Overall assessment]
MISSING_STATEMENTS: [NONE|LIST] - [If LIST, name the missing statement headings]
STATEMENTS:
1. <statement_key>: Section count [YES/NO], names [YES/NO], boundaries [YES/NO], missing [NONE/LIST], extra [NONE/LIST]
... (one line per statement)

If INVALID, list specific corrections needed.
"""

            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": file_id},
                            {"type": "input_text", "text": validation_prompt}
                        ]
                    }]
                )

                validation_output = response.output_text if hasattr(response, 'output_text') else str(response)

                is_valid = ("OVERALL: VALID" in validation_output or "OVERALL: [VALID]" in validation_output)
                # If the model lists missing content, treat as invalid even if it mistakenly says OVERALL: VALID.
                if "missing LIST" in validation_output or "FOUND MISSING" in validation_output:
                    is_valid = False
                if "MISSING_STATEMENTS:" in validation_output and "MISSING_STATEMENTS: NONE" not in validation_output:
                    is_valid = False
                errors = []
                warnings = []

                if not is_valid:
                    if "OVERALL: INVALID" in validation_output:
                        parts = validation_output.split("OVERALL: INVALID")
                        if len(parts) > 1:
                            reason = parts[1].strip().split('\n')[0]
                            errors.append(f"Structure validation failed: {reason}")
                    if "missing" in validation_output.lower():
                        warnings.append("Some sections may be missing from extraction")
                    if "extra" in validation_output.lower():
                        errors.append("Extra sections created that shouldn't exist")
                    if "MISSING_STATEMENTS:" in validation_output and "MISSING_STATEMENTS: NONE" not in validation_output:
                        errors.append("One or more statements appear in the PDF but were not extracted at all")

                confidence = 100 if is_valid else max(0, 100 - (len(errors) * 15) - (len(warnings) * 5))

                if verbose:
                    if is_valid:
                        console.print(f"\n[green]✓ Multi-statement section structure validation passed (confidence: {confidence}%)[/green]")
                    else:
                        console.print(f"\n[red]✗ Multi-statement section structure validation failed (confidence: {confidence}%)[/red]")
                        if errors:
                            console.print(f"[red]Errors ({len(errors)}):[/red]")
                            for error in errors:
                                console.print(f"  [red]• {error}[/red]")
                        if warnings:
                            console.print(f"[yellow]Warnings ({len(warnings)}):[/yellow]")
                            for warning in warnings:
                                console.print(f"  [yellow]• {warning}[/yellow]")
                    console.print(f"\n[dim]Detailed validation response:[/dim]\n[dim]{validation_output}[/dim]")

                return {
                    'is_valid': is_valid,
                    'errors': errors,
                    'warnings': warnings,
                    'confidence': confidence,
                    'validation_output': validation_output
                }
            except Exception as e:
                console.print(f"[red]✗ Validation failed: {str(e)}[/red]")
                return {
                    "is_valid": False,
                    "confidence": 0,
                    "errors": [f"Validation error: {str(e)}"],
                    "warnings": [],
                    "validation_output": None
                }
        else:
            # Single statement validation
            return self._validate_single_statement_structure(
                file_id, data, statement_type, verbose
            )

    def _validate_single_statement_structure(
        self,
        file_id: str,
        data: Dict[str, Any],
        statement_type: str = "financial statement",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate section structure for a single statement.
        This is the extracted logic from validate_section_structure().
        """
        if verbose:
            console.print(f"[blue]Validating section structure against PDF...[/blue]")

        errors = []
        warnings = []

        # Get all dynamic sections from the extracted data
        extracted_sections = [
            k for k in data.keys()
            if k not in ["metadata", "extraction_notes"] and isinstance(data[k], list)
        ]

        if not extracted_sections:
            errors.append("No sections found in extracted data")
            return {
                "is_valid": False,
                "confidence": 0,
                "errors": errors,
                "warnings": warnings,
                "validation_output": None
            }

        # Build section summary for validation
        section_summary = f"The extraction identified {len(extracted_sections)} sections:\n\n"
        for i, section_name in enumerate(extracted_sections, 1):
            section_data = data[section_name]
            line_count = len(section_data)

            # Get first and last line labels to show range
            first_label = section_data[0].get("label", "Unknown") if section_data else "Empty"
            last_label = section_data[-1].get("label", "Unknown") if section_data else "Empty"

            section_summary += f"{i}. Section '{section_name}' ({line_count} line items)\n"
            section_summary += f"   - First item: {first_label}\n"
            section_summary += f"   - Last item: {last_label}\n\n"

        # Add extraction notes if available
        if "extraction_notes" in data and data["extraction_notes"]:
            section_summary += "Extraction notes:\n"
            for note in data["extraction_notes"][:5]:  # Limit to first 5 notes
                section_summary += f"- {note}\n"
            section_summary += "\n"

        # Create validation prompt
        validation_prompt = f"""You are validating a financial statement extraction against the source PDF.

TASK: Verify that sections are correctly identified and structured.

IMPORTANT CONTEXT - SCOPE OF EXTRACTION:
- The extraction was tasked to extract ONLY: {statement_type}
- The PDF may contain other financial statements (income statement, balance sheet, cash flow, notes, etc.)
- You should ONLY validate whether the {statement_type} sections are correct
- DO NOT flag as errors if other statements in the PDF were not extracted
- Focus your validation on: "Are the {statement_type} sections extracted correctly?"

EXTRACTED STRUCTURE:
{section_summary}

VALIDATION CRITERIA (answer YES or NO for each):

1. SECTION COUNT: Is the number of sections ({len(extracted_sections)}) correct FOR THE {statement_type.upper()}?
   - Look at the PDF and find the {statement_type.upper()} sections
   - Count how many MAJOR sections/tables belong to the {statement_type}
   - IGNORE sections from other statements (they should not have been extracted)
   - Only count sections with BOLD HEADINGS or CLEAR VISUAL SEPARATORS within the {statement_type}

2. SECTION NAMES: Are the section names accurate to the {statement_type} headers in the PDF?
   - Check each section name against the actual {statement_type} headers
   - Section names should match the semantic meaning of the headers

3. SECTION BOUNDARIES: Are items correctly grouped into sections within the {statement_type}?
   - Check if the first/last items of each section match the actual PDF structure
   - Verify that items haven't been incorrectly split or merged

4. MISSING SECTIONS: Are there any {statement_type} sections that weren't extracted?
   - Look ONLY for {statement_type} sections - ignore other statements
   - Check if any bold headers within the {statement_type} should have been separate sections

5. EXTRA SECTIONS: Are there any sections that don't belong to the {statement_type}?
   - Check if sections from OTHER statements were incorrectly included
   - Check if any sections were created from content that should be nested within another section
   - Common error: Creating separate sections for summary/total lines that belong at the end of a table

RESPOND IN THIS FORMAT:

1. Section count: [YES/NO] - [Brief explanation]
2. Section names: [YES/NO] - [Brief explanation]
3. Section boundaries: [YES/NO] - [Brief explanation]
4. Missing sections: [NO MISSING/FOUND MISSING] - [List any missing or say "None"]
5. Extra sections: [NO EXTRA/FOUND EXTRA] - [List any extra or say "None"]

OVERALL: [VALID/INVALID] - [Overall assessment]

If INVALID, list specific corrections needed.
"""

        try:
            # Send validation request
            response = self.client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": file_id},
                        {"type": "input_text", "text": validation_prompt}
                    ]
                }]
            )

            validation_output = response.output_text if hasattr(response, 'output_text') else str(response)

            # Parse validation response
            is_valid = "OVERALL: VALID" in validation_output or "OVERALL: [VALID]" in validation_output

            # Extract errors from the response
            if not is_valid:
                # Look for "INVALID" reason
                if "OVERALL: INVALID" in validation_output:
                    # Extract text after "OVERALL: INVALID"
                    parts = validation_output.split("OVERALL: INVALID")
                    if len(parts) > 1:
                        reason = parts[1].strip().split('\n')[0]
                        errors.append(f"Structure validation failed: {reason}")

                # Look for specific issues
                if "Section count: NO" in validation_output:
                    errors.append("Incorrect number of sections detected")

                if "Section names: NO" in validation_output:
                    errors.append("Section names don't match PDF headers")

                if "Section boundaries: NO" in validation_output:
                    errors.append("Items are incorrectly grouped into sections")

                if "FOUND MISSING" in validation_output:
                    warnings.append("Some sections may be missing from extraction")

                if "FOUND EXTRA" in validation_output:
                    errors.append("Extra sections created that shouldn't exist")

            # Calculate confidence based on validation response
            if is_valid:
                confidence = 100
            else:
                confidence = max(0, 100 - (len(errors) * 15) - (len(warnings) * 5))

            # Display results
            if verbose:
                if is_valid:
                    console.print(f"[green]✓ Section structure validation passed[/green]")
                else:
                    console.print(f"[red]✗ Section structure validation failed[/red]")
                    if errors:
                        console.print(f"[red]Errors ({len(errors)}):[/red]")
                        for error in errors:
                            console.print(f"  [red]• {error}[/red]")
                    if warnings:
                        console.print(f"[yellow]Warnings ({len(warnings)}):[/yellow]")
                        for warning in warnings:
                            console.print(f"  [yellow]• {warning}[/yellow]")

                # Always show the detailed validation output for review
                console.print(f"\n[dim]Detailed validation response:[/dim]")
                console.print(f"[dim]{validation_output}[/dim]\n")

            return {
                "is_valid": is_valid,
                "confidence": confidence,
                "errors": errors,
                "warnings": warnings,
                "validation_output": validation_output
            }

        except Exception as e:
            error_msg = f"Section structure validation error: {str(e)}"
            if verbose:
                console.print(f"[red]✗ {error_msg}[/red]")

            return {
                "is_valid": False,
                "confidence": 0,
                "errors": [error_msg],
                "warnings": [],
                "validation_output": None
            }

    def apply_corrections_from_validation(
        self,
        data: Dict[str, Any],
        validation_output: str,
        verbose: bool = True
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Parse validation output and apply corrections to the data.

        Args:
            data: Extracted JSON data
            validation_output: Raw validation output from LLM
            verbose: Whether to print correction details

        Returns:
            Tuple of (corrected_data, list_of_corrections_made)
        """
        import json
        from copy import deepcopy

        corrections = []
        corrected_data = deepcopy(data)

        # Parse validation output for specific correction instructions
        # Look for patterns like:
        # - "Earnings per ordinary share" ... "2024: 4.38 (not 4,380,000)"
        # - Line item values with incorrect multiplier application

        lines = validation_output.split('\n')
        current_statement = None

        for i, line in enumerate(lines):
            # Detect statement context (e.g., "1) Income statement EPS unit handling:")
            if 'Income statement' in line or 'Balance sheet' in line or 'Cash flow' in line:
                # Extract statement type
                if 'Income statement' in line:
                    current_statement = 'income_statement'
                elif 'Balance sheet' in line:
                    current_statement = 'balance_sheet'
                elif 'Cash flow' in line:
                    current_statement = 'cash_flow'

            # Look for correction patterns like "2024: 4.38 (not 4,380,000)"
            if 'not ' in line and ':' in line:
                try:
                    # Parse the correction
                    # Example: "- 2024: 4.38 (not 4,380,000)"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        period_part = parts[0].strip('- ').strip()
                        value_part = parts[1].split('(not')[0].strip()

                        correct_value = float(value_part.replace(',', ''))

                        # Look for the line item label in previous lines
                        label = None
                        for j in range(max(0, i-5), i):
                            if 'earnings per' in lines[j].lower() or 'diluted' in lines[j].lower() or '"' in lines[j]:
                                # Extract label
                                label_match = lines[j].strip('- "').strip('"').strip()
                                if label_match and len(label_match) > 3:
                                    label = label_match
                                    break

                        if label and current_statement:
                            # Apply correction
                            correction_made = self._apply_single_correction(
                                corrected_data,
                                current_statement,
                                label,
                                period_part,
                                correct_value
                            )

                            if correction_made:
                                corrections.append(
                                    f"Corrected {current_statement}.{label}[{period_part}]: {correct_value}"
                                )
                except (ValueError, IndexError):
                    # Skip malformed correction lines
                    continue

        if verbose and corrections:
            console.print(f"\n[yellow]Applied {len(corrections)} corrections from validation:[/yellow]")
            for corr in corrections:
                console.print(f"  [yellow]• {corr}[/yellow]")

        return corrected_data, corrections

    def _apply_single_correction(
        self,
        data: Dict[str, Any],
        statement_key: str,
        label: str,
        period: str,
        correct_value: float
    ) -> bool:
        """
        Apply a single correction to the data.
        Returns True if correction was applied, False otherwise.
        """
        if statement_key not in data:
            return False

        statement_data = data[statement_key]

        # Search through all sections for the line item with matching label
        for section_name, section_data in statement_data.items():
            if section_name in ['metadata', 'extraction_notes']:
                continue

            if not isinstance(section_data, list):
                continue

            for item in section_data:
                if isinstance(item, dict) and 'label' in item:
                    # Fuzzy match label (handles case differences)
                    if label.lower() in item['label'].lower() or item['label'].lower() in label.lower():
                        # Found the item, update its value for the period
                        if 'values' in item and isinstance(item['values'], dict):
                            # Find matching period key
                            for period_key in item['values'].keys():
                                if period in period_key or period_key in period:
                                    item['values'][period_key] = correct_value
                                    return True

        return False

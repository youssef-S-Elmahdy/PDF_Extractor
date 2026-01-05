"""
Validation module for verifying extraction accuracy
"""

import re
from typing import Dict, Any, Optional, List
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

            # Validate units_multiplier value
            if "units_multiplier" in metadata:
                valid_multipliers = [1, 1000, 1000000, 1000000000]
                if metadata["units_multiplier"] not in valid_multipliers:
                    warnings.append(
                        f"Unusual units_multiplier: {metadata['units_multiplier']} "
                        f"(expected one of: {valid_multipliers})"
                    )

        # === STRUCTURE VALIDATION ===
        # Generic validation - check that there are data arrays (not specific field names)

        # Get all top-level arrays (excluding metadata and extraction_notes)
        top_level_arrays = [
            k for k in data.keys()
            if k not in ["metadata", "extraction_notes"] and isinstance(data[k], list)
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

                        # Check that period keys match metadata.periods and are valid ISO dates
                        if "metadata" in data and "periods" in data["metadata"]:
                            # Extract expected period keys based on format
                            metadata_periods = data["metadata"]["periods"]
                            if metadata_periods and isinstance(metadata_periods[0], dict):
                                # New format: extract "iso_date" values
                                expected_periods = set(p.get("iso_date") for p in metadata_periods if "iso_date" in p)
                            else:
                                # Old format: use strings directly (backward compatibility)
                                expected_periods = set(metadata_periods)

                            actual_periods = set(item["values"].keys())

                            # Validate that all actual period keys are ISO dates (YYYY-MM-DD format)
                            iso_date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
                            for period_key in actual_periods:
                                if not iso_date_pattern.match(period_key):
                                    errors.append(
                                        f"{item_ref}: Period key '{period_key}' is not in ISO date format (YYYY-MM-DD)"
                                    )

                            if expected_periods != actual_periods:
                                warnings.append(
                                    f"{item_ref}: Period keys {actual_periods} don't match metadata.periods iso_date values {expected_periods}"
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

EXTRACTED STRUCTURE:
{section_summary}

VALIDATION CRITERIA (answer YES or NO for each):

1. SECTION COUNT: Is the number of sections ({len(extracted_sections)}) correct?
   - Look at the PDF and count how many MAJOR sections/tables there are
   - Only count sections with BOLD HEADINGS or CLEAR VISUAL SEPARATORS
   - Ignore subsections within tables (they should be nested, not separate)

2. SECTION NAMES: Are the section names accurate to what's in the PDF?
   - Check each section name against the actual headers in the document
   - Section names should match the semantic meaning of the headers

3. SECTION BOUNDARIES: Are items correctly grouped into sections?
   - Check if the first/last items of each section match the actual PDF structure
   - Verify that items haven't been incorrectly split or merged

4. MISSING SECTIONS: Are there any major sections in the PDF that weren't extracted?
   - Look for any bold section headers that should have been separate sections

5. EXTRA SECTIONS: Are there any sections in the extraction that shouldn't exist?
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

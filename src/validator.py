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
        model: str = "gpt-5.2"
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

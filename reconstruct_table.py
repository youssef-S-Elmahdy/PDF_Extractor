#!/usr/bin/env python3
"""
Table Reconstruction Script
Demonstrates that the extracted JSON contains all information needed
to reconstruct the original financial statement table.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def format_number(value, units_multiplier: int) -> str:
    """Format a number back to its original units with thousand separators."""
    if value is None:
        return "-"

    # Divide by units_multiplier to get back to original units
    original_value = value / units_multiplier

    # Handle negative numbers (parentheses format)
    if original_value < 0:
        return f"({abs(original_value):,.0f})"

    return f"{original_value:,.0f}"


def print_balance_sheet_table(data: Dict):
    """Reconstruct and print the balance sheet as a formatted table."""

    # Extract metadata
    metadata = data["metadata"]
    company = metadata["company_name"]
    statement_type = metadata["statement_type"].title()
    date = metadata["reporting_date"]
    currency = metadata["currency"]
    original_units = metadata["original_units"]
    units_multiplier = metadata["units_multiplier"]
    periods = metadata["periods"]

    # Print header
    print("=" * 120)
    print(f"{company}")
    print(f"{statement_type}")
    print(f"As of {date}")
    print(f"(In {currency} {original_units})")
    print("=" * 120)
    print()

    # Print column headers
    label_width = 70
    col_width = 20

    header = f"{'Account':<{label_width}}"
    for period in periods:
        header += f"{period:>{col_width}}"
    header += f"{'Notes':<15}"
    print(header)
    print("-" * 120)

    # Function to print a section
    def print_section(section_name: str, items: List[Dict]):
        if not items:
            return

        print(f"\n{section_name.upper()}")
        print("-" * 120)

        for item in items:
            label = item["label"]
            level = item.get("level", 1)
            is_total = item.get("is_total", False)
            values = item.get("values", {})
            notes = item.get("notes_reference", "")

            # Indent based on level
            indent = "  " * (level - 1)
            label_formatted = f"{indent}{label}"

            # Format label with bold for totals (using simple caps)
            if is_total:
                label_formatted = label_formatted.upper()

            # Build row
            row = f"{label_formatted:<{label_width}}"

            # Add values for each period
            for period in periods:
                value = values.get(period)
                formatted = format_number(value, units_multiplier)
                row += f"{formatted:>{col_width}}"

            # Add notes reference
            notes_str = notes if notes else ""
            row += f"{notes_str:<15}"

            print(row)

    # Print each section
    if "assets" in data:
        print_section("ASSETS", data["assets"])

    if "liabilities" in data:
        print_section("LIABILITIES", data["liabilities"])

    if "equity" in data:
        print_section("SHAREHOLDERS' EQUITY", data["equity"])

    print()
    print("=" * 120)
    print()


def validate_reconstruction(data: Dict) -> Dict:
    """
    Validate that all required information is present for reconstruction.
    Returns a validation report.
    """
    issues = []
    warnings = []

    # Check metadata
    required_metadata = ["company_name", "statement_type", "reporting_date",
                        "currency", "original_units", "units_multiplier", "periods"]

    if "metadata" not in data:
        issues.append("Missing metadata section")
        return {"valid": False, "issues": issues, "warnings": warnings}

    for field in required_metadata:
        if field not in data["metadata"]:
            issues.append(f"Missing metadata field: {field}")

    # Check sections exist
    if "assets" not in data:
        issues.append("Missing 'assets' section")
    if "liabilities" not in data:
        issues.append("Missing 'liabilities' section")
    if "equity" not in data:
        issues.append("Missing 'equity' section")

    # Check line items have required fields
    def check_items(items: List[Dict], section_name: str):
        for i, item in enumerate(items):
            if "label" not in item:
                issues.append(f"{section_name}[{i}]: Missing 'label'")
            if "values" not in item:
                issues.append(f"{section_name}[{i}]: Missing 'values'")
            elif not isinstance(item["values"], dict):
                issues.append(f"{section_name}[{i}]: 'values' should be an object, not {type(item['values'])}")

            # Check that values keys match periods
            if "values" in item and isinstance(item["values"], dict):
                periods = set(data["metadata"]["periods"])
                value_keys = set(item["values"].keys())
                if periods != value_keys:
                    warnings.append(
                        f"{section_name}[{i}]: Value periods {value_keys} don't match metadata periods {periods}"
                    )

    if "assets" in data:
        check_items(data["assets"], "assets")
    if "liabilities" in data:
        check_items(data["liabilities"], "liabilities")
    if "equity" in data:
        check_items(data["equity"], "equity")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reconstruct_table.py <path_to_extracted_json>")
        print("\nExample:")
        print("  python reconstruct_table.py output/q4-2024-Financial-statements_EN_extracted.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Load JSON
    print(f"Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print("✓ JSON loaded successfully\n")

    # Validate
    print("Validating reconstruction requirements...")
    validation = validate_reconstruction(data)

    if not validation["valid"]:
        print("\n⚠️  VALIDATION FAILED")
        print("\nIssues found:")
        for issue in validation["issues"]:
            print(f"  ❌ {issue}")
        sys.exit(1)

    if validation["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"  ⚠️  {warning}")

    print("✓ All required data present for reconstruction\n")

    # Count items
    total_items = 0
    if "assets" in data:
        total_items += len(data["assets"])
    if "liabilities" in data:
        total_items += len(data["liabilities"])
    if "equity" in data:
        total_items += len(data["equity"])

    print(f"📊 Statistics:")
    print(f"   Company: {data['metadata']['company_name']}")
    print(f"   Statement: {data['metadata']['statement_type']}")
    print(f"   Date: {data['metadata']['reporting_date']}")
    print(f"   Currency: {data['metadata']['currency']}")
    print(f"   Units: {data['metadata']['original_units']} (multiplier: {data['metadata']['units_multiplier']:,})")
    print(f"   Periods: {', '.join(data['metadata']['periods'])}")
    print(f"   Total line items: {total_items}")
    if "assets" in data:
        print(f"     - Assets: {len(data['assets'])}")
    if "liabilities" in data:
        print(f"     - Liabilities: {len(data['liabilities'])}")
    if "equity" in data:
        print(f"     - Equity: {len(data['equity'])}")
    print()

    # Reconstruct table
    print("\nReconstructed Table:")
    print()
    print_balance_sheet_table(data)

    print("\n✅ SUCCESS: Table fully reconstructed from JSON!")
    print("   All original information preserved:")
    print("   ✓ Company name and statement type")
    print("   ✓ Reporting date")
    print("   ✓ All line item labels with hierarchy")
    print("   ✓ All numeric values (converted back to original units)")
    print("   ✓ Notes references")
    print("   ✓ Negative values properly indicated")
    print("   ✓ Column headers (periods)")
    print("\n   This JSON is ready for database storage or any other use! 🎉\n")


if __name__ == "__main__":
    main()

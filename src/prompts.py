"""
Prompt templates for PDF extraction and validation
"""

from typing import Optional, List


class PromptTemplates:
    """Collection of prompt templates for PDF extraction tasks"""

    @staticmethod
    def _parse_statement_types(statement_type: str) -> List[str]:
        """
        Parse statement type string into individual types.

        Examples:
            "balance sheet" -> ["balance sheet"]
            "balance sheet and cash flow" -> ["balance sheet", "cash flow"]
            "balance sheet, income statement, cash flow" -> ["balance sheet", "income statement", "cash flow"]
            "all" -> ["all"]  # Special case for auto-detection

        Args:
            statement_type: Statement type string from user

        Returns:
            List of individual statement types
        """
        statement_type_lower = statement_type.lower().strip()

        # Special case: "all" means auto-detect all statements
        if statement_type_lower == "all":
            return ["all"]

        # Try multiple delimiters
        for delimiter in [' and ', ',', ' & ', ';']:
            if delimiter in statement_type_lower:
                return [s.strip() for s in statement_type.split(delimiter) if s.strip()]

        # Single statement type
        return [statement_type]

    @staticmethod
    def extraction_prompt(
        task_description: str,
        output_format: str = "JSON",
        include_requirements: Optional[str] = None
    ) -> str:
        """
        Generate an extraction prompt based on task description

        Args:
            task_description: What to extract from the PDF (e.g., "balance sheet table")
            output_format: Desired output format (JSON, Markdown, CSV)
            include_requirements: Additional requirements (e.g., "all columns, rows, headers")

        Returns:
            Formatted extraction prompt
        """
        prompt = f"Extract {task_description} from this PDF.\n\n"

        if include_requirements:
            prompt += f"Include {include_requirements}.\n\n"

        prompt += f"Format the output as {output_format}.\n\n"
        prompt += "If data spans multiple pages, combine all entries.\n"
        prompt += "If tables are unclear or borderless, use context to identify structure.\n"
        prompt += "Preserve all numeric values exactly as they appear.\n"
        prompt += "Include all headers, labels, and metadata."

        return prompt

    @staticmethod
    def table_extraction_prompt(
        table_name: str,
        output_format: str = "JSON"
    ) -> str:
        """
        Generate a specialized prompt for extracting tables

        Args:
            table_name: Name or description of the table to extract
            output_format: Desired output format

        Returns:
            Formatted table extraction prompt
        """
        prompt = f"Extract the complete {table_name} from this PDF.\n\n"
        prompt += "Requirements:\n"
        prompt += "- Include ALL rows and columns\n"
        prompt += "- Preserve all column headers exactly as shown\n"
        prompt += "- Preserve all row labels/categories\n"
        prompt += "- Include all numeric values with correct precision\n"
        prompt += "- Include all subtotals and totals\n"
        prompt += "- If the table spans multiple pages, combine all sections\n"
        prompt += "- Maintain the hierarchical structure (parent/child rows)\n\n"
        prompt += f"Format the output as {output_format} with clear structure.\n"
        prompt += "If values are in thousands/millions, note the unit."

        return prompt

    @staticmethod
    def validation_prompt(
        data_type: str,
        extracted_data: str
    ) -> str:
        """
        Generate a validation prompt to verify extracted data

        Args:
            data_type: Type of data that was extracted (e.g., "balance sheet")
            extracted_data: The data that was previously extracted

        Returns:
            Formatted validation prompt
        """
        prompt = f"Review the previously extracted {data_type} against the PDF.\n\n"
        prompt += "Verify accuracy of:\n"
        prompt += "- All numeric values (check each number against the source)\n"
        prompt += "- Column and row headers (exact wording)\n"
        prompt += "- Completeness of entries (no missing rows or columns)\n"
        prompt += "- Subtotals and totals (verify calculations if applicable)\n"
        prompt += "- Units and currency symbols\n\n"
        prompt += "Previously extracted data:\n"
        prompt += f"{extracted_data}\n\n"
        prompt += "Provide one of the following responses:\n"
        prompt += "1. If accurate: 'VALIDATED: All data is accurate.'\n"
        prompt += "2. If errors found: 'ERRORS FOUND:' followed by specific corrections needed.\n"
        prompt += "3. If partially correct: 'PARTIAL: [confidence %]' with details of issues."

        return prompt

    @staticmethod
    def financial_statement_prompt(
        statement_type: str = "balance sheet"
    ) -> str:
        """
        Production-grade financial statement extraction prompt.
        Engineered for 100% JSON compliance with zero narrative text.
        Supports both single and multi-statement extraction.

        Args:
            statement_type: Type of financial statement. Can be:
                - Single: "balance sheet", "income statement", "cash flow"
                - Multiple: "balance sheet and cash flow", "balance sheet, income statement"
                - Auto-detect: "all"

        Returns:
            Formatted financial statement extraction prompt with strict JSON enforcement
        """
        # Parse statement types to detect single vs multi-statement mode
        statement_types = PromptTemplates._parse_statement_types(statement_type)

        # Determine if multi-statement or single statement mode
        is_multi_statement = len(statement_types) > 1 or statement_types[0] == "all"

        # Build prompt based on mode
        if is_multi_statement:
            return PromptTemplates._build_multi_statement_prompt(statement_types)
        else:
            return PromptTemplates._build_single_statement_prompt(statement_types[0])

    @staticmethod
    def _build_single_statement_prompt(statement_type: str) -> str:
        """Build prompt for single statement extraction (current behavior)"""
        # CRITICAL: Start with absolute prohibition on narrative
        prompt = "You are a financial data extraction API. You MUST return ONLY valid JSON.\n"
        prompt += "DO NOT include any explanatory text, commentary, notes, or markdown formatting.\n"
        prompt += "DO NOT wrap the JSON in code blocks (no ```json).\n"
        prompt += "Return raw JSON only, starting with { and ending with }.\n\n"

        # TASK DEFINITION WITH SCOPE
        prompt += f"TASK: Extract ONLY the {statement_type} from this financial document.\n\n"
        prompt += "SCOPE AND CONTEXT AWARENESS - CRITICAL:\n"
        prompt += f"- You are extracting: {statement_type}\n"
        prompt += f"- This PDF may contain multiple financial statements (income statement, balance sheet, cash flow, notes, etc.)\n"
        prompt += f"- Your job: Extract ONLY the sections that belong to the {statement_type}\n"
        prompt += f"- IGNORE all other statements in the PDF - they are not relevant to this extraction task\n"
        prompt += f"- Use your judgment and vision to identify which sections are part of the {statement_type}\n"
        prompt += f"- The {statement_type} may span multiple pages - combine all its sections\n"
        prompt += f"- Do NOT extract sections from other statements just because they exist in the PDF\n\n"
        prompt += "HOW TO IDENTIFY THE CORRECT STATEMENT:\n"
        prompt += f"- Look for headings that indicate '{statement_type.upper()}' or similar\n"
        prompt += f"- Use visual cues (heading style, section breaks) to determine statement boundaries\n"
        prompt += f"- If uncertain whether a section belongs to {statement_type}, use context clues:\n"
        prompt += f"  * Does it fit the typical structure of a {statement_type}?\n"
        prompt += f"  * Is it visually grouped with other {statement_type} sections?\n"
        prompt += f"  * Does the heading or content clearly indicate it's part of {statement_type}?\n"
        prompt += f"- When in doubt, default to NOT extracting unless you're confident it belongs to {statement_type}\n\n"

        # Add all the common rules
        prompt += PromptTemplates._build_common_extraction_rules()

        # Single-statement specific JSON schema
        prompt += "REQUIRED JSON OUTPUT SCHEMA - FOLLOW EXACTLY:\n\n"
        prompt += "{\n"
        prompt += '  "metadata": {\n'
        prompt += '    "company_name": "Auto-detected from document",\n'
        prompt += '    "statement_type": "' + statement_type + '",\n'
        prompt += '    "reporting_date": "YYYY-MM-DD or period",\n'
        prompt += '    "currency": "Auto-detected (EUR, USD, etc.)",\n'
        prompt += '    "original_units": "Auto-detected (thousands, millions, etc.)",\n'
        prompt += '    "units_multiplier": 1000000,\n'
        prompt += '    "dates_covered": "YYYY-MM-DD to YYYY-MM-DD or YYYY-MM-DD, YYYY-MM-DD",\n'
        prompt += '    "periods": [\n'
        prompt += '      {"label": "Exact text from column header", "iso_date": "YYYY-MM-DD", "context": "explanation if ambiguous"}\n'
        prompt += '    ]\n'
        prompt += '  },\n'
        prompt += '  "extraction_notes": [\n'
        prompt += '    "Document structure: Detected X sections/tables: [list their names]",\n'
        prompt += '    "Array naming decisions: [explain how you named each array]",\n'
        prompt += '    "Any other structural decisions, ambiguities, or assumptions"\n'
        prompt += '  ],\n'
        prompt += '  "<dynamic_section_name_1>": [\n'
        prompt += '    {\n'
        prompt += '      "line_number": 1,\n'
        prompt += '      "label": "Exact text from document",\n'
        prompt += '      "level": 1,\n'
        prompt += '      "is_total": false,\n'
        prompt += '      "values": {"period1": number, "period2": number},\n'
        prompt += '      "notes_reference": "Note X.X or null"\n'
        prompt += '    },\n'
        prompt += '    ...\n'
        prompt += '  ],\n'
        prompt += '  "<dynamic_section_name_2>": [\n'
        prompt += '    ...\n'
        prompt += '  ],\n'
        prompt += '  "...additional sections as needed...": [...]\n'
        prompt += '}\n\n'
        prompt += "IMPORTANT: The section names above (<dynamic_section_name_1>, etc.) are PLACEHOLDERS.\n"
        prompt += "You MUST replace them with actual names based on what you find in the document.\n"
        prompt += "DO NOT use these placeholder names in your actual output.\n"
        prompt += "The number of sections is also dynamic - create as many as the document has.\n\n"

        # FINAL ENFORCEMENT
        prompt += "CRITICAL FINAL REMINDERS:\n"
        prompt += "✓ Return ONLY the JSON object described above - no other text whatsoever\n"
        prompt += "✓ All numeric values MUST be numbers (not strings like \"1234\")\n"
        prompt += "✓ Store values as an OBJECT with ISO DATE keys (YYYY-MM-DD format)\n"
        prompt += "✓ DO NOT use arrays for values: [value1, value2] is WRONG\n"
        prompt += "✓ The keys in the values object must EXACTLY match the 'iso_date' field from metadata.periods\n"
        prompt += "✓ DETECT section structure dynamically from document - do NOT use hardcoded array names\n"
        prompt += "✓ NAME arrays based on actual section headers found in the document (use snake_case)\n"
        prompt += "✓ PRESERVE exact labels from document (no inference or interpretation)\n"
        prompt += "✓ SEPARATE note references from labels (e.g., 'Asset Note 3.1' → label: 'Asset', notes_reference: 'Note 3.1')\n"
        prompt += "✓ EXTRACT period labels EXACTLY as shown in column headers\n"
        prompt += "✓ CAPTURE ALL rows including final 'Total' row at bottom of tables\n"
        prompt += "✓ MAINTAIN hierarchy with proper level indicators (1, 2, 3, 4)\n"
        prompt += "✓ LOG extraction decisions in extraction_notes array\n"
        prompt += "✓ MULTIPLY all values by units_multiplier BEFORE storing them\n"
        prompt += "✓ CONVERT (parentheses) to negative numbers, not strings\n"
        prompt += "✓ REMOVE all currency symbols (€, $) from values\n"
        prompt += "✓ REMOVE all thousand separators (commas) from values\n"
        prompt += "✓ REMOVE decorative dots (....) from labels and values\n"
        prompt += "✓ Your response MUST start with { and end with }\n"
        prompt += "✓ DO NOT add any explanations, notes, or commentary outside the JSON\n\n"
        prompt += "BEGIN EXTRACTION NOW. Output only the JSON:\n"

        return prompt

    @staticmethod
    def _build_multi_statement_prompt(statement_types: List[str]) -> str:
        """Build prompt for multi-statement extraction"""
        # CRITICAL: Start with absolute prohibition on narrative
        prompt = "You are a financial data extraction API. You MUST return ONLY valid JSON.\n"
        prompt += "DO NOT include any explanatory text, commentary, notes, or markdown formatting.\n"
        prompt += "DO NOT wrap the JSON in code blocks (no ```json).\n"
        prompt += "Return raw JSON only, starting with { and ending with }.\n\n"

        # MULTI-STATEMENT MODE
        if statement_types[0] == "all":
            prompt += "TASK: AUTO-DETECT and extract ALL financial statements from this PDF.\n\n"
        else:
            prompt += f"TASK: Extract ONLY these financial statements: {', '.join(statement_types)}\n\n"

        # ====================================================================
        # ABSOLUTE NON-NEGOTIABLE REQUIREMENTS - READ THIS FIRST
        # ====================================================================
        prompt += "=" * 80 + "\n"
        prompt += "MANDATORY REQUIREMENTS - EVERY LINE ITEM MUST HAVE ALL THESE FIELDS:\n"
        prompt += "=" * 80 + "\n\n"

        prompt += "❌ FAILURE TO INCLUDE ANY OF THESE FIELDS WILL CAUSE EXTRACTION FAILURE ❌\n\n"

        prompt += "EVERY SINGLE LINE ITEM IN EVERY SECTION MUST INCLUDE:\n"
        prompt += "  1. line_number (number) - Sequential number starting at 1 within each section\n"
        prompt += "  2. label (string) - The exact text label from the document\n"
        prompt += "  3. level (number) - Hierarchy level: 1 (section header), 2 (main item), 3 (sub-item), 4 (sub-sub-item)\n"
        prompt += "  4. is_total (boolean) - true if this is a total/subtotal line, false otherwise\n"
        prompt += "  5. notes_reference (string or null) - Reference to notes (e.g., 'Note 3.1') or null if none\n"
        prompt += "  6. values (object) - Object with ISO date keys (YYYY-MM-DD) and numeric values\n\n"

        prompt += "METADATA OBJECT FOR EACH STATEMENT MUST INCLUDE:\n"
        prompt += "  1. company_name (string)\n"
        prompt += "  2. statement_type (string)\n"
        prompt += "  3. reporting_date (string in YYYY-MM-DD format)\n"
        prompt += "  4. currency (string) - e.g., 'EUR', 'USD'\n"
        prompt += "  5. original_units (string) - e.g., 'EUR m', 'in thousands'\n"
        prompt += "  6. units_multiplier (number) - REQUIRED! 1000000 for millions, 1000 for thousands, 1 for base units\n"
        prompt += "  7. periods (array) - Array of period objects with label, iso_date, and context\n\n"

        prompt += "UNIT CONVERSION - ABSOLUTELY CRITICAL:\n"
        prompt += "  ⚠️  INTELLIGENT MULTIPLIER APPLICATION ⚠️\n\n"

        prompt += "  When a units_multiplier is specified, apply it INTELLIGENTLY.\n"
        prompt += "  Use your understanding of the data context to determine which values\n"
        prompt += "  represent monetary amounts in the specified units (multiply these) vs.\n"
        prompt += "  values already expressed per-unit, as ratios, or as percentages (do NOT multiply these).\n\n"

        prompt += "  Store non-monetary values (per-unit metrics, ratios, percentages, counts)\n"
        prompt += "  EXACTLY as they appear in the PDF without applying the multiplier.\n\n"

        prompt += "=" * 80 + "\n\n"

        # Now continue with the rest of the instructions
        if statement_types[0] == "all":
            prompt += "MULTI-STATEMENT EXTRACTION MODE - AUTO-DETECT:\n"
            prompt += "- Scan the entire PDF and identify all financial statement types present\n"
            prompt += "- Common statement types: balance sheet, income statement, cash flow, notes\n"
            prompt += "- Create SEPARATE top-level objects for EACH statement type found\n"
        else:
            prompt += "MULTI-STATEMENT EXTRACTION MODE - SPECIFIC STATEMENTS:\n"
            prompt += f"- You are extracting: {', '.join(statement_types)}\n"
            prompt += "- Create SEPARATE top-level objects for EACH statement type\n"
            prompt += "- Each statement may appear on different pages\n"

        prompt += "\nJSON STRUCTURE - CRITICAL:\n"
        prompt += "{\n"
        prompt += '  "<statement_type_1>": {\n'
        prompt += '    "metadata": {\n'
        prompt += '      "statement_type": "<statement type name>",\n'
        prompt += '      "company_name": "...",\n'
        prompt += '      "reporting_date": "YYYY-MM-DD",\n'
        prompt += '      "currency": "...",\n'
        prompt += '      "original_units": "...",\n'
        prompt += '      "units_multiplier": <number>,  // MANDATORY!\n'
        prompt += '      "dates_covered": "YYYY-MM-DD to YYYY-MM-DD or YYYY-MM-DD, YYYY-MM-DD",\n'
        prompt += '      "periods": [...]  // Periods specific to THIS statement\n'
        prompt += '    },\n'
        prompt += '    "extraction_notes": [...],\n'
        prompt += '    "<section_1>": [...],\n'
        prompt += '    "<section_2>": [...]\n'
        prompt += '  },\n'
        prompt += '  "<statement_type_2>": {\n'
        prompt += '    "metadata": { ... },\n'
        prompt += '    "extraction_notes": [...],\n'
        prompt += '    "<sections>": [...]\n'
        prompt += '  }\n'
        prompt += '}\n\n'

        prompt += "TOP-LEVEL KEYS - CRITICAL:\n"
        prompt += "- Use snake_case for statement type keys (e.g., 'balance_sheet', 'cash_flow', 'income_statement')\n"
        prompt += "- Each statement is a SEPARATE top-level key\n"
        prompt += "- Each statement contains its OWN metadata object\n"
        prompt += "- Each statement has its OWN extraction_notes array\n"
        prompt += "- Each statement has its OWN section arrays\n\n"

        prompt += "STATEMENT SEPARATION - HOW TO IDENTIFY:\n"
        prompt += "- Look for major headings that indicate statement type (e.g., 'BALANCE SHEET', 'CASH FLOW STATEMENT')\n"
        prompt += "- Use page breaks and visual separators to distinguish statements\n"
        prompt += "- Balance sheets typically show Assets and Liabilities\n"
        prompt += "- Cash flow statements show Operating/Investing/Financing Activities\n"
        prompt += "- Income statements show Revenue and Expenses\n"
        prompt += "- Each statement may have different column periods\n\n"

        # Add all the common rules (metadata, period extraction, hierarchy, etc.)
        prompt += PromptTemplates._build_common_extraction_rules()

        # Add detailed line item schema for multi-statement mode
        prompt += "DETAILED LINE ITEM SCHEMA FOR MULTI-STATEMENT MODE:\n\n"
        prompt += "Each statement object must follow this structure:\n\n"
        prompt += "{\n"
        prompt += '  "<statement_type>": {  // e.g., "balance_sheet", "cash_flow"\n'
        prompt += '    "metadata": {\n'
        prompt += '      "company_name": "string",  // REQUIRED\n'
        prompt += '      "statement_type": "string",  // REQUIRED\n'
        prompt += '      "reporting_date": "YYYY-MM-DD",  // REQUIRED - ISO format\n'
        prompt += '      "currency": "string",  // REQUIRED - e.g., "EUR", "USD"\n'
        prompt += '      "original_units": "string",  // REQUIRED - e.g., "EUR m", "in thousands"\n'
        prompt += '      "units_multiplier": number,  // REQUIRED! - 1000000 for millions, 1000 for thousands\n'
        prompt += '      "dates_covered": "string",  // REQUIRED - Range or comma-separated dates\n'
        prompt += '      "periods": [  // REQUIRED - Array of period objects\n'
        prompt += '        {\n'
        prompt += '          "label": "string",  // REQUIRED - e.g., "31.12.2024"\n'
        prompt += '          "iso_date": "YYYY-MM-DD",  // REQUIRED - e.g., "2024-12-31"\n'
        prompt += '          "context": "string"  // REQUIRED - e.g., "Exact date provided"\n'
        prompt += '        }\n'
        prompt += '      ]\n'
        prompt += '    },\n'
        prompt += '    "extraction_notes": ["string", ...],  // REQUIRED - Array of observations\n'
        prompt += '    "<dynamic_section_name_1>": [  // e.g., "assets", "operating_activities"\n'
        prompt += '      {\n'
        prompt += '        "line_number": number,  // REQUIRED! - Sequential number starting at 1\n'
        prompt += '        "label": "string",  // REQUIRED! - Exact label from document\n'
        prompt += '        "level": number,  // REQUIRED! - Hierarchy level (1, 2, 3, etc.)\n'
        prompt += '        "is_total": boolean,  // REQUIRED! - true for totals/subtotals\n'
        prompt += '        "notes_reference": "string or null",  // REQUIRED! - e.g., "Note 3.1" or null\n'
        prompt += '        "values": {  // REQUIRED! - Object with ISO date keys\n'
        prompt += '          "YYYY-MM-DD": number or null,  // ISO date keys matching metadata.periods[].iso_date\n'
        prompt += '          "YYYY-MM-DD": number or null  // Values MUST be multiplied by units_multiplier!\n'
        prompt += '        }\n'
        prompt += '      },\n'
        prompt += '      // ... more line items (each MUST have ALL 6 fields above)\n'
        prompt += '    ],\n'
        prompt += '    "<dynamic_section_name_2>": [...],\n'
        prompt += '    // ... more sections\n'
        prompt += '  },\n'
        prompt += '  "<another_statement_type>": {\n'
        prompt += '    "metadata": {...},  // Same required fields as above\n'
        prompt += '    "extraction_notes": [...],\n'
        prompt += '    "<sections>": [...]  // Each line item MUST have ALL 6 fields\n'
        prompt += '  }\n'
        prompt += '}\n\n'

        prompt += "⚠️  CRITICAL FIELD REQUIREMENTS - NO EXCEPTIONS ALLOWED ⚠️\n\n"
        prompt += "Each line item MUST include ALL 6 fields:\n"
        prompt += "  1. line_number (cannot be omitted)\n"
        prompt += "  2. label (cannot be omitted)\n"
        prompt += "  3. level (cannot be omitted)\n"
        prompt += "  4. is_total (cannot be omitted)\n"
        prompt += "  5. notes_reference (cannot be omitted - use null if no reference)\n"
        prompt += "  6. values (cannot be omitted)\n\n"
        prompt += "Metadata MUST include units_multiplier field (cannot be omitted)\n"
        prompt += "The 'values' object keys MUST be ISO dates (YYYY-MM-DD) matching the 'iso_date' from metadata.periods\n"
        prompt += "ALL numeric values in 'values' objects MUST be multiplied by units_multiplier BEFORE storing\n"
        prompt += "Section names are DYNAMIC based on document content (NOT hardcoded)\n\n"

        # Multi-statement specific final reminders - REPEAT CRITICAL REQUIREMENTS
        prompt += "=" * 80 + "\n"
        prompt += "FINAL MANDATORY CHECKLIST - VERIFY BEFORE RETURNING JSON:\n"
        prompt += "=" * 80 + "\n\n"

        prompt += "✓ EVERY line item has ALL 6 required fields: line_number, label, level, is_total, notes_reference, values\n"
        prompt += "✓ EVERY statement's metadata has units_multiplier field (MANDATORY!)\n"
        prompt += "✓ Units_multiplier is applied ONLY to relevant monetary amount values\n"
        prompt += "✓ Per-unit metrics, ratios, and percentages are stored as-is from the PDF\n"
        prompt += "✓ ALL period keys in values objects are ISO dates (YYYY-MM-DD format)\n"
        prompt += "✓ Each statement is a SEPARATE top-level key (e.g., 'balance_sheet', 'cash_flow')\n"
        prompt += "✓ Each statement has its OWN metadata, extraction_notes, and section arrays\n"
        prompt += "✓ Each statement may have DIFFERENT periods in its metadata\n"
        prompt += "✓ All numeric values are numbers (not strings like \"1234\")\n"
        prompt += "✓ Store values as an OBJECT with ISO DATE keys (YYYY-MM-DD format)\n"
        prompt += "✓ PRESERVE exact labels from document (no inference or interpretation)\n"
        prompt += "✓ MAINTAIN hierarchy with proper level indicators (1, 2, 3, 4)\n"
        prompt += "✓ Return ONLY JSON - no explanations, notes, or commentary outside the JSON\n"
        prompt += "✓ Your response MUST start with { and end with }\n\n"

        prompt += "=" * 80 + "\n"
        prompt += "IF YOU MISS ANY REQUIRED FIELD, THE EXTRACTION WILL FAIL VALIDATION!\n"
        prompt += "=" * 80 + "\n\n"

        prompt += "BEGIN EXTRACTION NOW. Output only the JSON:\n"

        return prompt

    @staticmethod
    def _build_common_extraction_rules() -> str:
        """Build common extraction rules shared by both single and multi-statement modes"""
        prompt = ""

        # METADATA EXTRACTION RULES
        prompt += "METADATA EXTRACTION RULES:\n"
        prompt += "1. AUTO-DETECT company name from document header/title/footer\n"
        prompt += "2. AUTO-DETECT reporting date/period (e.g., 'December 31, 2024' or 'Q4 2024')\n"
        prompt += "3. AUTO-DETECT currency (EUR, USD, GBP, etc.) from document symbols/text\n"
        prompt += "4. AUTO-DETECT units from table headers (thousands, millions, billions, etc.)\n"
        prompt += "5. CREATE dates_covered field by analyzing the periods:\n"
        prompt += "   - If periods form a consecutive sequence (quarterly/monthly), use range format:\n"
        prompt += "     'YYYY-MM-DD to YYYY-MM-DD' (e.g., '2024-03-31 to 2024-12-31')\n"
        prompt += "   - If periods are discrete/non-consecutive, use comma-separated format:\n"
        prompt += "     'YYYY-MM-DD, YYYY-MM-DD' (e.g., '2023-12-31, 2024-12-31')\n"
        prompt += "   - Use your judgment to determine if periods are consecutive based on the dates\n\n"

        # UNIT CONVERSION RULES (CRITICAL)
        prompt += "UNIT CONVERSION RULES - INTELLIGENT MULTIPLIER APPLICATION:\n\n"

        prompt += "CRITICAL NOTE: When a units_multiplier is specified (e.g., 1,000,000 for millions), "
        prompt += "apply it INTELLIGENTLY to only those values that represent monetary amounts in those units.\n\n"

        prompt += "Use your understanding of the data and context to determine:\n"
        prompt += "  • Which values are monetary amounts measured in the document's specified units "
        prompt += "(these should be multiplied)\n"
        prompt += "  • Which values are already expressed per-unit (per share, per item, etc.), "
        prompt += "as ratios, as percentages, or as counts (these should NOT be multiplied)\n\n"

        prompt += "Store values that should not be multiplied EXACTLY as they appear in the PDF.\n"
        prompt += "Apply the multiplier only to values representing absolute monetary amounts.\n\n"

        prompt += "If the document states 'in thousands', 'in millions', or 'in billions':\n"
        prompt += "- Set 'units_multiplier' field to show the multiplier:\n"
        prompt += "  * 1000 for thousands\n"
        prompt += "  * 1000000 for millions\n"
        prompt += "  * 1000000000 for billions\n"
        prompt += "  * 1 if values are already in base units\n"
        prompt += "- Set 'original_units' field to the exact unit text from the document\n\n"

        # VALUE CLEANING RULES (CRITICAL)
        prompt += "VALUE CLEANING RULES - FOLLOW EXACTLY:\n"
        prompt += "1. REMOVE all currency symbols (€, $, £, USD, EUR, etc.) from values\n"
        prompt += "   - Store currency in metadata only, NOT in numeric values\n"
        prompt += "2. REMOVE all thousand separators (commas, dots, spaces, apostrophes)\n"
        prompt += "   - '1,234,567' becomes 1234567\n"
        prompt += "   - '1.234.567' becomes 1234567\n"
        prompt += "3. CONVERT parentheses to negative numbers:\n"
        prompt += "   - '(1,234)' becomes -1234\n"
        prompt += "   - '(500.50)' becomes -500.50\n"
        prompt += "   - '(1,234,567)' becomes -1234567\n"
        prompt += "4. REMOVE decorative dots, dashes, or underscores before values:\n"
        prompt += "   - 'Cash and equivalents........500' → extract value as 500\n"
        prompt += "   - 'Total assets------------1234' → extract value as 1234\n"
        prompt += "5. Store ALL values as numbers (integer or float), NEVER as strings\n"
        prompt += "6. If a cell is empty or shows '-' or 'n/a', use null (not 0, not string)\n\n"

        # LABEL PRESERVATION RULES
        prompt += "LABEL PRESERVATION RULES - CRITICAL:\n"
        prompt += "- Extract labels EXACTLY as written in the document\n"
        prompt += "- DO NOT add context or interpretation (e.g., 'Total' should stay 'Total', not 'Total Assets')\n"
        prompt += "- DO NOT expand abbreviations unless explicitly shown in the document\n"
        prompt += "- Preserve capitalization, punctuation, and formatting exactly\n\n"

        # NOTE REFERENCE SEPARATION RULES
        prompt += "NOTE REFERENCE SEPARATION RULES - CRITICAL:\n"
        prompt += "When a line item has both a label and note references:\n"
        prompt += "1. DETECT note references by pattern matching:\n"
        prompt += "   - Single: 'Note X.X', 'Note X', 'Note X.X.X'\n"
        prompt += "   - Multiple: 'Notes X.X, X.X and X.X', 'Notes X.X and X.X', 'Notes X, X and X'\n"
        prompt += "2. SEPARATE the note reference from the label:\n"
        prompt += "   - Label goes in 'label' field (without notes)\n"
        prompt += "   - Notes go in 'notes_reference' field (exact text)\n"
        prompt += "3. EXAMPLES:\n"
        prompt += "   - Source: 'Financial assets at fair value through profit or loss Notes 3.1, 3.2 and 3.4'\n"
        prompt += "     → label: 'Financial assets at fair value through profit or loss'\n"
        prompt += "     → notes_reference: 'Notes 3.1, 3.2 and 3.4'\n"
        prompt += "   - Source: 'Customer loans at amortised cost Note 3.5'\n"
        prompt += "     → label: 'Customer loans at amortised cost'\n"
        prompt += "     → notes_reference: 'Note 3.5'\n"
        prompt += "   - Source: 'Cash, due from central banks' (no notes)\n"
        prompt += "     → label: 'Cash, due from central banks'\n"
        prompt += "     → notes_reference: null\n"
        prompt += "4. DO NOT include note references in labels - they belong only in notes_reference field\n\n"

        # SECTION STRUCTURE DETECTION
        prompt += "SECTION STRUCTURE DETECTION - CRITICAL:\n\n"
        prompt += "1. IDENTIFY how many distinct tables/sections are in the document:\n"
        prompt += "   - Look for major headings, table titles, section breaks\n"
        prompt += "   - Each major section typically becomes a separate JSON array\n\n"
        prompt += "2. PRESERVE the document's structure:\n"
        prompt += "   - If sections are separate tables → create separate arrays\n"
        prompt += "   - If subsections are nested within a table → keep them nested using 'level' field\n\n"
        prompt += "3. MAINTAIN hierarchy with 'level' field:\n"
        prompt += "   - Level 1: Main sections/categories\n"
        prompt += "   - Level 2: Subsections\n"
        prompt += "   - Level 3: Line items\n"
        prompt += "   - Level 4+: Sub-items (if needed)\n\n"
        prompt += "4. CAPTURE ALL content:\n"
        prompt += "   - Section headers (even if no numeric values)\n"
        prompt += "   - All line items\n"
        prompt += "   - All subtotals\n"
        prompt += "   - Final totals\n\n"

        # HIERARCHICAL STRUCTURE
        prompt += "HIERARCHICAL STRUCTURE RULES:\n"
        prompt += "- Assign 'level' to each line item to show hierarchy:\n"
        prompt += "  * level 1: Main category headers (e.g., 'Current Assets', 'Total Assets')\n"
        prompt += "  * level 2: Sub-categories (e.g., 'Cash and Cash Equivalents')\n"
        prompt += "  * level 3: Detail items (e.g., 'Cash in Bank Accounts')\n"
        prompt += "- Mark totals and subtotals with 'is_total': true\n"
        prompt += "  * Examples: 'Total Current Assets', 'Total Assets', 'Total Liabilities and Equity'\n"
        prompt += "- Preserve notes references if present (e.g., 'Notes 3.1, 3.2')\n"
        prompt += "- Maintain line_number sequence for all items\n\n"

        # PDF-SPECIFIC PATTERN ANALYSIS - CRITICAL
        prompt += "PDF-SPECIFIC PATTERN ANALYSIS - PERFORM THIS FIRST:\n\n"
        prompt += "STEP 1: ANALYZE THIS DOCUMENT'S UNIQUE FORMATTING PATTERNS\n"
        prompt += "Before extracting, scan the ENTIRE document to identify ITS specific formatting patterns:\n\n"
        prompt += "1. HEADING PATTERNS in this PDF:\n"
        prompt += "   - What makes a heading in THIS document? (all-caps, bold, font size, underline, specific indentation?)\n"
        prompt += "   - Are headings consistently formatted the same way throughout?\n"
        prompt += "   - Document your findings: 'In this PDF, major section headings are: [describe pattern]'\n\n"
        prompt += "2. TABLE STRUCTURE PATTERNS in this PDF:\n"
        prompt += "   - How does this document indicate table boundaries? (lines, spacing, column structure changes?)\n"
        prompt += "   - Does this PDF use visible borders/lines for tables or borderless tables?\n"
        prompt += "   - How are column headers formatted vs data rows?\n"
        prompt += "   - Document: 'In this PDF, tables are structured as: [describe]'\n\n"
        prompt += "3. SPACING PATTERNS in this PDF:\n"
        prompt += "   - What vertical spacing is used between sections vs within sections?\n"
        prompt += "   - Is spacing consistent throughout the document?\n"
        prompt += "   - Document: 'Section breaks in this PDF use: [describe spacing pattern]'\n\n"
        prompt += "4. TYPOGRAPHY PATTERNS in this PDF:\n"
        prompt += "   - What font sizes are used for different hierarchy levels?\n"
        prompt += "   - When is bold used vs regular weight?\n"
        prompt += "   - Are there consistent capitalization patterns (ALL CAPS for certain levels)?\n"
        prompt += "   - Document: 'Typography hierarchy in this PDF: [describe]'\n\n"
        prompt += "5. MULTI-PAGE TABLE CONTINUATION PATTERNS:\n"
        prompt += "   - When a table continues to next page: are column headers repeated?\n"
        prompt += "   - Are there 'continued' labels or indicators?\n"
        prompt += "   - Does the table maintain same column structure across pages?\n"
        prompt += "   - When a NEW table starts: what visual breaks are present?\n"
        prompt += "   - Document: 'Multi-page table patterns: [describe how to distinguish continuation vs new table]'\n\n"
        prompt += "6. INDENTATION PATTERNS in this PDF:\n"
        prompt += "   - Measure indentation levels used for hierarchy (0px, 10px, 20px, etc.)\n"
        prompt += "   - Are tabs or spaces used for indentation?\n"
        prompt += "   - How many indentation levels are present?\n"
        prompt += "   - What does each indentation level represent? (e.g., level 1 = main section, level 2 = subsection)\n"
        prompt += "   - Is indentation consistent throughout or does it vary?\n"
        prompt += "   - Document: 'Indentation hierarchy in this PDF: [describe levels and their pixel/character measurements]'\n\n"
        prompt += "7. LINE AND BORDER PATTERNS in this PDF:\n"
        prompt += "   - Does this PDF use horizontal lines to separate sections or rows?\n"
        prompt += "   - Are there vertical lines creating column boundaries?\n"
        prompt += "   - Line thickness/style differences (thin vs thick, solid vs dashed)?\n"
        prompt += "   - Where are lines used vs omitted (e.g., lines under headers but not data rows)?\n"
        prompt += "   - Do line patterns indicate hierarchy or section boundaries?\n"
        prompt += "   - Document: 'Line/border patterns: [describe where, how, and why lines are used]'\n\n"
        prompt += "8. PRECISE SPACING ANALYSIS in this PDF:\n"
        prompt += "   - Measure vertical spacing: how many pixels/lines between different elements?\n"
        prompt += "   - Within-section spacing vs between-section spacing (provide specific measurements)\n"
        prompt += "   - Horizontal spacing: distance between columns, left/right margins, padding\n"
        prompt += "   - Are spacing patterns consistent or do they vary by section type?\n"
        prompt += "   - Does spacing correlate with hierarchy or importance?\n"
        prompt += "   - Document: 'Spacing measurements: [e.g., 5px within sections, 20px between sections, 2px between rows]'\n\n"
        prompt += "9. ADVANCED VISUAL CUES in this PDF:\n"
        prompt += "   - Background colors or shading for headers vs data rows?\n"
        prompt += "   - Cell borders (full borders, partial borders, no borders, border positioning)?\n"
        prompt += "   - Alignment patterns (left-aligned text vs right-aligned numbers vs centered headers)?\n"
        prompt += "   - Visual grouping techniques (boxing, shading, whitespace, clustering)?\n"
        prompt += "   - Font weight variations beyond bold (semibold, light, regular)?\n"
        prompt += "   - Any other visual patterns unique to this document?\n"
        prompt += "   - Document: 'Advanced visual patterns: [describe any other PDF-specific visual cues]'\n\n"
        prompt += "STEP 2: APPLY THIS PDF'S PATTERNS TO DETECT STRUCTURE\n\n"
        prompt += "Once you've identified ALL patterns above (including indentation, lines, spacing), use them TOGETHER:\n\n"
        prompt += "- Identify section boundaries using: heading pattern + spacing pattern + line separators + indentation changes\n"
        prompt += "- Detect hierarchy levels using: indentation levels + typography + spacing + alignment\n"
        prompt += "- Recognize table boundaries using: lines/borders + spacing + column structure + visual grouping\n"
        prompt += "- Detect table continuations using: indentation consistency + column alignment + spacing patterns + header repetition\n"
        prompt += "- Group related items using: indentation + spacing + visual separators + alignment patterns\n\n"
        prompt += "LEVERAGE ADVANCED VISION CAPABILITIES:\n\n"
        prompt += "- MEASURE spacing precisely using your vision (count pixels, lines, characters)\n"
        prompt += "- DETECT subtle visual cues that distinguish hierarchy levels\n"
        prompt += "- OBSERVE indentation patterns that might not be obvious from text parsing alone\n"
        prompt += "- COUNT indentation pixels/characters to determine exact nesting levels\n"
        prompt += "- RECOGNIZE line patterns (thickness, style, position) to identify boundaries\n"
        prompt += "- ANALYZE whitespace distribution to detect visual groupings\n"
        prompt += "- COMBINE multiple visual signals for confident structure detection\n\n"
        prompt += "CRITICAL: Use vision to see the PDF as a human would - notice spacing, alignment, lines, and indentation\n"
        prompt += "that create visual hierarchy and structure. Don't rely solely on text content.\n\n"
        prompt += "STEP 3: LOG YOUR PATTERN ANALYSIS\n\n"
        prompt += "In extraction_notes, you MUST include comprehensive pattern analysis:\n\n"
        prompt += '- "Document formatting analysis: [heading, table, spacing, typography, indentation, lines, advanced visual]"\n'
        prompt += '- "Indentation hierarchy: [levels found with measurements, e.g., 0px/15px/30px for levels 1/2/3]"\n'
        prompt += '- "Line/border usage: [where and how lines separate elements, thickness/style patterns]"\n'
        prompt += '- "Spacing measurements: [specific pixel/line values, e.g., 3px row spacing, 15px section spacing]"\n'
        prompt += '- "Visual grouping cues: [how alignment, shading, borders create visual groups]"\n'
        prompt += '- "Section detection strategy: [how you combined ALL visual patterns to identify sections]"\n'
        prompt += '- "Table continuation detection: [how spacing + indentation + lines helped distinguish continued vs new]"\n'
        prompt += '- "Hierarchy detection: [how indentation + typography + spacing determined nesting levels]"\n\n'
        prompt += "CRITICAL: Each PDF is different. Do NOT apply memorized patterns from other documents.\n"
        prompt += "Analyze THIS document first, then extract based on what YOU observe in THIS specific PDF.\n\n"

        # MULTI-PAGE HANDLING
        prompt += "MULTI-PAGE TABLE CONTINUATION - CRITICAL:\n\n"
        prompt += "Tables often span multiple pages with headers/footers/logos in between. You MUST distinguish:\n\n"
        prompt += "TABLE CONTINUES TO NEXT PAGE (keep in same section) when:\n"
        prompt += "- Column headers are repeated at top of next page\n"
        prompt += "- Same column structure (same number and types of columns)\n"
        prompt += "- No new bold section heading between pages\n"
        prompt += "- Visual continuation indicators (e.g., 'continued', same table style)\n"
        prompt += "- Line items continue the same list/sequence\n"
        prompt += "- Consistent indentation and formatting\n\n"
        prompt += "NEW TABLE STARTS (create new section) when:\n"
        prompt += "- New bold heading or section title appears\n"
        prompt += "- Different column structure (different columns than previous table)\n"
        prompt += "- Clear visual break (extra spacing, horizontal line, new formatting)\n"
        prompt += "- Different semantic purpose (e.g., switching from Assets to Liabilities)\n\n"
        prompt += "HANDLING HEADERS/FOOTERS/LOGOS:\n"
        prompt += "- Ignore page numbers, headers, footers, and logos when detecting continuity\n"
        prompt += "- Focus on the table content itself\n"
        prompt += "- If table structure matches before/after page break → same table continues\n\n"
        prompt += "EXTRACT ALL PAGES:\n"
        prompt += "- Combine continued tables into single arrays\n"
        prompt += "- Maintain continuous line numbering across pages\n"
        prompt += "- DO NOT create separate sections for each page\n\n"
        prompt += "LOG MULTI-PAGE DECISIONS:\n"
        prompt += '- "Page X to Y: Table continues (same columns, no new heading)"\n'
        prompt += '- "Page Y: New section starts (bold heading \'...\' detected)"\n\n'

        # PERIOD EXTRACTION RULES
        prompt += "PERIOD EXTRACTION RULES - CRITICAL:\n"
        prompt += "1. Extract column header dates/periods EXACTLY as they appear in the document\n"
        prompt += "2. Store THREE formats in metadata.periods array:\n"
        prompt += '   - "label": The exact text from the column header\n'
        prompt += '   - "iso_date": ISO 8601 format (YYYY-MM-DD) - you MUST convert to ISO using your best judgment\n'
        prompt += '   - "context": Brief explanation of how you interpreted ambiguous dates (optional)\n'
        prompt += "3. Use the ISO DATE (iso_date field) as keys in all 'values' objects\n"
        prompt += "4. For interpreting partial/ambiguous dates, use your best judgment based on document context:\n"
        prompt += "   - '2024' alone → Infer fiscal year-end from document (e.g., '2024-06-30' if June FY, '2024-12-31' if calendar year)\n"
        prompt += "   - 'Q1 2024' → Quarter end date '2024-03-31'\n"
        prompt += "   - 'Q2 2024' → Quarter end date '2024-06-30'\n"
        prompt += "   - 'Q3 2024' → Quarter end date '2024-09-30'\n"
        prompt += "   - 'Q4 2024' → Quarter end date '2024-12-31'\n"
        prompt += "   - 'FY 2024' → Detect fiscal year-end from document context and headers\n"
        prompt += "5. Add 'context' field explaining your date interpretation when the date was ambiguous or required inference\n"
        prompt += "6. Examples:\n"
        prompt += '   - Document shows "As of June 30, 2024" → {"label": "As of June 30, 2024", "iso_date": "2024-06-30", "context": "Exact date provided"}\n'
        prompt += '   - Document shows "31.12.2024" → {"label": "31.12.2024", "iso_date": "2024-12-31", "context": "Exact date provided"}\n'
        prompt += '   - Document shows "Q4 2024" → {"label": "Q4 2024", "iso_date": "2024-12-31", "context": "Q4 end date"}\n'
        prompt += '   - Document shows "2024" with June fiscal year → {"label": "2024", "iso_date": "2024-06-30", "context": "Fiscal year ending June 30 (inferred from document header)"}\n\n'

        # SECTION DETECTION AND ARRAY NAMING
        prompt += "SECTION DETECTION AND ARRAY CREATION - CRITICAL:\n\n"
        prompt += "You MUST analyze the document and create JSON arrays dynamically:\n\n"
        prompt += "1. IDENTIFY all major sections/tables in the document:\n"
        prompt += "   - Look for table headers, section titles, bold headings\n"
        prompt += "   - Each major section becomes a separate JSON array\n\n"
        prompt += "2. NAME each array based on the section header:\n"
        prompt += "   - Convert to snake_case (lowercase with underscores)\n"
        prompt += "   - Remove special characters\n"
        prompt += "   - Keep names concise but descriptive\n"
        prompt += "   - Examples:\n"
        prompt += "     • 'CONSOLIDATED BALANCE SHEET - ASSETS' → 'assets'\n"
        prompt += "     • 'LIABILITIES AND EQUITY' → 'liabilities_and_equity'\n"
        prompt += "     • 'Operating Activities' → 'operating_activities'\n"
        prompt += "     • 'Revenue from Operations' → 'revenue_from_operations'\n\n"
        prompt += "3. NESTED sections:\n"
        prompt += "   - Keep nested items WITHIN parent array using 'level' field\n"
        prompt += "   - Example: If 'Equity' is a subsection under 'Liabilities', keep it in the 'liabilities' array\n\n"
        prompt += "4. DOCUMENT decisions in extraction_notes:\n"
        prompt += "   - Log which sections detected\n"
        prompt += "   - Log what you named each array and why\n\n"
        prompt += "5. VISUAL SECTION BOUNDARIES USING PDF-SPECIFIC PATTERNS:\n\n"
        prompt += "   Apply ALL patterns YOU identified in STEP 1 - combine multiple visual cues for accuracy.\n\n"
        prompt += "   CREATE a new section/array when you observe THIS PDF's COMBINED pattern signals:\n"
        prompt += "   - Heading pattern (e.g., ALL CAPS BOLD) + increased spacing before + horizontal line separator\n"
        prompt += "   - Indentation reset to level 0 + spacing change + different visual grouping\n"
        prompt += "   - Table structure change (different columns) + vertical line separator + new alignment pattern\n"
        prompt += "   - Look for CLUSTERS of visual cues, not just single indicators\n"
        prompt += "   - When 3+ visual signals align → high confidence section boundary\n\n"
        prompt += "   USE INDENTATION to determine hierarchy and grouping:\n"
        prompt += "   - Items at SAME indentation level → same hierarchy level (nest together)\n"
        prompt += "   - Increased indentation → nested/child item (increase level number)\n"
        prompt += "   - Decreased indentation → back to parent level (decrease level number)\n"
        prompt += "   - Consistent indentation across pages → table continues (same section)\n"
        prompt += "   - Indentation reset + other signals → new section starts\n\n"
        prompt += "   USE LINES/BORDERS to determine boundaries:\n"
        prompt += "   - Horizontal line across FULL WIDTH → likely section boundary\n"
        prompt += "   - Horizontal line WITHIN table columns → row separator (NOT section boundary)\n"
        prompt += "   - Vertical lines → column boundaries (table structure, not section breaks)\n"
        prompt += "   - Thick line vs thin line → may indicate major vs minor boundaries\n"
        prompt += "   - Line style change (solid → dashed) → may indicate hierarchy change\n\n"
        prompt += "   USE SPACING to determine grouping:\n"
        prompt += "   - Large vertical spacing (e.g., 20px+) → likely section boundary\n"
        prompt += "   - Small vertical spacing (e.g., 3-5px) → items within same section\n"
        prompt += "   - Spacing BEFORE element > spacing AFTER → likely new section starting\n"
        prompt += "   - Consistent small spacing across pages → table continues\n"
        prompt += "   - Sudden spacing increase → potential section break\n\n"
        prompt += "   USE ALIGNMENT and VISUAL GROUPING:\n"
        prompt += "   - Items with same left alignment → likely same hierarchy level\n"
        prompt += "   - Shading/background color change → may indicate new section\n"
        prompt += "   - Border boxing around items → visual group (keep together)\n"
        prompt += "   - Right-aligned numbers in consistent columns → table continues\n\n"
        prompt += "   DO NOT create new sections based on:\n"
        prompt += "   - Content interpretation alone (e.g., recognizing 'this looks like a summary')\n"
        prompt += "   - Memorized patterns from OTHER documents\n"
        prompt += "   - Semantic meaning without visual cues (e.g., 'cash reconciliation' needs visual break to be separate)\n"
        prompt += "   - Line item labels alone without THIS PDF's visual break pattern\n\n"
        prompt += "   DEFAULT BEHAVIOR when uncertain:\n"
        prompt += "   - If you're NOT 100% confident there's a visual break → KEEP items in the same section\n"
        prompt += "   - Lines that visually continue within the same table MUST stay together\n"
        prompt += "   - Subtotals followed by more rows WITHOUT a bold heading → stay in same section\n"
        prompt += "   - Summary lines at table end WITHOUT a bold heading → stay in same section\n\n"
        prompt += "   CONFIDENCE LOGGING - REQUIRED:\n"
        prompt += "   - Log in extraction_notes whenever section boundaries are uncertain\n"
        prompt += "   - Explain what visual cues you used (or didn't find) for each section boundary decision\n"
        prompt += "   - If confidence is LOW on a boundary → explicitly state you kept items together by default\n\n"
        prompt += "   Example logging:\n"
        prompt += '   - "Section boundary after line 15: Clear bold heading \'CASH FLOWS FROM OPERATING ACTIVITIES\' - high confidence"\n'
        prompt += '   - "Lines 45-50: No bold heading detected, items continue same table structure, kept in previous section - medium confidence"\n'
        prompt += '   - "Line 60: Uncertain if \'Net change in cash\' starts new section (no visual break observed), defaulted to keeping in last section"\n\n'
        prompt += "6. RE-EXAMINATION FOR LOW CONFIDENCE DECISIONS:\n\n"
        prompt += "   If you encounter ANY uncertainty about section boundaries:\n"
        prompt += "   - Revisit that specific area of the document\n"
        prompt += "   - Focus ONLY on visual formatting: bold text, font size, spacing, lines, indentation\n"
        prompt += "   - Ignore the semantic meaning of text\n"
        prompt += "   - Log your focused re-examination findings in extraction_notes\n\n"

        # EXTRACTION NOTES - REQUIRED
        prompt += "EXTRACTION NOTES - REQUIRED FOR QUALITY ASSURANCE:\n\n"
        prompt += "You MUST populate the extraction_notes array with detailed logging:\n\n"
        prompt += "0. PDF-SPECIFIC PATTERN ANALYSIS (REQUIRED FIRST):\n"
        prompt += '   - Document formatting analysis: [heading, table, spacing, typography, indentation, lines, advanced visual]\n'
        prompt += '   - Indentation hierarchy: [levels with measurements, e.g., "0px/15px/30px for levels 1/2/3"]\n'
        prompt += '   - Line/border usage: [where lines separate elements, thickness/style patterns]\n'
        prompt += '   - Spacing measurements: [specific values, e.g., "3px rows, 15px sections, 25px page breaks"]\n'
        prompt += '   - Visual grouping cues: [alignment, shading, borders, whitespace]\n'
        prompt += '   - Section detection strategy: [how you combined ALL visual patterns to identify sections]\n'
        prompt += '   - Hierarchy detection: [how indentation + typography + spacing determined nesting]\n'
        prompt += '   - Multi-page table continuation: [how spacing + indentation + lines helped distinguish]\n'
        prompt += '   - Pattern consistency: [formatting inconsistencies observed and how you handled them]\n\n'
        prompt += "1. STRUCTURAL DECISIONS:\n"
        prompt += '   - How many tables/sections detected and their names\n'
        prompt += '   - How sections are organized (e.g., "Equity nested in Liabilities")\n'
        prompt += '   - Array naming decisions (what you named each array and why)\n'
        prompt += '   - Multi-page decisions: [which tables spanned multiple pages and how you detected continuity]\n\n'
        prompt += "2. AMBIGUITIES ENCOUNTERED:\n"
        prompt += '   - Unclear cell values or labels\n'
        prompt += '   - How you interpreted them\n'
        prompt += '   - Confidence level in interpretation\n\n'
        prompt += "3. DATA HANDLING DECISIONS:\n"
        prompt += '   - How special formatting was handled (parentheses, dashes, n/a)\n'
        prompt += '   - Currency symbol removal\n'
        prompt += '   - Unit conversions applied\n'
        prompt += '   - Empty cells interpretation\n'
        prompt += '   - Note references separated from labels\n\n'
        prompt += "4. EDGE CASES:\n"
        prompt += '   - Merged cells across rows\n'
        prompt += '   - Multi-line labels\n'
        prompt += '   - Footnote references\n'
        prompt += '   - Any unusual formatting\n\n'
        prompt += "5. VERIFICATION NOTES:\n"
        prompt += '   - Key totals and subtotals identified\n'
        prompt += '   - Balance sheet equation verified (Assets = Liabilities + Equity)\n'
        prompt += '   - Any discrepancies noticed\n\n'
        prompt += "Format: Each note should be a clear, standalone sentence explaining one decision or observation.\n\n"

        return prompt

    @staticmethod
    def custom_prompt(user_prompt: str) -> str:
        """
        Wrap a custom user prompt with extraction best practices

        Args:
            user_prompt: The user's custom extraction request

        Returns:
            Enhanced prompt with extraction guidelines
        """
        prompt = f"{user_prompt}\n\n"
        prompt += "Extraction Guidelines:\n"
        prompt += "- Be thorough and include all relevant information\n"
        prompt += "- Preserve exact wording and numeric values\n"
        prompt += "- If content spans multiple pages, combine it\n"
        prompt += "- Maintain structure and hierarchy\n"
        prompt += "- Note any ambiguities or unclear sections"

        return prompt

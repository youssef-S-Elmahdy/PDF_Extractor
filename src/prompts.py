"""
Prompt templates for PDF extraction and validation
"""

from typing import Optional, List, Dict


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
    def financial_statement_repair_prompt(
        statement_heading: str,
        statement_key: str,
        required_row_labels: Optional[List[str]] = None
    ) -> str:
        """
        Prompt to (re-)extract a single missing statement, wrapped in a one-key JSON object
        so it can be merged into a multi-statement output.
        """
        prompt = "You are a financial data extraction API. You MUST return ONLY valid JSON.\n"
        prompt += "DO NOT include any explanatory text, commentary, notes, or markdown formatting.\n"
        prompt += "DO NOT wrap the JSON in code blocks (no ```json).\n"
        prompt += "Return raw JSON only.\n\n"

        prompt += f"TASK: Extract the COMPLETE statement whose heading matches: {statement_heading}\n"
        prompt += "Extract ALL rows and ALL numeric columns from that statement (no summarization).\n"
        prompt += "If the statement spans multiple pages, combine all parts.\n\n"

        prompt += "OUTPUT MUST BE A SINGLE JSON OBJECT WITH EXACTLY ONE TOP-LEVEL KEY:\n"
        prompt += "{\n"
        prompt += f'  "{statement_key}": {{\n'
        prompt += '    "metadata": {...},\n'
        prompt += '    "extraction_notes": [...],\n'
        prompt += '    "...section_arrays...": [...]\n'
        prompt += "  }\n"
        prompt += "}\n\n"

        prompt += "Follow these rules (high priority):\n"
        prompt += "- The statement object must include metadata and at least one top-level array of line items.\n"
        prompt += "- For matrix/reconciliation tables: store columns ONLY in metadata.columns and use a single line-item array named 'lines'.\n"
        prompt += "- Forbidden keys: sections, metadata_columns, line_items, items, line_kind, and any top-level 'columns' array outside metadata.\n"
        prompt += "- The ONLY allowed place for column definitions is metadata.columns (array of {key,label}). Do NOT use a 'header' field.\n"
        prompt += "- DO NOT AGGREGATE OR MERGE DISTINCT PDF ROWS. Each distinct PDF row must become one line item.\n"
        prompt += "- If a row label wraps across multiple lines, keep it as ONE row label; but never merge multiple separate rows into one.\n"
        prompt += "- If you cannot confidently read a numeric cell, include the row anyway and set that cell to null (do not drop the row).\n"
        prompt += "- Each line item must include: line_number, label, level, is_total, notes_reference, values.\n"

        if required_row_labels:
            prompt += "\nROWS THAT MUST APPEAR AS SEPARATE LINE ITEMS (do not merge them):\n"
            for r in required_row_labels[:30]:
                prompt += f"- {r}\n"

        prompt += "\n"
        prompt += PromptTemplates._build_common_extraction_rules()
        return prompt

    @staticmethod
    def _build_single_statement_prompt(statement_type: str) -> str:
        """Build prompt for single statement extraction (current behavior)"""
        # CRITICAL: Start with absolute prohibition on narrative
        prompt = "You are a financial data extraction API. You MUST return ONLY valid JSON.\n"
        prompt += "DO NOT include any explanatory text, commentary, notes, or markdown formatting.\n"
        prompt += "DO NOT wrap the JSON in code blocks (no ```json).\n"
        prompt += "Return raw JSON only, starting with { and ending with }.\n\n"

        # TASK & SCOPE
        prompt += f"TASK: Extract ONLY the {statement_type} from this financial document.\n"
        prompt += f"Extract ONLY sections belonging to {statement_type} - ignore other statements in the PDF.\n"
        prompt += f"Use headings and visual cues to identify {statement_type} boundaries. May span multiple pages.\n\n"

        # Add all the common rules
        prompt += PromptTemplates._build_common_extraction_rules()

        # Single-statement specific JSON schema (top-level section arrays only)
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
        prompt += '    ],\n'
        prompt += '    "columns": [\n'
        prompt += '      {"key": "snake_case", "label": "Exact column header text", "group": "optional"}\n'
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
        prompt += '      "row_kind": "position|movement|subtotal",\n'
        prompt += '      "row_as_of": "YYYY-MM-DD or null",\n'
        prompt += '      "row_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},\n'
        prompt += '      "row_description": "opening_balance|closing_balance|other or null",\n'
        prompt += '      "values": {"YYYY-MM-DD": number, "YYYY-MM-DD": number},\n'
        prompt += '      "notes_reference": ["X.X"]\n'
        prompt += '    }\n'
        prompt += '  ],\n'
        prompt += '  "<dynamic_section_name_2>": [ ... ],\n'
        prompt += '  "...additional sections as needed...": [ ... ]\n'
        prompt += '}\n\n'
        prompt += "IMPORTANT: The section names above (<dynamic_section_name_1>, etc.) are PLACEHOLDERS.\n"
        prompt += "You MUST replace them with actual names based on what you find in the document.\n"
        prompt += "DO NOT use these placeholder names in your actual output.\n"
        prompt += "The number of sections is also dynamic - create as many as the document has.\n\n"

        # FINAL REMINDERS
        prompt += "KEY REQUIREMENTS:\n"
        prompt += "- Return ONLY raw JSON (no text, markdown, or code blocks)\n"
        prompt += "- Values as numbers (not strings), stored in object keyed by metadata.periods iso_date OR metadata.columns keys\n"
        prompt += "- Use ONLY top-level section arrays (no 'sections' wrapper)\n"
        prompt += "- Dynamic section detection with snake_case array names from document headers\n"
        prompt += "- Preserve exact labels, split note references into notes_reference array\n"
        prompt += "- Matrix tables only (when metadata.periods is [] and metadata.columns is set): add row_kind/row_as_of/row_period\n"
        prompt += "- Apply units_multiplier to monetary amounts only\n\n"

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

        prompt += "DOCUMENT-LEVEL COVERAGE REQUIREMENT:\n"
        prompt += "- Scan the PDF and identify every distinct financial-statement table/section (including reconciliation/matrix statements).\n"
        prompt += "- Output one top-level statement object for each detected statement. Do NOT omit a detected statement.\n"
        prompt += "- If a detected statement is complex, you MUST still extract ALL rows/columns (do not summarize).\n\n"

        # REQUIRED FIELDS
        prompt += "REQUIRED FIELDS:\n"
        prompt += "Line items (EVERY item, no exceptions): line_number, label, level, is_total, notes_reference, values (object keyed by metadata.periods iso_date OR metadata.columns keys)\n"
        prompt += "Metadata: company_name, statement_type, reporting_date, currency, original_units, units_multiplier, dates_covered, periods\n"
        prompt += "Apply units_multiplier ONLY to monetary amounts, NOT to per-unit metrics, ratios, percentages, or counts.\n\n"

        # STRUCTURE
        prompt += "Create one top-level object per detected statement.\n"
        prompt += "Top-level statement keys must be derived from statement headings and converted to snake_case.\n"
        prompt += "If a statement is not present, omit its key. Do NOT invent extra top-level keys.\n"
        prompt += "Each statement is a top-level object with its own metadata, extraction_notes, and section arrays.\n"
        prompt += "JSON STRUCTURE: {<statement_key>: {metadata: {...}, extraction_notes: [...], <section_name>: [...]}}\n\n"

        # Add all the common rules (metadata, period extraction, hierarchy, etc.)
        prompt += PromptTemplates._build_common_extraction_rules()


        return prompt

    @staticmethod
    def _build_common_extraction_rules() -> str:
        """Build common extraction rules shared by both single and multi-statement modes"""
        prompt = ""

        # METADATA & PERIODS
        prompt += "METADATA & PERIODS:\n"
        prompt += "Auto-detect from document: company name, reporting date, currency, units (thousands/millions/billions)\n"
        prompt += "Create dates_covered: consecutive periods → 'YYYY-MM-DD to YYYY-MM-DD', discrete → 'YYYY-MM-DD, YYYY-MM-DD'\n"
        prompt += "Period extraction:\n"
        prompt += "  - Extract NUMERIC COLUMN headers exactly as shown (label)\n"
        prompt += "  - periods is ONLY for numeric column headers (never derive periods from row labels like 'At 1 January ...')\n"
        prompt += "  - Convert to ISO format YYYY-MM-DD using context (iso_date)\n"
        prompt += "  - Add context field if date interpretation required\n"
        prompt += "  - Use ISO dates as keys in values ONLY when the numeric columns are time periods\n\n"

        # VALUE EXTRACTION & CLEANING
        prompt += "VALUE EXTRACTION & CLEANING:\n"
        prompt += "Intelligent multiplier: Apply units_multiplier ONLY to monetary amounts in specified units. "
        prompt += "Do NOT multiply per-unit metrics, ratios, percentages, or counts - store these exactly as shown.\n"
        prompt += "Units: Set units_multiplier (1000/1000000/1000000000) and original_units from document.\n"
        prompt += "Cleaning:\n"
        prompt += "  - Remove currency symbols, thousand separators, decorative characters\n"
        prompt += "  - Convert (parentheses) to negative numbers\n"
        prompt += "  - Store as numbers (never strings)\n"
        prompt += "  - Missing values: use null (examples: blank, '-', '—', 'n/a', 'na')\n"
        prompt += "  - Never invent 0: only use 0 if the PDF explicitly shows 0 (or 0.0)\n"
        prompt += "Labels: Extract exactly as written, no interpretation or abbreviation expansion.\n"
        prompt += "Note references: Separate 'Note X' or 'Notes X, Y' patterns from labels into notes_reference field.\n\n"
        prompt += "notes_reference format: array of note identifiers (e.g., [\"3.1\", \"3.2\"]). Use [] if none.\n\n"

        # COMPLETENESS (ANTI-SUMMARY)
        prompt += "COMPLETENESS (NO SUMMARIZATION):\n"
        prompt += "Extract COMPLETE tables: ALL rows and ALL numeric columns.\n"
        prompt += "Do NOT replace a full table with a 'summary' unless the PDF itself is only a summary.\n"
        prompt += "If the table is large/complex, keep extraction_notes short; do not drop rows/columns.\n\n"

        # AXIS DECISION (TIME VS COMPONENT COLUMNS)
        prompt += "AXIS DECISION (TIME vs COMPONENT COLUMNS):\n"
        prompt += "Before extracting values, decide what the numeric columns represent:\n"
        prompt += "  A) Time periods (e.g., 31.12.2024 / 31.12.2023) → Use metadata.periods (with iso_date) and values keys = iso_date.\n"
        prompt += "  B) Components/measures (e.g., equity components, segment columns, risk categories) and time appears in ROW labels →\n"
        prompt += "     - Do NOT concatenate row labels with column headers.\n"
        prompt += "     - Set metadata.columns = [{\"key\":\"snake_case\",\"label\":\"Exact column header text\"}, ...]. Do NOT use a 'header' field.\n"
        prompt += "     - Set metadata.periods to an empty array [] (even if row labels contain dates like 'At 1 January 2023').\n"
        prompt += "     - For each row, set label = leftmost row header only, and values keys = metadata.columns[*].key (NOT the human header text).\n\n"

        # MATRIX / RECONCILIATION TABLES
        prompt += "MATRIX / RECONCILIATION TABLES (IMPORTANT):\n"
        prompt += "If the statement is a reconciliation/matrix (e.g., statement of changes in equity, rollforwards, segment matrices):\n"
        prompt += "  - OUTPUT MUST NOT BE A SUMMARY. Include ALL intermediate movement rows.\n"
        prompt += "  - Treat EACH ROW as one movement line item (opening balance, movements, subtotals, closing balance).\n"
        prompt += "  - Treat EACH NUMERIC COLUMN as a separate component (use metadata.columns).\n"
        prompt += "  - Store columns ONLY in metadata.columns as {key,label} (do NOT create metadata_columns; do NOT use a 'header' field).\n"
        prompt += "  - Use exactly ONE top-level array for the matrix rows named 'lines'. Do NOT use 'rows' for line items.\n"
        prompt += "  - Include ALL movement rows exactly as shown (including subtotals and allocation/transfer rows).\n"
        prompt += "  - Use null for blanks/dashes (do NOT use 0 unless the PDF explicitly shows 0).\n"
        prompt += "  - Sanity check alignment: if the table has a Total column, ensure it matches the sum of component columns for each row; if mismatch, re-check column alignment.\n"
        prompt += "  - Each matrix row is still a LINE ITEM and MUST include: line_number, label, level, is_total, notes_reference, values.\n"
        prompt += "  - Add row context fields to support reconstruction:\n"
        prompt += "      * row_kind: 'position'|'movement'|'subtotal'\n"
        prompt += "      * row_as_of: ISO date for 'position' rows (e.g., rows starting with 'At ...'); else null\n"
        prompt += "      * row_period: {\"start\":\"YYYY-MM-DD\",\"end\":\"YYYY-MM-DD\"} for movement/subtotal rows, inferred from surrounding 'At ...' rows\n"
        prompt += "      * row_description: optional short meaning (snake_case) to aid reconstruction\n"
        prompt += "  - SELF-CHECK BEFORE FINAL OUTPUT: if you output only 'position' rows (e.g., only 'At ...' lines) and no movement/subtotal rows, the extraction is INCOMPLETE → re-scan the table and include all intermediate rows.\n"
        prompt += "  - Do NOT collapse the matrix into a single totals row or a per-column summary.\n\n"

        # DOCUMENT ANALYSIS & STRUCTURE DETECTION
        prompt += "DOCUMENT ANALYSIS (PERFORM FIRST):\n"
        prompt += "Scan entire document to identify THIS PDF's unique visual patterns:\n"
        prompt += "  - Headings: bold, font size, caps, underline, indentation\n"
        prompt += "  - Tables: borders, column structure, header formatting\n"
        prompt += "  - Spacing: measure vertical/horizontal spacing (px/lines) between/within sections\n"
        prompt += "  - Typography: font sizes, weights, capitalization for different hierarchy levels\n"
        prompt += "  - Indentation: measure levels (0px/10px/20px), consistency, hierarchy meaning\n"
        prompt += "  - Lines/Borders: horizontal (section breaks vs rows), vertical (columns), thickness/style\n"
        prompt += "  - Visual grouping: alignment, shading, background colors, whitespace\n"
        prompt += "  - Multi-page: how continuations are shown (repeated headers, same structure)\n"
        prompt += "Apply patterns together: Use 3+ combined visual cues for confident structure detection.\n"
        prompt += "Leverage vision: Measure spacing precisely, count indentation pixels, recognize line patterns.\n"
        prompt += "Each PDF is unique - analyze THIS document, don't apply memorized patterns from others.\n\n"

        # MULTI-PAGE HANDLING
        prompt += "MULTI-PAGE TABLE CONTINUATION:\n"
        prompt += "Table continues (same section): Repeated headers, same column structure, no new bold heading, consistent indentation.\n"
        prompt += "New table starts (new section): New bold heading, different columns, clear visual break, spacing/line change.\n"
        prompt += "Ignore page numbers/headers/footers when detecting continuity.\n"
        prompt += "Combine continued tables into single arrays with continuous numbering.\n\n"


        # SECTION DETECTION AND ARRAY NAMING
        prompt += "SECTION DETECTION AND ARRAY CREATION:\n"
        prompt += "Create dynamic JSON arrays: Each major section/table becomes a separate array.\n"
        prompt += "Array naming: Convert section headers to snake_case, remove special chars, keep concise but descriptive.\n"
        prompt += "Nested sections: Keep within parent array using 'level' field.\n"
        prompt += "Section boundaries: Combine 3+ visual signals (heading + spacing + lines/borders + indentation changes).\n"
        prompt += "Hierarchy detection: Same indentation → same level, increased → nested child, decreased → back to parent.\n"
        prompt += "Default when uncertain: If no clear visual break, keep items in same section.\n"
        prompt += "Do NOT create sections based on content interpretation alone or memorized patterns from other documents.\n\n"

        # OUTPUT SHAPE (STRICT)
        prompt += "OUTPUT SHAPE (STRICT):\n"
        prompt += "Top-level within each statement must be: metadata, extraction_notes, and one or more SECTION ARRAYS of line items.\n"
        prompt += "Do NOT wrap sections (forbidden patterns: sections:[{section_name, line_items:[...]}] or sections:[{section_name, items:[...]}]).\n"
        prompt += "If you extracted column definitions, they MUST be inside metadata.columns.\n"
        prompt += "Do NOT create top-level arrays for non-line-item data (forbidden: metadata_columns, and any top-level 'columns' outside metadata).\n"
        prompt += "Do NOT create extra shape keys like items, rows (except as section arrays), line_items, line_kind, line_type.\n"
        prompt += "FINAL SELF-CHECK: ensure NO forbidden keys exist anywhere: sections, metadata_columns, line_items, items, line_kind; and ensure column definitions exist ONLY as metadata.columns.\n\n"

        # EXTRACTION NOTES
        prompt += "EXTRACTION NOTES:\n"
        prompt += "Log in extraction_notes array:\n"
        prompt += "  - Pattern analysis: Document formatting patterns identified (heading, spacing, indentation, lines, hierarchy)\n"
        prompt += "  - Structural decisions: Sections detected, array names, nesting, multi-page continuations\n"
        prompt += "  - Ambiguities: Prefix with 'AMBIGUITY:' and keep short (unclear values, interpretations, confidence levels)\n"
        prompt += "  - Data handling: Special formatting, unit conversions, empty cells, note references\n"
        prompt += "  - Edge cases: Merged cells, multi-line labels, unusual formatting\n"
        prompt += "  - Verification: Key totals, balance sheet equation (Assets = Liabilities + Equity), discrepancies\n\n"

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

    @staticmethod
    def notes_extraction_prompt(note_ids: List[str]) -> str:
        """
        Generate prompt for extracting specific notes from financial statements.

        Args:
            note_ids: List of note identifiers to extract (e.g., ["3.1", "7.1", "7.2"])

        Returns:
            Formatted notes extraction prompt
        """
        note_list = ", ".join(note_ids)

        prompt = "You are a financial notes extraction API. You MUST return ONLY valid JSON.\n"
        prompt += "DO NOT include any explanatory text, commentary, notes, or markdown formatting.\n"
        prompt += "DO NOT wrap the JSON in code blocks (no ```json).\n"
        prompt += "Return raw JSON only.\n\n"

        prompt += f"TASK: Extract ONLY these notes from the financial document: {note_list}\n"
        prompt += "IMPORTANT: Extract ONLY TABLES under each requested note. Skip all narrative paragraphs.\n"
        prompt += "If a note contains multiple tables (e.g., Table 8.3.A, 8.3.B, 8.3.C), you MUST extract ALL of them.\n"
        prompt += "NEVER return an empty tables array if the note contains tables.\n"
        prompt += "NEVER say you are skipping/deferring/omitting tables due to constraints.\n\n"

        prompt += "OUTPUT STRUCTURE - CRITICAL:\n"
        prompt += "- Each note MUST be a TOP-LEVEL KEY: note_<note_id> with dots → underscores (e.g., note_8_3).\n"
        prompt += "- Each note MUST contain metadata, extraction_notes, and a tables array.\n"
        prompt += "- Each item in tables MUST represent exactly ONE printed table (do not merge tables).\n\n"

        prompt += "{\n"
        prompt += '  "note_8_3": {\n'
        prompt += '    "metadata": {\n'
        prompt += '      "statement_type": "note",\n'
        prompt += '      "note_id": "8.3",\n'
        prompt += '      "note_title": "Exact note heading from PDF",\n'
        prompt += '      "company_name": "...",\n'
        prompt += '      "reporting_date": "YYYY-MM-DD",\n'
        prompt += '      "currency": "Auto-detected",\n'
        prompt += '      "original_units": "Auto-detected",\n'
        prompt += '      "units_multiplier": 1000000,\n'
        prompt += '      "dates_covered": "YYYY-MM-DD to YYYY-MM-DD or YYYY-MM-DD, YYYY-MM-DD",\n'
        prompt += '      "periods": []\n'
        prompt += '    },\n'
        prompt += '    "extraction_notes": ["Kept short; list detected tables and any ambiguities"],\n'
        prompt += '    "tables": [\n'
        prompt += '      {\n'
        prompt += '        "table_id": "Table 8.3.A",\n'
        prompt += '        "table_title": "Exact table caption/heading near the table (if present)",\n'
        prompt += '        "table_description": "One short sentence describing what the table represents.",\n'
        prompt += '        "table_type": "time_series or matrix",\n'
        prompt += '        "metadata": {\n'
        prompt += '          "currency": "Auto-detected or null if not applicable",\n'
        prompt += '          "original_units": "Auto-detected (EUR m / Number of shares / % / etc.)",\n'
        prompt += '          "units_multiplier": 1000000,\n'
        prompt += '          "dates_covered": "YYYY-MM-DD to YYYY-MM-DD or YYYY-MM-DD, YYYY-MM-DD",\n'
        prompt += '          "periods": [{"label":"Exact column header","iso_date":"YYYY-MM-DD","context":"optional"}],\n'
        prompt += '          "columns": [{"key":"snake_case","label":"Exact leaf column header text","value_type":"number|text|date|percent"}]\n'
        prompt += '        },\n'
        prompt += '        "lines": [\n'
        prompt += '          {"line_number": 1, "label": "Exact row label", "level": 0, "is_total": false, "notes_reference": [], "values": {"YYYY-MM-DD": 123}}\n'
        prompt += '        ]\n'
        prompt += '      }\n'
        prompt += '    ]\n'
        prompt += '  }\n'
        prompt += "}\n\n"

        prompt += "HARD RULES (NON-NEGOTIABLE):\n"
        prompt += "- TABLES ONLY. Do NOT include narrative text sections; do NOT output explanatory_text.\n"
        prompt += "- Do NOT summarize tables. Extract ALL rows and ALL numeric columns.\n"
        prompt += "- Do NOT output only totals/aggregates. Every printed row must be a line item.\n"
        prompt += "- Do NOT merge multiple distinct tables into one.\n"
        prompt += "- Do NOT collapse multiple printed rows into one row.\n"
        prompt += "- Table identification: table_id must match the printed label exactly (e.g., 'Table 8.3.A').\n"
        prompt += "- If you detect a printed table label (e.g., 'Table 3.1.H'), you MUST include a corresponding table object with non-empty lines.\n"
        prompt += "- Note metadata.periods MUST be [] (notes store axes per table only).\n"
        prompt += "- If a table shows comparative columns (e.g., 31.12.2024 and 31.12.2023), you MUST include BOTH (never drop a year/date block).\n"
        prompt += "- Apply the SAME numeric cleaning rules as statements:\n"
        prompt += "  * Missing values (blank, '-', '—', 'n/a') => null\n"
        prompt += "  * Never invent 0; only use 0 if the PDF explicitly shows 0\n"
        prompt += "  * Parentheses => negative numbers\n"
        prompt += "  * Apply units_multiplier ONLY to monetary amounts; do NOT multiply counts/percentages/ratios\n"
        prompt += "- Axis decision per table (choose EXACTLY ONE):\n"
        prompt += "  * time_series: metadata.periods (non-empty), metadata.columns = [], and values keys = iso_date\n"
        prompt += "  * matrix: metadata.columns (non-empty), metadata.periods = [], and values keys = metadata.columns[*].key\n"
        prompt += "- Multi-level / grouped headers (e.g., date groups with sub-columns) MUST be matrix tables:\n"
        prompt += "  * Flatten to LEAF columns; each leaf column becomes one metadata.columns entry.\n"
        prompt += "  * labels should include full header path (e.g., '31.12.2024 | Level 1').\n"
        prompt += "  * keys must be snake_case and MUST NOT be raw ISO dates (prefix if needed).\n"
        prompt += "- Text or mixed tables: set column value_type accordingly:\n"
        prompt += "  * number => int/float/null\n"
        prompt += "  * percent => numeric percent (e.g., '5%' => 5.0), not a string\n"
        prompt += "  * date => ISO date string 'YYYY-MM-DD' if present, otherwise keep as text\n"
        prompt += "  * text => string\n"
        prompt += "- Always keep table_description to 1 short sentence.\n\n"

        prompt += "NOTES LOCATION:\n"
        prompt += "- Notes usually appear after primary statements.\n"
        prompt += "- Each note starts with an identifier (e.g., 'NOTE 7.1', 'Note 7.1:').\n"
        prompt += "- Extract ALL tables until the next note heading begins.\n\n"

        prompt += f"Extract ONLY notes: {note_list}\n"
        prompt += "If a requested note is not found, omit it from output.\n"

        # Keep notes prompt lean: the table-level metadata rules above are the source of truth.

        return prompt

    @staticmethod
    def notes_tables_repair_prompt(note_to_table_ids: Dict[str, List[str]]) -> str:
        """
        Generate a prompt to re-extract ONLY specific table(s) within one or more notes.

        Args:
            note_to_table_ids: Mapping like {"3.4": ["Table 3.4.A"], "8.3": ["Table 8.3.B", "Table 8.3.C"]}
        """
        # Keep stable ordering for determinism
        note_ids = sorted(note_to_table_ids.keys(), key=lambda x: [int(p) for p in x.split(".") if p.isdigit()])

        prompt = "You are a financial notes extraction API. You MUST return ONLY valid JSON.\n"
        prompt += "DO NOT include any explanatory text, commentary, notes, or markdown formatting.\n"
        prompt += "DO NOT wrap the JSON in code blocks (no ```json).\n"
        prompt += "Return raw JSON only.\n\n"

        prompt += "TASK: Re-extract ONLY the specified tables under the specified notes.\n"
        prompt += "IMPORTANT: TABLES ONLY. Skip all narrative paragraphs.\n\n"

        prompt += "NOTES + TABLES TO EXTRACT:\n"
        for nid in note_ids:
            tables = note_to_table_ids.get(nid) or []
            if not tables:
                continue
            prompt += f"- Note {nid}: " + ", ".join(tables) + "\n"

        prompt += "\nOUTPUT RULES:\n"
        prompt += "- Output MUST include ONLY these notes as top-level keys (note_<id> with dots→underscores).\n"
        prompt += "- For each note, include metadata, extraction_notes, and tables[].\n"
        prompt += "- For each requested table_id, include exactly ONE table object with COMPLETE rows/columns.\n"
        prompt += "- Do NOT include non-requested tables.\n"
        prompt += "- Apply the SAME axis + null/0 + multiplier rules as notes_extraction_prompt.\n\n"

        prompt += "Return JSON only.\n"
        return prompt

"""
Prompt templates for PDF extraction and validation
"""

from typing import Optional


class PromptTemplates:
    """Collection of prompt templates for PDF extraction tasks"""

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
        Specialized prompt for financial statement extraction

        Args:
            statement_type: Type of financial statement (balance sheet, income statement, etc.)

        Returns:
            Formatted financial statement extraction prompt
        """
        prompt = f"Extract the complete {statement_type} from this PDF.\n\n"
        prompt += "Financial Statement Requirements:\n"
        prompt += "- Include all line items with exact labels\n"
        prompt += "- Capture all time periods/columns (years, quarters, etc.)\n"
        prompt += "- Include all subtotals and category totals\n"
        prompt += "- Preserve the account hierarchy (assets > current assets > cash, etc.)\n"
        prompt += "- Include all numeric values with correct signs (positive/negative)\n"
        prompt += "- Note the currency and units (thousands, millions, etc.)\n"
        prompt += "- Include any footnote references or annotations\n"
        prompt += "- If multi-page, combine all sections into one complete statement\n\n"
        prompt += "Format as JSON with this structure:\n"
        prompt += "{\n"
        prompt += '  "statement_type": "' + statement_type + '",\n'
        prompt += '  "currency": "EUR/USD/etc",\n'
        prompt += '  "units": "thousands/millions",\n'
        prompt += '  "periods": ["2024", "2023", ...],\n'
        prompt += '  "line_items": [\n'
        prompt += '    {\n'
        prompt += '      "label": "Account name",\n'
        prompt += '      "level": 1,  // hierarchy level\n'
        prompt += '      "values": [value_2024, value_2023, ...],\n'
        prompt += '      "is_total": false\n'
        prompt += '    }\n'
        prompt += '  ]\n'
        prompt += '}'

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

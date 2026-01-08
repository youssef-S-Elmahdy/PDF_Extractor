"""
Notes extraction module for financial statements
"""
from typing import List, Set
import re


class NoteIdentifier:
    """Structured note identifier"""
    def __init__(self, full_id: str):
        self.full_id = full_id
        parts = full_id.split('.')
        self.parent = parts[0]
        self.children = parts[1:] if len(parts) > 1 else []

    def __str__(self):
        return self.full_id

    def __lt__(self, other):
        # For sorting: "3" < "3.1" < "7" < "7.1"
        def to_tuple(id_str):
            return tuple(int(x) for x in id_str.split('.'))
        return to_tuple(self.full_id) < to_tuple(other.full_id)


class NotesExtractor:
    """Extract referenced notes from financial statements"""

    @staticmethod
    def collect_note_references(financial_json: dict) -> List[str]:
        """
        Traverse financial JSON and collect all unique note references.

        Args:
            financial_json: Extracted financial statements

        Returns:
            Sorted list of unique note identifiers (e.g., ["3.1", "7.1", "7.2"])
        """
        references: Set[str] = set()

        def traverse(obj):
            if isinstance(obj, dict):
                # Check for notes_reference field
                if "notes_reference" in obj and obj["notes_reference"]:
                    refs = obj["notes_reference"]
                    if isinstance(refs, list):
                        for ref in refs:
                            normalized = NotesExtractor.normalize_note_reference(ref)
                            if normalized:
                                references.add(normalized)
                    elif isinstance(refs, str):
                        normalized = NotesExtractor.normalize_note_reference(refs)
                        if normalized:
                            references.add(normalized)

                # Recurse through dict values
                for value in obj.values():
                    traverse(value)

            elif isinstance(obj, list):
                for item in obj:
                    traverse(item)

        traverse(financial_json)

        # Sort note references
        return sorted(list(references), key=lambda x: NoteIdentifier(x))

    @staticmethod
    def normalize_note_reference(ref: str) -> str:
        """
        Normalize various note reference formats to standard form.

        Examples:
            "Note 7.1" → "7.1"
            "Notes 3" → "3"
            "7.1.2" → "7.1.2"
            "3" → "3"
            "X.X" → "X.X"  # Keep as-is if non-numeric

        Args:
            ref: Raw note reference string

        Returns:
            Normalized note identifier
        """
        if not ref or not isinstance(ref, str):
            return ""

        # Extract numeric pattern: digits with optional dots
        pattern = r'(\d+(?:\.\d+)*)'
        match = re.search(pattern, ref.strip())

        return match.group(1) if match else ref.strip()

    @staticmethod
    def parse_note_identifier(note_id: str) -> NoteIdentifier:
        """Parse note identifier into structured form"""
        return NoteIdentifier(note_id)

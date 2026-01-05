"""
PDF upload module for OpenAI Files API
"""

import os
from pathlib import Path
from typing import Optional
from openai import OpenAI
from rich.console import Console

console = Console()


class PDFUploader:
    """Handles PDF file uploads to OpenAI Files API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the PDF uploader

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key)
        self.uploaded_files = {}  # Cache of uploaded files {file_path: file_id}

    def upload_pdf(self, pdf_path: str) -> str:
        """
        Upload a PDF to OpenAI Files API

        Args:
            pdf_path: Path to the PDF file

        Returns:
            file_id: The OpenAI file ID for the uploaded PDF

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a PDF
        """
        # Validate file exists
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Validate it's a PDF
        if pdf_file.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF, got: {pdf_file.suffix}")

        # Check cache
        abs_path = str(pdf_file.absolute())
        if abs_path in self.uploaded_files:
            console.print(f"[yellow]Using cached file_id for {pdf_file.name}[/yellow]")
            return self.uploaded_files[abs_path]

        # Upload to OpenAI
        console.print(f"[blue]Uploading {pdf_file.name} ({self._format_size(pdf_file.stat().st_size)})...[/blue]")

        try:
            with open(pdf_path, "rb") as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose="user_data"
                )

            file_id = file_response.id
            self.uploaded_files[abs_path] = file_id

            console.print(f"[green]✓ Upload successful: {file_id}[/green]")
            return file_id

        except Exception as e:
            console.print(f"[red]✗ Upload failed: {str(e)}[/red]")
            raise

    def get_file_info(self, file_id: str) -> dict:
        """
        Get information about an uploaded file

        Args:
            file_id: The OpenAI file ID

        Returns:
            Dictionary with file information
        """
        try:
            file_info = self.client.files.retrieve(file_id)
            return {
                "id": file_info.id,
                "filename": file_info.filename,
                "bytes": file_info.bytes,
                "created_at": file_info.created_at,
                "purpose": file_info.purpose,
                "status": file_info.status
            }
        except Exception as e:
            console.print(f"[red]Error retrieving file info: {str(e)}[/red]")
            raise

    def delete_file(self, file_id: str) -> bool:
        """
        Delete an uploaded file from OpenAI

        Args:
            file_id: The OpenAI file ID to delete

        Returns:
            True if deletion was successful
        """
        try:
            self.client.files.delete(file_id)
            # Remove from cache
            self.uploaded_files = {k: v for k, v in self.uploaded_files.items() if v != file_id}
            console.print(f"[green]✓ File deleted: {file_id}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error deleting file: {str(e)}[/red]")
            return False

    def list_uploaded_files(self) -> list:
        """
        List all files uploaded to OpenAI

        Returns:
            List of file information dictionaries
        """
        try:
            files = self.client.files.list()
            return [
                {
                    "id": f.id,
                    "filename": f.filename,
                    "bytes": f.bytes,
                    "created_at": f.created_at
                }
                for f in files.data
            ]
        except Exception as e:
            console.print(f"[red]Error listing files: {str(e)}[/red]")
            raise

    @staticmethod
    def _format_size(bytes_size: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"

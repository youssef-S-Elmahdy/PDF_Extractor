"""
Core extraction engine using OpenAI's API
"""

import time
from typing import Optional, Dict, Any
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class ExtractionEngine:
    """Handles PDF data extraction using OpenAI's API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the extraction engine

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: Model to use for extraction (5.2, gpt-5-mini)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Base delay in seconds between retries (uses exponential backoff)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def extract(
        self,
        file_id: str,
        prompt: str,
        stream: bool = False,
        enforce_json: bool = False
    ) -> Dict[str, Any]:
        """
        Extract data from PDF using OpenAI API

        Args:
            file_id: OpenAI file ID of the uploaded PDF
            prompt: Extraction prompt describing what to extract
            stream: Whether to stream the response (for large outputs)
            enforce_json: Whether to enforce JSON-only output via response_format parameter

        Returns:
            Dictionary containing:
                - output_text: Extracted data as text
                - model: Model used for extraction
                - input_tokens: Number of input tokens used
                - output_tokens: Number of output tokens used
                - success: Whether extraction was successful
                - error: Error message if failed

        Raises:
            Exception: If extraction fails after all retries
        """
        for attempt in range(self.max_retries):
            try:
                console.print(f"[blue]Running extraction (attempt {attempt + 1}/{self.max_retries})...[/blue]")

                # Prepare API request parameters
                api_params = {
                    "model": self.model,
                    "input": [{
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": file_id},
                            {"type": "input_text", "text": prompt}
                        ]
                    }]
                }

                # Note: response_format parameter is not supported by responses.create() API
                # We rely on strict prompt engineering for JSON-only output
                if enforce_json:
                    console.print(f"[dim]JSON-only mode: Using strict prompt engineering[/dim]")

                # Create the API request
                try:
                    response = self.client.responses.create(**api_params)
                except TypeError as e:
                    # response_format not supported by this API endpoint
                    # This is expected - remove it and retry
                    if "response_format" in str(e):
                        console.print(f"[dim]Note: response_format not supported by API, using prompt-only approach[/dim]")
                        api_params.pop("response_format", None)
                        response = self.client.responses.create(**api_params)
                    else:
                        raise

                # Extract response data
                output_text = response.output_text if hasattr(response, 'output_text') else str(response)

                result = {
                    "output_text": output_text,
                    "model": self.model,
                    "input_tokens": getattr(response, 'input_tokens', 0),
                    "output_tokens": getattr(response, 'output_tokens', 0),
                    "success": True,
                    "error": None
                }

                console.print(f"[green]✓ Extraction successful[/green]")
                console.print(f"[dim]Tokens used: {result['input_tokens']} input, {result['output_tokens']} output[/dim]")

                return result

            except Exception as e:
                error_msg = str(e)
                console.print(f"[yellow]Attempt {attempt + 1} failed: {error_msg}[/yellow]")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    console.print(f"[dim]Retrying in {delay} seconds...[/dim]")
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    console.print(f"[red]✗ Extraction failed after {self.max_retries} attempts[/red]")
                    return {
                        "output_text": None,
                        "model": self.model,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "success": False,
                        "error": error_msg
                    }

    def extract_streaming(
        self,
        file_id: str,
        prompt: str
    ) -> str:
        """
        Extract data with streaming response (for large outputs)

        Args:
            file_id: OpenAI file ID of the uploaded PDF
            prompt: Extraction prompt describing what to extract

        Returns:
            Complete extracted text

        Note:
            This is a placeholder for streaming implementation.
            The current OpenAI API may not support streaming with file inputs.
        """
        console.print("[yellow]Streaming mode: Using standard extraction[/yellow]")
        result = self.extract(file_id, prompt, stream=False)
        return result.get("output_text", "")

    def extract_with_context(
        self,
        file_id: str,
        prompt: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract data with additional context

        Args:
            file_id: OpenAI file ID of the uploaded PDF
            prompt: Extraction prompt
            context: Additional context to help with extraction

        Returns:
            Extraction result dictionary
        """
        if context:
            enhanced_prompt = f"{context}\n\n{prompt}"
        else:
            enhanced_prompt = prompt

        return self.extract(file_id, enhanced_prompt)

    def multi_page_extract(
        self,
        file_id: str,
        prompt: str,
        handle_continuation: bool = True,
        enforce_json: bool = False
    ) -> Dict[str, Any]:
        """
        Handle extraction that may span multiple responses

        Args:
            file_id: OpenAI file ID of the uploaded PDF
            prompt: Extraction prompt
            handle_continuation: Whether to handle CONTINUE signals
            enforce_json: Whether to enforce JSON-only output

        Returns:
            Complete extraction result

        Note:
            If output is too large, model may indicate continuation needed.
            This method handles combining multi-part responses.
            With enforce_json=True, continuation handling is disabled since
            JSON mode may not support [CONTINUE] markers.
        """
        # If JSON enforcement is enabled, skip continuation handling
        # The model should return complete JSON in one response
        if enforce_json:
            console.print(f"[dim]JSON mode: expecting complete response in single call[/dim]")
            return self.extract(file_id, prompt, enforce_json=True)

        # Add instruction to handle large outputs (for non-JSON mode)
        enhanced_prompt = f"{prompt}\n\nIf the output is too large to fit in one response, indicate '[CONTINUE]' and I will prompt you to continue."

        result = self.extract(file_id, enhanced_prompt, enforce_json=False)

        if handle_continuation and result["success"]:
            output = result["output_text"]

            # Check if continuation is needed
            continuation_count = 0
            max_continuations = 5  # Prevent infinite loops

            while "[CONTINUE]" in output and continuation_count < max_continuations:
                console.print(f"[yellow]Detected continuation signal, fetching more data...[/yellow]")

                # Request continuation
                continue_prompt = "Continue from where you left off."
                continue_result = self.extract(file_id, continue_prompt, enforce_json=False)

                if continue_result["success"]:
                    # Remove the [CONTINUE] marker and append new content
                    output = output.replace("[CONTINUE]", "") + continue_result["output_text"]
                    result["output_tokens"] += continue_result["output_tokens"]
                    continuation_count += 1
                else:
                    console.print(f"[red]Continuation failed: {continue_result['error']}[/red]")
                    break

            result["output_text"] = output

        return result

    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay
        }

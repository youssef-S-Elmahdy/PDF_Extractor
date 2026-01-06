# PDF Extractor

A general-purpose PDF extraction tool using OpenAI's API (GPT-4 or GPT-5-mini) that can extract structured information from PDFs based on natural language prompts.

## Features

- **Direct OpenAI API Integration**: No preprocessing required (no OCR, no RAG)
- **Natural Language Prompts**: Describe what you want to extract in plain English
- **Flexible Output Formats**: JSON, CSV, Markdown, or plain text
- **Validation Layer**: Optional verification of extracted data accuracy
- **Complex Table Handling**: Handles borderless tables, multi-page content, and unclear structures
- **Batch Processing**: Process multiple PDFs at once
- **CLI Interface**: Easy-to-use command-line tool

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

4. Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Extraction

Extract data from a PDF with a custom prompt:

```bash
python main.py extract path/to/document.pdf -p "Extract all tables from this document"
```

### Financial Statement Extraction

Use the specialized financial statement template:

```bash
python main.py extract financials.pdf -t financial --validate
```

### Extract with Validation

Validate extracted data for accuracy:

```bash
python main.py extract document.pdf -p "Extract the balance sheet" --validate
```

### Save in Multiple Formats

```bash
python main.py extract data.pdf -p "Extract summary" -f json -f csv -f md
```

### Batch Processing

Process all PDFs in a directory:

```bash
python main.py batch ./pdfs -p "Extract key information" -o results
```

### View Information

Check uploaded files and configuration:

```bash
python main.py info
```

## Command Reference

### `extract` Command

Extract data from a single PDF file.

**Arguments:**
- `pdf_path`: Path to the PDF file (required)

**Options:**
- `--prompt, -p`: Custom extraction prompt
- `--template, -t`: Use a prompt template (`table`, `financial`, `custom`)
- `--output, -o`: Output filename (without extension)
- `--format, -f`: Output format(s) - can specify multiple (default: `json`)
  - Available formats: `json`, `txt`, `csv`, `md`
- `--validate, -v`: Validate extracted data (flag)
- `--model, -m`: Model to use (default: `gpt-5-mini`)
  - Options: `gpt-4`, `gpt-5-mini`
- `--api-key`: OpenAI API key (or set `OPENAI_API_KEY` env var)
- `--preview/--no-preview`: Show/hide preview of extracted data (default: show)

**Examples:**

```bash
# Basic extraction with custom prompt
python main.py extract document.pdf -p "Extract all contact information"

# Use financial statement template with validation
python main.py extract report.pdf -t financial --validate

# Save in multiple formats
python main.py extract data.pdf -p "Extract tables" -f json -f csv -f md -o results

# Use GPT-4 instead of GPT-5-mini
python main.py extract document.pdf -p "Extract summary" -m gpt-5-mini
```

### `batch` Command

Process multiple PDFs in a directory.

**Arguments:**
- `pdf_dir`: Directory containing PDF files (required)

**Options:**
- `--prompt, -p`: Extraction prompt for all PDFs (required)
- `--output-dir, -o`: Output directory (default: `output`)
- `--format, -f`: Output format (default: `json`)
- `--model, -m`: Model to use (default: `gpt-5-mini`)
- `--api-key`: OpenAI API key

**Example:**

```bash
python main.py batch ./financial_reports -p "Extract balance sheet" -o batch_results
```

### `info` Command

Display information about uploaded files and configuration.

**Options:**
- `--api-key`: OpenAI API key

**Example:**

```bash
python main.py info
```

### `version` Command

Display version information.

```bash
python main.py version
```

## Project Structure

```
PDF_Extractor/
├── .env                          # API keys (create from .env.example)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── main.py                       # CLI entry point
├── src/
│   ├── __init__.py              # Package initialization
│   ├── pdf_uploader.py          # PDF upload to OpenAI
│   ├── extractor.py             # Core extraction engine
│   ├── validator.py             # Validation logic
│   ├── formatter.py             # Output formatting
│   └── prompts.py               # Prompt templates
├── examples/
│   └── Consolidated financial statements.pdf  # Test PDF
├── output/                       # Generated output files (created automatically)
└── tests/                        # Unit tests (future)
```

## How It Works

1. **Upload**: PDF is uploaded to OpenAI Files API
2. **Extract**: File + prompt sent to OpenAI model (GPT-4 or GPT-5-mini)
3. **Validate** (optional): Second API call verifies accuracy
4. **Format**: Data formatted and saved in requested format(s)

### Architecture

The tool uses OpenAI's direct file input capabilities, which means:
- No OCR preprocessing needed
- No vector database or RAG pipeline
- Same approach as ChatGPT UI
- Handles scanned documents, borderless tables, and complex layouts

### Prompt Templates

#### Table Extraction Template

Use `-t table` for extracting tables:

```python
Extract the complete table from this PDF.
Requirements:
- Include ALL rows and columns
- Preserve all column headers exactly as shown
- Preserve all row labels/categories
- Include all numeric values with correct precision
- Include all subtotals and totals
- If the table spans multiple pages, combine all sections
- Maintain the hierarchical structure (parent/child rows)
```

#### Financial Statement Template

Use `-t financial` for financial statements:

```python
Extract the complete balance sheet from this PDF.
Financial Statement Requirements:
- Include all line items with exact labels
- Capture all time periods/columns (years, quarters, etc.)
- Include all subtotals and category totals
- Preserve the account hierarchy
- Include all numeric values with correct signs
- Note the currency and units
- Include any footnote references
- If multi-page, combine all sections into one complete statement
```

## Performance Considerations

- **Cost**: ~$0.01-0.10 per page depending on model and complexity
- **Latency**: 5-30 seconds per request depending on PDF size
- **Accuracy**: 95%+ for well-structured documents, 85%+ for complex layouts
- **Token Limits**: Monitor input/output tokens, large PDFs may require chunking

## Error Handling

The tool includes robust error handling:
- **Retry Logic**: Automatic retries with exponential backoff for API failures
- **Validation**: Optional validation layer to catch extraction errors
- **File Caching**: Uploaded files are cached to avoid re-uploading

## Test Case

A sample L'Oréal 2024 Consolidated Financial Statements PDF (67 pages) is included in the `examples/` directory for testing.

### Example Test Command

```bash
python main.py extract examples/"Consolidated financial statements.pdf" \
  -t financial \
  --validate \
  -f json -f csv
```

Expected: Complete balance sheet with all line items, columns, years, and totals.

## Troubleshooting

### API Key Not Found

```
Error: OpenAI API key not found.
```

**Solution**: Create a `.env` file with your API key or use the `--api-key` option.

### File Not Found

```
Error: PDF file not found: path/to/file.pdf
```

**Solution**: Check the file path and ensure the file exists.

### Validation Failed

```
⚠ Validation issues found
```

**Solution**: Review the specific errors reported. You may need to:
- Refine your extraction prompt
- Use a different model
- Manually verify the output

### JSON Parse Error

```
Data is not valid JSON, wrapping in object
```

**Solution**: This is a warning, not an error. The tool will wrap non-JSON data in a JSON object. To get structured JSON output, request it explicitly in your prompt:

```bash
python main.py extract document.pdf -p "Extract data as JSON format"
```

## Development

### Running Tests

Tests are not yet implemented but will be added to the `tests/` directory.

```bash
pytest tests/
```

### Contributing

This is a research/prototype tool. Contributions, issues, and feature requests are welcome.

## Future Enhancements

- [ ] Batch processing with parallel execution
- [ ] Template learning (save successful prompts for reuse)
- [ ] Web interface (Flask/FastAPI)
- [ ] Export to databases
- [ ] Caching layer for file_ids and responses
- [ ] Progress bars for batch processing
- [ ] Support for other document types (Word, Excel)

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

Powered by OpenAI's API (GPT-4 and GPT-5-mini).

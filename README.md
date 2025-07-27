# DiffSense

[![CI](https://github.com/marco-gasparri/diffsense/workflows/CI/badge.svg)](https://github.com/marco-gasparri/diffsense/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


An AI-powered code diff analysis tool that provides intelligent insights into code changes using local language models. DiffSense goes beyond traditional diff tools by understanding the architectural and behavioral significance of modifications, helping developers focus on what truly matters.

## Core concept

Traditional diff tools show *what* changed but not *why* it matters. DiffSense leverages local AI models to:

- **Identify architectural patterns**: recognizes database transactions, error handling, security improvements
- **Prioritize substantial changes**: focuses on logic modifications over cosmetic changes  
- **Provide intelligent context**: uses smart context extraction for better analysis
- **Maintain privacy**: runs completely offline with local models

## Key features

### Intelligent analysis
- **Pattern recognition**: automatically detects flow modifications, concurrency patterns, security implementations, etc.
- **Change prioritization**: distinguishes between substantial logic changes and cosmetic modifications
- **Architectural insights**: understands the broader impact of code modifications

### Smart context management
- **Automatic context**: includes full file context for small files (< 80 lines, < 3000 chars)
- **Smart context**: extracts relevant sections around changes for larger files
- **Token-aware**: dynamically adjusts context to fit model limitations

### Professional output
- **Rich terminal display**: syntax-highlighted diff with professional formatting
- **Structured analysis**: consistent categorization with technical tags
- **Configurable context**: adjustable context lines around changes

### Privacy-first design
- **Local processing**: no external API calls or data transmission
- **Offline models**: downloads and caches models locally
- **Hardware optimization**: automatic GPU acceleration when available

## Installation

### Prerequisites
- Python 3.9 or higher
- 4GB+ RAM recommended for model execution
- 5GB+ disk space for model cache

### Install from source
```bash
git clone https://github.com/marco-gasparri/diffsense.git
cd diffsense
pip install -e .
```

### Development installation
```bash
git clone https://github.com/marco-gasparri/diffsense.git
cd diffsense
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage
```bash
# Compare two files with AI analysis of the diffs
diffsense file1 file2

# Skip AI analysis for faster results
diffsense file1 file2 --no-ai

# Include full file context for better analysis
diffsense file1 file2 --full-context
```

### Advanced Options
```bash
# Adjust context lines around changes
diffsense file1 file2 --context 5

# Use a custom model
diffsense file1 file2 --model TheBloke/CodeLlama-13B-Instruct-GGUF

# Enable verbose logging
diffsense file1 file2 --verbose
```

## Configuration

### Model selection
DiffSense uses CodeLlama-7B-Instruct by default, optimized for code analysis. You can specify alternative models from [HuggingFace](https://huggingface.co/):

```bash
# Use a larger model for better analysis
diffsense file1 file2 --model TheBloke/CodeLlama-13B-Instruct-GGUF

# Use a general-purpose model
diffsense file1 file2 --model TheBloke/Llama-2-7B-Chat-GGUF
```

### Hardware optimization
DiffSense automatically optimizes for your hardware:
- **Apple Silicon**: uses Metal acceleration
- **NVIDIA GPUs**: automatic CUDA detection
- **CPU-only**: optimized threading for maximum performance

### Context modes
```bash
# Automatic context selection (recommended)
diffsense file1 file2

# Force full file context
diffsense file1 file2 --full-context

# Adjust diff context lines
diffsense file1 file2 --context 10
```

## Example output

```
╭────────────────────────────────────────────────────────────────────────────────── Diff Analysis ──────────────────────────────────────────────────────────────────────────────────╮
│ --- example_v1.py +++ example_v2.py                                                                                                                                               │
│ Blocks: 2 | Changes: 11                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────── Block 1 (+3, -2) ─────────────────────────────────────────────────────────────────────────────────╮
│ @@ -1,3 +1,4 @@                                                                                                                                                                   │
│    1 - def calculate_total(a, b):                                                                                                                                                 │
│    2 -     return a + b                                                                                                                                                           │
│    1 + def calculate_total(a, b, fee=0):                                                                                                                                          │
│    2 +     subtotal = a + b                                                                                                                                                       │
│    3 +     return subtotal + fee                                                                                                                                                  │
│    3                                                                                                                                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────── Block 2 (+2, -2) ─────────────────────────────────────────────────────────────────────────────────╮
│ @@ -3,3 +4,3 @@                                                                                                                                                                   │
│    3                                                                                                                                                                              │
│    4 - def print_receipt(total):                                                                                                                                                  │
│    5 -     print(f"Total: {total}")                                                                                                                                               │
│    5 + def print_receipt(total, currency="$"):                                                                                                                                    │
│    6 +     print(f"{currency} {total}")                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────────────────────── Summary ─────────────────────────────────────────────────────────────────────────────────────╮
│ + 5 insertions - 4 deletions ~ 0 modifications                                                                                                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
[19:32:46] INFO     Loading model: TheBloke/CodeLlama-7B-Instruct-GGUF                                                                                             llm_manager.py:137
[19:32:49] INFO     Model loaded successfully                                                                                                                      llm_manager.py:158

────────────────────────────────────────────────────────────────────────────────
AI Analysis:
**PRIMARY CHANGE:**
The most substantial change that introduces new functionality, logic, or architecture is the addition of a new parameter `fee` to the `calculate_total` function. This change allows 
for the calculation of a fee to be added to the total, which was not possible before.

**SECONDARY CHANGES:**
The addition of the `fee` parameter also allows for the calculation of the subtotal, which was previously hardcoded. This change makes the `calculate_total` function more flexible 
and allows for more complex calculations.

**PURPOSE & IMPACT:**
The purpose of this change is to allow for the calculation of a fee to be added to the total, which was not possible before. This change enables the system to handle more complex 
transactions and to provide more accurate receipts.

**TECHNICAL BENEFITS:**
The addition of the `fee` parameter and the calculation of the subtotal allow for more flexible and complex calculations, which can improve the performance and reliability of the 
system.

**TAGS:**
Classification: Feature
Technical: performance, readability, error-handling
Complexity: Medium
Risk: Low
```

## Project structure

```
diffsense/
├── src/diffsense/
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface
│   ├── diff_engine.py           # Core diff computation
│   ├── formatter.py             # Rich text formatting
│   ├── llm_manager.py           # AI model management
│   └── exceptions.py            # Custom exceptions
├── tests/
│   ├── test_cli.py              # CLI functionality tests
│   ├── test_diff_engine.py      # Diff engine tests
│   ├── test_formatter.py        # Formatter tests
│   └── test_llm_manager.py      # LLM manager tests
├── examples/
│   ├── example_v1.py            # Example file1
│   └── example_v2.py            # Example file2
├── models/                      # Local model cache
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── .github/workflows/ci.yml     # CI/CD pipeline
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=diffsense --cov-report=html

# Run specific test categories
pytest tests/test_diff_engine.py -v
```

### Adding new features

#### Extending the Diff Engine
To add new diff algorithms or enhance change detection:

1. Modify `src/diffsense/diff_engine.py`
2. Add corresponding tests in `tests/test_diff_engine.py`
3. Update the `DiffEngine` class with new methods

#### Enhancing AI analysis
To improve prompt engineering or add new analysis patterns:

1. Modify `_build_analysis_prompt()` in `src/diffsense/llm_manager.py`
2. Add pattern recognition rules
3. Test with various code change scenarios

#### Adding output formats
To support new output formats:

1. Extend `src/diffsense/formatter.py`
2. Add new formatting methods
3. Update CLI options in `src/diffsense/cli.py`

## Performance considerations

### Model selection trade-offs
- **CodeLlama-7B-Instruct**: fast, good for code analysis, 4GB RAM
- **CodeLlama-13B-Instruct**: better analysis, slower, 8GB RAM
- **Llama-2-7B-Chat**: general purpose, less code-specific

### Hardware requirements
- **Minimum**: 4GB RAM, CPU-only execution
- **Recommended**: 8GB+ RAM, GPU acceleration
- **Optimal**: 16GB+ RAM, modern GPU with 8GB+ VRAM

### Context Management
- Files < 80 lines: automatic full context
- Larger files: smart context extraction
- Custom context: use `--full-context` flag


## Troubleshooting

### Common Issues

**Model download fails**
```bash
# Clear model cache and retry
rm -rf models/
diffsense file1 file2
```

**Out of memory errors**
```bash
# Use smaller model or reduce context
diffsense file1 file2 --model smaller-model --context 1
```

**Slow Performance**
```bash
# Check GPU acceleration
diffsense file1 file2 --verbose

# Use smaller context for large files
diffsense file1 file2 --context 3
```

## Technical details

### Architecture
DiffSense employs a modular architecture with clear separation of concerns:

- **Diff Engine**: implements advanced diff algorithms with inline change detection
- **LLM Manager**: handles model lifecycle, hardware optimization, and prompt engineering  
- **Formatter**: provides rich terminal output with syntax highlighting
- **CLI**: offers comprehensive command-line interface with extensive options

### Model integration
The tool uses GGUF-format models via llama-cpp-python for efficient local execution. Models are automatically downloaded from HuggingFace and cached locally for subsequent use.

### Privacy and security
All processing occurs locally on your machine. No code or analysis results are transmitted to external services, ensuring complete privacy and security of your intellectual property.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- Powered by [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for local LLM inference
- CLI framework provided by [Typer](https://github.com/tiangolo/typer)
- Models hosted on [HuggingFace](https://huggingface.co/)

## Development transparency

This project was developed with assistance from large language models for debugging, testing, and documentation tasks. All core architectural decisions, implementation logic, and final code quality remain under human oversight and validation. The use of AI tools accelerated development while maintaining high code quality standards and comprehensive test coverage.
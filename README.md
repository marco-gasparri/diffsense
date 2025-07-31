# DiffSense

[![CI](https://github.com/marco-gasparri/diffsense/workflows/CI/badge.svg)](https://github.com/marco-gasparri/diffsense/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


An AI-powered code diff analysis tool that provides intelligent insights into code changes using local language models. DiffSense goes beyond traditional diff tools by understanding the architectural and behavioral significance of modifications, helping developers focus on what truly matters. It provides also a Git integration for analyzing commits and working directory changes.

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

### Git integration
- **Commit comparison**: analyze changes between any two commits
- **Working directory diffs**: compare uncommitted changes against any commit

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

### Git Integration
```bash

# Basic syntax
diffsense --git <ref1> [<ref2>] [<file>]

# Compare working directory changes against HEAD
diffsense --git HEAD src/file.py

# Compare specific file between commits
diffsense --git main feature-branch src/file.py

# Compare working directory against specific commit
diffsense --git HEAD~1 src/file.py

# Compare file between commits
diffsense --git abc123 def456 src/file.py   

# Compare file between branches
diffsense --git main dev src/file.py  

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

```bash
$ diffsense examples/example_v1.py examples/example_v2.py
```

<pre>
┌──────────────────────── <b>Diff Analysis</b> ─────────────────────────┐
│ <span style="color:#e74c3c"><b>---</b></span> example_v1.py <span style="color:#27ae60"><b>+++</b></span> example_v2.py                            │
│ Blocks: 2 | Changes: 11                                        │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────── <b>Block 1</b> (<span style="color:#27ae60">+3</span>, <span style="color:#e74c3c">-2</span>) ───────────────────────┐
│ <span style="color:#3498db"><b>@@ -1,3 +1,4 @@</b></span>                                                │
│   <span style="color:#95a5a6">1</span> <span style="color:#e74c3c">- def calculate_total(a, b):</span>                               │
│   <span style="color:#95a5a6">2</span> <span style="color:#e74c3c">-     return a + b</span>                                         │
│   <span style="color:#95a5a6">1</span> <span style="color:#27ae60">+ def calculate_total(a, b, fee=0):</span>                        │
│   <span style="color:#95a5a6">2</span> <span style="color:#27ae60">+     subtotal = a + b</span>                                     │
│   <span style="color:#95a5a6">3</span> <span style="color:#27ae60">+     return subtotal + fee</span>                                │
│   <span style="color:#95a5a6">3</span>                                                            │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────── <b>Block 2</b> (<span style="color:#27ae60">+2</span>, <span style="color:#e74c3c">-2</span>) ───────────────────────┐
│ <span style="color:#3498db"><b>@@ -3,3 +4,3 @@</b></span>                                                │
│   <span style="color:#95a5a6">4</span>                                                            │
│   <span style="color:#95a5a6">5</span> <span style="color:#e74c3c">- def print_receipt(total):</span>                                │
│   <span style="color:#95a5a6">6</span> <span style="color:#e74c3c">-     print(f"Total: {total}")</span>                             │
│   <span style="color:#95a5a6">5</span> <span style="color:#27ae60">+ def print_receipt(total, currency="$"):</span>                  │
│   <span style="color:#95a5a6">6</span> <span style="color:#27ae60">+     print(f"{currency} {total}")</span>                         │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────── <b>Summary</b> ───────────────────────────┐
│ <span style="color:#27ae60">+ 3 insertions</span> <span style="color:#e74c3c">- 2 deletions</span> <span style="color:#f39c12">~ 0 modifications</span>                │
└───────────────────────────────────────────────────────────────┘

<span style="color:#3498db">[19:32:46]</span> <span style="color:#2ecc71"><b>INFO</b></span>    Loading model: TheBloke/CodeLlama-7B-Instruct-GGUF
<span style="color:#3498db">[19:32:49]</span> <span style="color:#2ecc71"><b>INFO</b></span>    Model loaded successfully

────────────────────────────────────────────────────────────────────
<span style="color:#f1c40f"><b>AI Analysis:</b></span>

<b>PRIMARY CHANGE:</b>
The most substantial change that introduces new functionality,
logic, or architecture is the addition of a new parameter `fee`
to the `calculate_total` function. This change allows for the 
calculation of a fee to be added to the total, which was not
possible before.

<b>SECONDARY CHANGES:</b>
The addition of the `fee` parameter also allows for the
calculation of taxes and other fees that may be applicable to a
transaction. Additionally, the `print_receipt` function now
takes an additional parameter `currency` to specify the currency
symbol to be used in the receipt.

<b>PURPOSE & IMPACT:</b>
The purpose of these changes is to enable the calculation of
fees and taxes in the receipt, and to allow for the use of
different currencies in the receipt.

<b>TECHNICAL BENEFITS:</b>
The addition of the `fee` parameter and the `print_receipt`
function with the `currency` parameter allow for more flexible
and customizable receipts, which can be useful for different
types of transactions and businesses.

<b>SUMMARY:</b>
Classification: Feature
Tag: Generic
Complexity: Medium
Risk: Medium
</pre>

## Project structure

```
diffsense/
├── src/diffsense/
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface
│   ├── diff_engine.py           # Core diff computation
│   ├── formatter.py             # Rich text formatting
│   ├── git_manager.py           # Git integration
│   ├── llm_manager.py           # AI model management
│   └── exceptions.py            # Custom exceptions
├── tests/
│   ├── test_cli.py              # CLI functionality tests
│   ├── test_cli_git.py          # Git mode CLI tests
│   ├── test_diff_engine.py      # Diff engine tests
│   ├── test_formatter.py        # Formatter tests
│   ├── test_git_manager.py      # Git manager tests
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

#### Extending Git integration
To add new Git features:

1. Modify `src/diffsense/git_manager.py`
2. Add tests in `tests/test_git_manager.py`
3. Update CLI in `src/diffsense/cli.py` if needed

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
- **Git Manager**: provides Git integration to retrieve file diffs from commits data
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
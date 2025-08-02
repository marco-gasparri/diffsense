# DiffSense

[![CI](https://github.com/marco-gasparri/diffsense/workflows/CI/badge.svg)](https://github.com/marco-gasparri/diffsense/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered code diff analysis tool that understands the *why* behind code changes. DiffSense uses AI models (local or cloud-based) to provide intelligent insights, helping developers focus on what truly matters in code reviews.

## Why DiffSense?

Traditional diff tools show *what* changed but not *why* it matters. DiffSense:
- **Identifies architectural patterns** and their impact
- **Prioritizes substantial changes** over cosmetic ones
- **Provides context-aware analysis** using AI
- **Supports both local and cloud models** for flexibility

![DiffSense screenshot](/docs/screen.jpg?raw=true "DiffSense screenshot")

## Quick Start

```bash
# Install
pip install -e .

# Basic usage
diffsense file1 file2

# Git integration
diffsense --git HEAD~1 file

# Use Anthropic API
export DIFFSENSE_ANTHROPIC_API_KEY="your-key"
diffsense file1 file2 --model anthropic

# Use OpenAI API
export DIFFSENSE_OPENAI_API_KEY="your-api-key"
diffsense file1 file2 --model openai
```

## Features

- **AI Analysis**: understands code changes using LLMs (local or cloud)
- **Git Integration**: analyze commits and working directory changes
- **Privacy Options**: choose between local models (offline) or cloud APIs
- **Smart Context**: automatically manages context for optimal analysis

## Installation

```bash
# Basic installation
git clone https://github.com/marco-gasparri/diffsense.git
cd diffsense
pip install -e .

# With cloud model support
pip install -e ".[remote]"

# Development setup
pip install -e ".[dev]"
```

## Usage Examples

### File Comparison
```bash
# Compare two files
diffsense file1 file2

# Skip AI analysis
diffsense file1 file2 --no-ai

# More context lines
diffsense file1 file2 --context 10
```

### Git Integration
```bash
# Compare with previous commit
diffsense --git HEAD~1 file

# Compare branches
diffsense --git main feature-branch file

# List changed files
diffsense --git HEAD
```

### Model selection: local models (default)

DiffSense uses CodeLlama-7B-Instruct by default. Models are downloaded automatically and cached locally in `./models` directory.

```bash
# Default model
diffsense file1 file2

# Specific HuggingFace model
diffsense file1 file2 --model TheBloke/CodeLlama-13B-Instruct-GGUF

# General purpose model
diffsense file1 file2 --model TheBloke/Llama-2-7B-Chat-GGUF
```

**Requirements**:
- 4GB+ RAM for 7B models
- 8GB+ RAM for 13B models
- GPU acceleration automatic on Apple Silicon and CUDA-enabled systems

### Model selection: cloud models

For enhanced analysis with cloud-based models:

#### Anthropic
```bash
export DIFFSENSE_ANTHROPIC_API_KEY="your-api-key"
diffsense file1 file2 --model anthropic
```
Uses Claude Opus 4, the best-in-class Anthropic model.

#### OpenAI
```bash
export DIFFSENSE_OPENAI_API_KEY="your-api-key"
diffsense file1 file2 --model openai
```
Uses GPT-4o, the most advanced reasoning OpenAI model.

**Note**: Cloud models require `pip install diffsense[remote]`

## Command Line Options

### Basic Options
- `--no-ai` - Skip AI analysis, show only diff
- `--context N` - Number of context lines (default: 3)
- `--verbose` - Enable debug logging
- `--version` - Show version

### Git Mode
- `--git` - Enable Git mode for repository diffs

### Model Options
- `--model MODEL` - Choose AI model (local path or "anthropic"/"openai")
- `--full-context` - Force full file context in analysis

## Context Management

DiffSense automatically optimizes context based on file size:

- **Small files** (<100 lines): Full context included
- **Large files**: Smart context extraction around changes
- **Force full**: Use `--full-context` flag

## Performance Tuning

### Local Models

The tool auto-detects hardware and optimizes accordingly:

| Platform | Optimization |
|----------|-------------|
| Apple Silicon | Metal GPU acceleration |
| Linux + NVIDIA | CUDA acceleration |
| Windows/Other | Optimized CPU threading |

### Memory Management

- **Low memory** (<8GB): Reduced context window
- **High memory** (>16GB): Model locked in RAM
- **Model cache**: Default `./models/` directory

### API Rate Limits

Cloud models are subject to provider rate limits:
- Anthropic: varies by plan
- OpenAI: varies by tier

### Token consumption

In addition to the defined prompt, the token consumption is dependent on the size of the diff, especially in case of `--full-context` usage

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DIFFSENSE_ANTHROPIC_API_KEY` | Anthropic Anthropic API key |
| `DIFFSENSE_OPENAI_API_KEY` | OpenAI API key |


## Privacy & Security

- **Local models**: all processing happens on your machine. No data leaves your system.
- **Cloud models**: <ins>when using `--model anthropic` or `--model openai`, diffs are sent to respective APIs. Use according to your data policies.</ins> The default hardcoded models are `gpt-4o` (for OpenAI) and `claude-opus-4-20250514` (for Anthropic), at the moment the most advanced models to exploit the best-in-class analyzing capabilities.


## Development Transparency

This project was developed with AI assistance throughout the entire development process. DiffSense was born from both a practical need for better code review tools and the curiosity to see AI in action while developing a complete project from scratch. The AI served as a coding assistant, helping with implementation, testing, and documentation while all architectural decisions and code quality remained under human oversight.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Rich](https://github.com/Textualize/rich) for terminal output
- Local models via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- CLI powered by [Typer](https://github.com/tiangolo/typer)
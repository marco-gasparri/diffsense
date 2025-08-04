# DiffSense

[![CI](https://github.com/marco-gasparri/diffsense/workflows/CI/badge.svg)](https://github.com/marco-gasparri/diffsense/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered code diff analysis tool that understands the *why* behind code changes. DiffSense uses AI models (local or cloud-based) to provide intelligent insights, helping developers focus on what truly matters in code reviews and conflict resolution.

## Why DiffSense?

Traditional diff tools show *what* changed but not *why* it matters. DiffSense:
- **Identifies architectural patterns** and their impact
- **Prioritizes substantial changes** over cosmetic ones
- **Provides context-aware analysis** using AI
- **Resolves merge conflicts intelligently** with explanations
- **Supports both local and cloud models** for flexibility

## Quick Start

```bash
# Install
pip install -e .

# Basic usage
diffsense file1 file2

# Git integration
diffsense --git HEAD~1 file

# Resolve merge conflicts
diffsense --resolve-conflicts conflicted_file 

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
- **Conflict Resolution**: AI-powered merge conflict resolution with confidence levels
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

### Merge Conflict Resolution
```bash
# Resolve conflicts in a single file
diffsense --resolve-conflicts conflicted_file

# With additional context files
diffsense --resolve-conflicts conflicted_file.py --context-file models.py --context-file utils.py

# Using a specific model
diffsense --resolve-conflicts conflicted_file --model anthropic
```

The conflict resolver:
- analyzes both versions of conflicted code
- provides resolution with confidence levels (HIGH/MEDIUM/LOW)
- suggests alternatives when confidence is not high
- adds explanatory comments in the resolved code
- shows token usage for cost tracking
- can apply changes to the original file creating automatic backups

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

### Conflict Resolution
- `--resolve-conflicts` - Resolve merge conflicts using AI
- `--context-file PATH` / `-f PATH` - Additional files for context (can be used multiple times)

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

Token usage is reported for all operations:
- Diff analysis: typically 200-1000 tokens per analysis
- Conflict resolution: 300-2000 tokens per conflict
- Additional context files increase token usage

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DIFFSENSE_ANTHROPIC_API_KEY` | Anthropic API key |
| `DIFFSENSE_OPENAI_API_KEY` | OpenAI API key |

## Privacy & Security

- **Local models**: all processing happens on your machine. No data leaves your system.
- **Cloud models**: <ins>when using `--model anthropic` or `--model openai`, diffs are sent to respective APIs. Use according to your data policies.</ins> 

## Default models
The default hardcoded models:
- `TheBloke/CodeLlama-7B-Instruct-GGUF` as default model
- `gpt-4o` for the option `--model openai`
- `claude-opus-4-20250514` for the option `--model anthropic`

Suggestion: if there are ~30GB of available RAM, use `TheBloke/Phind-CodeLlama-34B-v2-GGUF` instead of the default model

## Development Transparency

This project was developed with AI assistance throughout the entire development process. DiffSense was born from both a practical need and the curiosity to see AI in action while developing a complete project from scratch. The AI served as a coding assistant, helping with implementation, testing, and documentation while all architectural decisions, features design and code quality remained under human oversight.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Rich](https://github.com/Textualize/rich) for terminal output
- Local models via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- CLI powered by [Typer](https://github.com/tiangolo/typer)

## Screenshots

### Diff analysis

![DiffSense diff screenshot](/docs/screen.jpg?raw=true "DiffSense diff screenshot")

### Merge conflict resolution

![DiffSense conflict resolution screenshot](/docs/screen-conflict.jpg?raw=true "DiffSense conflict resolution screenshot")

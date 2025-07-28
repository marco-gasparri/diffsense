"""
Local LLM management for intelligent diff analysis
Downloading, caching, and running LLMs
"""

import os
import platform
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import contextlib
import io

import psutil
from huggingface_hub import hf_hub_download, HfApi
from llama_cpp import Llama

from .diff_engine import DiffBlock
from .formatter import DiffFormatter
from .exceptions import ModelError

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages local LLM models for diff analysis
    """

    # Default model configuration
    DEFAULT_MODEL = "TheBloke/CodeLlama-7B-Instruct-GGUF"
    DEFAULT_FILENAME = "codellama-7b-instruct.Q4_K_M.gguf"

    def __init__(
        self,
        model_id: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3 # Magic!
    ):
        """
        Initialize the LLM manager
        Args:
            model_id: huggingFace model ID (uses default if None)
            cache_dir: directory for model cache (uses ./models if None)
            max_tokens: maximum tokens for generation
            temperature: sampling temperature for generation
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        self.cache_dir = cache_dir or Path("models")
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.cache_dir.mkdir(exist_ok=True)
        self._model_instance: Optional[Llama] = None
        self._formatter = DiffFormatter()

    def analyze_diff(
        self,
        blocks: List[DiffBlock],
        original_content: Optional[str] = None,
        modified_content: Optional[str] = None,
        file1_name: Optional[str] = None,
        file2_name: Optional[str] = None
    ) -> str:
        """
        Generate an intelligent analysis of diff blocks
        Returns AI-generated analysis of the changes
        """
        if not blocks:
            return "No changes detected in the provided files"

        try:
            # Load model if not already loaded
            if self._model_instance is None:
                self._model_instance = self._load_model()

            # Generate prompt with optional context
            prompt = self._build_analysis_prompt(
                blocks,
                original_content=original_content,
                modified_content=modified_content,
                file1_name=file1_name,
                file2_name=file2_name
            )

            # Run inference
            response = self._run_inference(prompt)

            return response

        except Exception as e:
            raise ModelError(f"Failed to analyze diff: {e}") from e

    def _estimate_token_count(self, text: str) -> int:
        """
        Rough estimation of token count for a text
        Uses approximate ratio: 1 token ≈ 4 characters for English text (needs improvements)
        """
        return len(text) // 4  # Conservative estimate

    def _load_model(self) -> Llama:
        """
        Load and initialize the LLM model.
        Returns the initialized Llama model instance
        """
        try:
            logger.info(f"Loading model: {self.model_id}")

            # Determine filename
            if self.model_id == self.DEFAULT_MODEL:
                filename = self.DEFAULT_FILENAME
            else:
                # For custom models, try to find GGUF files
                filename = self._find_model_filename()

            # Download model if not cached
            model_path = self._download_model(filename)

            # Get optimized parameters for current hardware
            model_kwargs = self._get_model_parameters()

            # Initialize model with suppressed output
            logger.debug("Initializing model with optimized parameters")
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                model = Llama(model_path=str(model_path), **model_kwargs)

            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            raise ModelError(f"Failed to load model {self.model_id}: {e}") from e

    def _find_model_filename(self) -> str:
        """
        Find the appropriate GGUF filename for a model if exists
        """
        try:
            api = HfApi()
            model_info = api.model_info(repo_id=self.model_id)
            gguf_files = [f.rfilename for f in model_info.siblings if f.rfilename.endswith(".gguf")]

            if not gguf_files:
                raise ModelError(f"No GGUF files found in model {self.model_id}")

            # Prefer Q4_K_M quantization for balance of speed and quality
            preferred_quantizations = ['Q4_K_M', 'Q4_0', 'Q5_K_M', 'Q8_0']

            for quant in preferred_quantizations:
                for filename in gguf_files:
                    if quant in filename:
                        return filename

            # If no preferred quantization found, use the first GGUF file
            logger.warning(f"No preferred quantization found, using: {gguf_files[0]}")
            return gguf_files[0]

        except Exception as e:
            raise ModelError(f"Failed to find model filename: {e}") from e

    def _download_model(self, filename: str) -> Path:
        """
        Download model file if not already cached
        Returns the path of the downloaded model file
        """
        try:
            logger.debug(f"Downloading model file: {filename}")

            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename=filename,
                local_dir=str(self.cache_dir),
            )

            logger.debug(f"Model cached at: {model_path}")
            return Path(model_path)

        except Exception as e:
            raise ModelError(f"Failed to download model: {e}") from e

    def _get_model_parameters(self) -> Dict[str, Any]:
        """
        Get optimized model parameters based on system hardware
        Returns a dictionary of model parameters
        """
        # Detect system capabilities
        system = platform.system().lower()
        arch = platform.machine().lower()
        cpu_count = psutil.cpu_count(logical=False) or 4
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)

        # Base parameters
        params = {
            "n_ctx": 5120,
            "n_threads": min(cpu_count, 8),
            "use_mlock": memory_gb > 16,  # Lock model in memory if enough RAM
            "verbose": False,
            "n_batch": 512,
        }

        # Platform-specific optimizations
        if system == "darwin" and "arm" in arch:
            # Apple Silicon optimizations
            params.update({
                "n_gpu_layers": 35,
                "metal": True, # Use Metal for GPU acceleration
            })
            logger.debug("Configured for Apple Silicon with Metal acceleration")

        elif system == "linux":
            # Linux optimizations
            if self._cuda_available():
                params.update({
                    "n_gpu_layers": 35,
                })
                logger.debug("Configured for Linux with CUDA acceleration")
            else:
                logger.debug("Configured for Linux CPU-only")

        else:
            # Windows and other systems
            logger.debug("Configured for CPU-only execution")

        # Memory-based adjustments
        if memory_gb < 8:
            params["n_ctx"] = 2048  # Reduce context for low-memory systems
            params["n_batch"] = 256
            logger.debug("Reduced parameters for low-memory system")

        return params

    def _cuda_available(self) -> bool:
        """
        Check if CUDA is available for GPU acceleration
        """
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _extract_smart_context(self, content: str, blocks: List[DiffBlock], context_lines: int = 10) -> str:
        """
        Extract relevant context around changes instead of full file content
        Args:
            content: Full file content
            blocks: Diff blocks to get context around
            context_lines: Number of lines to include around each change
        Returns:
            Relevant context snippets
        """
        lines = content.splitlines()
        relevant_lines = set()

        for block in blocks:
            # Get the range of lines this block affects
            start_line = max(0, block.old_start - context_lines - 1)  # Convert to 0-based
            end_line = min(len(lines), block.old_start + block.old_count + context_lines - 1)

            for line_num in range(start_line, end_line):
                relevant_lines.add(line_num)

        # Build context with line numbers
        context_parts = []
        sorted_lines = sorted(relevant_lines)

        current_group = []
        for line_num in sorted_lines:
            if not current_group or line_num == current_group[-1] + 1:
                current_group.append(line_num)
            else:
                # Gap found, save current group and start new one
                if current_group:
                    context_parts.append(self._format_context_group(lines, current_group))
                current_group = [line_num]

        # Don't forget the last group
        if current_group:
            context_parts.append(self._format_context_group(lines, current_group))

        return "\n...\n".join(context_parts)

    def _format_context_group(self, lines: List[str], line_numbers: List[int]) -> str:
        """Format a group of consecutive lines with line numbers"""
        result = []
        for line_num in line_numbers:
            if line_num < len(lines):
                result.append(f"{line_num + 1:4d}: {lines[line_num]}")
        return "\n".join(result)
        """
        Rough estimation of token count for a text (len/4)
        """
        return len(text) // 4

    def _build_analysis_prompt(
        self,
        blocks: List[DiffBlock],
        original_content: Optional[str] = None,
        modified_content: Optional[str] = None,
        file1_name: Optional[str] = None,
        file2_name: Optional[str] = None
    ) -> str:
        """
        Build a comprehensive prompt for diff analysis
        """
        # Convert diff to plain text for analysis
        plain_diff = self._formatter.format_plain_diff(blocks)

        # Build context section if full files are provided
        if original_content and modified_content:
            # Check if full context would fit in model's context window
            context_text = f"{original_content}\n{modified_content}"
            estimated_tokens = self._estimate_token_count(context_text + plain_diff)
            max_tokens = 5000

            if estimated_tokens > max_tokens:
                # Use smart context instead of full files
                logger.debug(f"Full context too large ({estimated_tokens} tokens). Using smart context around changes.")
                smart_context_original = self._extract_smart_context(original_content, blocks, context_lines=8)
                smart_context_modified = self._extract_smart_context(modified_content, blocks, context_lines=8)

                context_section = f"""RELEVANT CONTEXT (around changes):

--- Original File ({file1_name or 'file1'}) - Key Sections ---
{smart_context_original}

--- Modified File ({file2_name or 'file2'}) - Key Sections ---
{smart_context_modified}

SPECIFIC CHANGES:"""
                logger.debug("Using smart context around changes")
            else:
                context_section = f"""FULL FILE CONTEXT:

--- Original File ({file1_name or 'file1'}) ---
{original_content}

--- Modified File ({file2_name or 'file2'}) ---
{modified_content}

SPECIFIC CHANGES:"""
                logger.debug(f"Including full file context ({estimated_tokens} estimated tokens)")
        else:
            context_section = "CHANGES TO ANALYZE:"

        prompt = f"""CRITICAL: Focus ONLY on SUBSTANTIAL TECHNICAL CHANGES that alter system behavior or capabilities. Ignore superficial modifications.

{context_section}
{plain_diff}

BEFORE WRITING - IDENTIFY WHAT REALLY CHANGED:
1. If you see the same function/variable/class in both deletions (-) and additions (+), it was RENAMED or MOVED - this is NOT important
2. Focus ONLY on genuinely NEW code, NEW logic, NEW architectural patterns
3. Ignore: renames, moves, import changes, formatting, parameter reordering if they are not the only content of the diff

**PRIMARY CHANGE:**
What is the MOST SUBSTANTIAL change that introduces NEW functionality, logic, or architecture? Skip renames, moves, and cosmetic changes entirely.

**SECONDARY CHANGES:**
Only mention changes that introduce genuinely new capabilities or significantly alter existing behavior. Write in flowing paragraphs, not lists.

**PURPOSE & IMPACT:**
Why were these substantial changes made? What new capabilities or behaviors do they enable?

**TECHNICAL BENEFITS:**
Only if there are genuine architectural improvements (performance, reliability, security). Skip if changes are merely organizational.

**SUMMARY:**
Classification: [Bug Fix | Feature | Refactoring | Maintenance]
Tag: [Database | Readability | Typo | Error-handling | Performance | Security | Concurrency | Logging | Api-changes | Generic][USE ONLY ONE TAG, use 'Generic' if not sure]
Complexity: [Low | Medium | High]
Risk: [Low | Medium | High]

MANDATORY RULES:
- NEVER mention renames, moves, or parameter reordering as important changes
- Focus on NEW code that didn't exist before, not reorganized existing code  
- If most changes are cosmetic/organizational, say so directly
- Write in paragraphs, absolutely NO bullet points or lists"""

        return prompt

    def _run_inference(self, prompt: str) -> str:
        """
        Run inference on the loaded model
        """
        try:
            if self._model_instance is None:
                raise ModelError("Model not loaded")

            logger.debug("Running model inference")

            # Create chat completion
            response = self._model_instance.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior software engineer providing concise, professional code review analysis"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["Human:", "Assistant:", "###"],  # Stop sequences
            )

            # Extract response text
            if "choices" not in response or not response["choices"]:
                raise ModelError("Empty response from model")

            content = response["choices"][0]["message"]["content"].strip()

            if not content:
                raise ModelError("Model returned empty content")

            logger.debug("Model inference completed successfully")
            return content

        except Exception as e:
            raise ModelError(f"Inference failed: {e}") from e

    def clear_cache(self) -> None:
        """Clear the model cache directory"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("Model cache cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently configured model
        Returns a dictionary with model information
        """
        return {
            "model_id": self.model_id,
            "cache_dir": str(self.cache_dir),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "loaded": self._model_instance is not None,
        }
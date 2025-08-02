"""
Model providers for different LLM backends
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import contextlib
import io

from .exceptions import ModelError

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """
    Abstract base class for model providers

    This defines the interface that all model providers must implement,
    allowing seamless switching between local and remote models.
    """

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> Tuple[str, Dict[str, int]]:
        """
        Generate text from messages

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns a tuple of (generated text, token usage dict with 'input' and 'output' keys)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and properly configured"""
        pass


class LocalModelProvider(ModelProvider):
    """
    Provider for local GGUF models using llama-cpp-python

    This provider maintains compatibility with the existing local model
    infrastructure, wrapping the Llama model instance.
    """

    def __init__(self, model_instance):
        """
        Initialize with a Llama model instance

        Args:
            model_instance: Loaded Llama model from llama-cpp-python
        """
        self.model_instance = model_instance

    def generate(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> Tuple[str, Dict[str, int]]:
        """Generate using local model via llama-cpp-python"""
        if self.model_instance is None:
            raise ModelError("Model not loaded")

        logger.debug("Running local model inference")

        try:
            # Use the same chat completion API as the original
            response = self.model_instance.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["Human:", "Assistant:", "###"],  # Original stop sequences
            )

            # Extract response text (same as original)
            if "choices" not in response or not response["choices"]:
                raise ModelError("Empty response from model")

            content = response["choices"][0]["message"]["content"].strip()

            if not content:
                raise ModelError("Model returned empty content")

            # Extract token usage from response
            token_usage = {
                "input": response.get("usage", {}).get("prompt_tokens", 0),
                "output": response.get("usage", {}).get("completion_tokens", 0)
            }

            logger.debug("Local model inference completed successfully")
            return content, token_usage

        except Exception as e:
            raise ModelError(f"Local inference failed: {e}") from e

    def is_available(self) -> bool:
        """Check if local model is loaded"""
        return self.model_instance is not None


class AnthropicProvider(ModelProvider):
    """
    Provider for Anthropic API

    Uses the anthropic Python SDK to communicate with Anthropic API.
    Requires DIFFSENSE_ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self):
        """Initialize Anthropic provider with API key from environment"""
        self.api_key = os.environ.get("DIFFSENSE_ANTHROPIC_API_KEY")
        self._client = None

        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.debug("Anthropic provider initialized")
            except ImportError:
                logger.warning("anthropic package not installed. Install with: pip install anthropic")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> Tuple[str, Dict[str, int]]:
        """Generate using Anthropic API"""
        if not self._client:
            raise ModelError("Anthropic client not initialized. Check API key and anthropic package installation.")

        logger.debug("Calling Anthropic API")

        try:
            # Convert messages to Anthropic format
            # Extract system message if present
            system_message = None
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Call Anthropic API
            response = self._client.messages.create(
                model="claude-opus-4-20250514",  # Hardcoded Anthropic model
                messages=anthropic_messages,
                system=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.content[0].text.strip()

            if not content:
                raise ModelError("Anthropic returned empty content")

            # Extract token usage
            token_usage = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens
            }

            logger.debug("Anthropic API call completed successfully")
            return content, token_usage

        except Exception as e:
            raise ModelError(f"Anthropic API call failed: {e}") from e

    def is_available(self) -> bool:
        """Check if Anthropic provider is properly configured"""
        return bool(self.api_key and self._client)


class OpenAIProvider(ModelProvider):
    """
    Provider for OpenAI API

    Uses the openai Python SDK to communicate with OpenAI API.
    Requires DIFFSENSE_OPENAI_API_KEY environment variable.
    """

    def __init__(self):
        """Initialize OpenAI provider with API key from environment"""
        self.api_key = os.environ.get("DIFFSENSE_OPENAI_API_KEY")
        self._client = None

        if self.api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
                logger.debug("OpenAI provider initialized")
            except ImportError:
                logger.warning("openai package not installed. Install with: pip install openai")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> Tuple[str, Dict[str, int]]:
        """Generate using OpenAI API"""
        if not self._client:
            raise ModelError("OpenAI client not initialized. Check API key and openai package installation.")

        logger.debug("Calling OpenAI API")

        try:
            # Call OpenAI API
            response = self._client.chat.completions.create(
                model="gpt-4o",  # Hardcoded OpenAI model
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content.strip()

            if not content:
                raise ModelError("OpenAI returned empty content")

            # Extract token usage
            token_usage = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens
            }

            logger.debug("OpenAI API call completed successfully")
            return content, token_usage

        except Exception as e:
            raise ModelError(f"OpenAI API call failed: {e}") from e

    def is_available(self) -> bool:
        """Check if OpenAI provider is properly configured"""
        return bool(self.api_key and self._client)


def create_provider(model_id: str, model_instance=None) -> ModelProvider:
    """
    Factory function to create the appropriate model provider

    Args:
        model_id: Model identifier ("anthropic", "openai", or HuggingFace model ID)
        model_instance: For local models, the loaded Llama instance

    Returns the appropriate ModelProvider instance
    """
    if model_id == "anthropic":
        provider = AnthropicProvider()
        if not provider.is_available():
            raise ModelError(
                "Anthropic API key not found. Set DIFFSENSE_ANTHROPIC_API_KEY environment variable."
            )
        return provider

    elif model_id == "openai":
        provider = OpenAIProvider()
        if not provider.is_available():
            raise ModelError(
                "OpenAI API key not found. Set DIFFSENSE_OPENAI_API_KEY environment variable."
            )
        return provider

    else:
        # Local model provider
        if model_instance is None:
            raise ModelError("Local model instance required for local provider")
        return LocalModelProvider(model_instance)
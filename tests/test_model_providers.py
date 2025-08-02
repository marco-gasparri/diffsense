"""
Tests for model provider functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

from diffsense.model_providers import (
    ModelProvider,
    LocalModelProvider,
    AnthropicProvider,
    OpenAIProvider,
    create_provider
)
from diffsense.exceptions import ModelError


class TestLocalModelProvider:
    """Test cases for LocalModelProvider"""

    def test_initialization(self):
        """Test local provider initialization"""
        mock_model = Mock()
        provider = LocalModelProvider(mock_model)

        assert provider.model_instance == mock_model
        assert provider.is_available() is True

    def test_is_available_with_none_model(self):
        """Test availability check with None model"""
        provider = LocalModelProvider(None)
        assert provider.is_available() is False

    def test_generate_success(self):
        """Test successful generation"""
        # Mock model response
        mock_model = Mock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": "Test response"}
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20
            }
        }

        provider = LocalModelProvider(mock_model)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"}
        ]

        result, token_usage = provider.generate(messages, max_tokens=100, temperature=0.5)

        assert result == "Test response"
        assert token_usage["input"] == 50
        assert token_usage["output"] == 20
        mock_model.create_chat_completion.assert_called_once()

        # Check call arguments
        call_args = mock_model.create_chat_completion.call_args
        assert call_args.kwargs["messages"] == messages
        assert call_args.kwargs["max_tokens"] == 100
        assert call_args.kwargs["temperature"] == 0.5
        assert "stop" in call_args.kwargs

    def test_generate_success_without_usage(self):
        """Test successful generation without usage data"""
        # Mock model response without usage
        mock_model = Mock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": "Test response"}
            }]
        }

        provider = LocalModelProvider(mock_model)

        result, token_usage = provider.generate([{"role": "user", "content": "test"}], 100, 0.5)

        assert result == "Test response"
        assert token_usage["input"] == 0
        assert token_usage["output"] == 0

    def test_generate_empty_response(self):
        """Test handling of empty response"""
        mock_model = Mock()
        mock_model.create_chat_completion.return_value = {"choices": []}

        provider = LocalModelProvider(mock_model)

        with pytest.raises(ModelError, match="Empty response from model"):
            provider.generate([{"role": "user", "content": "test"}], 100, 0.5)

    def test_generate_empty_content(self):
        """Test handling of empty content"""
        mock_model = Mock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": ""}
            }]
        }

        provider = LocalModelProvider(mock_model)

        with pytest.raises(ModelError, match="Model returned empty content"):
            provider.generate([{"role": "user", "content": "test"}], 100, 0.5)

    def test_generate_with_none_model(self):
        """Test generation with None model"""
        provider = LocalModelProvider(None)

        with pytest.raises(ModelError, match="Model not loaded"):
            provider.generate([{"role": "user", "content": "test"}], 100, 0.5)

    def test_generate_exception_handling(self):
        """Test exception handling during generation"""
        mock_model = Mock()
        mock_model.create_chat_completion.side_effect = Exception("Test error")

        provider = LocalModelProvider(mock_model)

        with pytest.raises(ModelError, match="Local inference failed"):
            provider.generate([{"role": "user", "content": "test"}], 100, 0.5)


class TestAnthropicProvider:
    """Test cases for AnthropicProvider"""

    def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            provider = AnthropicProvider()

            assert provider.api_key is None
            assert provider._client is None
            assert provider.is_available() is False

    def test_initialization_with_api_key(self):
        """Test initialization with API key"""
        # Create a mock anthropic module
        mock_anthropic = MagicMock()
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"DIFFSENSE_ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'anthropic': mock_anthropic}):
                provider = AnthropicProvider()

                assert provider.api_key == "test-key"
                assert provider._client == mock_client
                assert provider.is_available() is True
                mock_anthropic.Anthropic.assert_called_once_with(api_key="test-key")

    def test_generate_success(self):
        """Test successful generation"""
        # Setup mock
        mock_anthropic = MagicMock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Anthropic response")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"DIFFSENSE_ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'anthropic': mock_anthropic}):
                provider = AnthropicProvider()

                messages = [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "User prompt"}
                ]

                result, token_usage = provider.generate(messages, max_tokens=1000, temperature=0.7)

                assert result == "Anthropic response"
                assert token_usage["input"] == 100
                assert token_usage["output"] == 50

                # Check API call
                mock_client.messages.create.assert_called_once()
                call_args = mock_client.messages.create.call_args

                assert call_args.kwargs["model"] == "claude-opus-4-20250514"
                assert call_args.kwargs["messages"] == [{"role": "user", "content": "User prompt"}]
                assert call_args.kwargs["system"] == "System prompt"
                assert call_args.kwargs["max_tokens"] == 1000
                assert call_args.kwargs["temperature"] == 0.7

    def test_generate_without_client(self):
        """Test generation without initialized client"""
        with patch.dict(os.environ, {}, clear=True):
            provider = AnthropicProvider()

            with pytest.raises(ModelError, match="Anthropic client not initialized"):
                provider.generate([{"role": "user", "content": "test"}], 100, 0.5)

    def test_generate_api_error(self):
        """Test handling of API errors"""
        mock_anthropic = MagicMock()
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API error")
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"DIFFSENSE_ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'anthropic': mock_anthropic}):
                provider = AnthropicProvider()

                with pytest.raises(ModelError, match="Anthropic API call failed"):
                    provider.generate([{"role": "user", "content": "test"}], 100, 0.5)


class TestOpenAIProvider:
    """Test cases for OpenAIProvider"""

    def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()

            assert provider.api_key is None
            assert provider._client is None
            assert provider.is_available() is False

    def test_initialization_with_api_key(self):
        """Test initialization with API key"""
        # Create a mock openai module
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(os.environ, {"DIFFSENSE_OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'openai': mock_openai}):
                provider = OpenAIProvider()

                assert provider.api_key == "test-key"
                assert provider._client == mock_client
                assert provider.is_available() is True
                mock_openai.OpenAI.assert_called_once_with(api_key="test-key")

    def test_generate_success(self):
        """Test successful generation"""
        # Setup mock
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "OpenAI response"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 30
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(os.environ, {"DIFFSENSE_OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'openai': mock_openai}):
                provider = OpenAIProvider()

                messages = [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "User prompt"}
                ]

                result, token_usage = provider.generate(messages, max_tokens=1000, temperature=0.7)

                assert result == "OpenAI response"
                assert token_usage["input"] == 80
                assert token_usage["output"] == 30

                # Check API call
                mock_client.chat.completions.create.assert_called_once()
                call_args = mock_client.chat.completions.create.call_args

                assert call_args.kwargs["model"] == "gpt-4o"
                assert call_args.kwargs["messages"] == messages
                assert call_args.kwargs["max_tokens"] == 1000
                assert call_args.kwargs["temperature"] == 0.7

    def test_generate_without_client(self):
        """Test generation without initialized client"""
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()

            with pytest.raises(ModelError, match="OpenAI client not initialized"):
                provider.generate([{"role": "user", "content": "test"}], 100, 0.5)

    def test_generate_api_error(self):
        """Test handling of API errors"""
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(os.environ, {"DIFFSENSE_OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'openai': mock_openai}):
                provider = OpenAIProvider()

                with pytest.raises(ModelError, match="OpenAI API call failed"):
                    provider.generate([{"role": "user", "content": "test"}], 100, 0.5)


class TestCreateProvider:
    """Test cases for create_provider factory function"""

    def test_create_anthropic_provider_success(self):
        """Test creating Anthropic provider with API key"""
        mock_anthropic = MagicMock()
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"DIFFSENSE_ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'anthropic': mock_anthropic}):
                provider = create_provider("anthropic")
                assert isinstance(provider, AnthropicProvider)

    def test_create_anthropic_provider_no_api_key(self):
        """Test creating Anthropic provider without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ModelError, match="Anthropic API key not found"):
                create_provider("anthropic")

    def test_create_openai_provider_success(self):
        """Test creating OpenAI provider with API key"""
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(os.environ, {"DIFFSENSE_OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'openai': mock_openai}):
                provider = create_provider("openai")
                assert isinstance(provider, OpenAIProvider)

    def test_create_openai_provider_no_api_key(self):
        """Test creating OpenAI provider without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ModelError, match="OpenAI API key not found"):
                create_provider("openai")

    def test_create_local_provider_success(self):
        """Test creating local provider with model instance"""
        mock_model = Mock()
        provider = create_provider("some/local/model", mock_model)

        assert isinstance(provider, LocalModelProvider)
        assert provider.model_instance == mock_model

    def test_create_local_provider_no_instance(self):
        """Test creating local provider without model instance"""
        with pytest.raises(ModelError, match="Local model instance required"):
            create_provider("some/local/model", None)
"""
Tests for the LLM manager functionality
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from diffsense.llm_manager import LLMManager
from diffsense.diff_engine import DiffEngine
from diffsense.exceptions import ModelError
from diffsense.diff_engine import DiffBlock
from diffsense.conflict_resolver import ConflictSection, ConflictResolution, ConflictResolutionConfidence


class TestLLMManager:
    """Test cases for LLMManager class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_cache = Path("test_models")
        self.temp_cache.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if self.temp_cache.exists():
            shutil.rmtree(self.temp_cache)

    def test_initialization(self):
        """Test LLM manager initialization"""
        manager = LLMManager(cache_dir=self.temp_cache)

        assert manager.model_id == LLMManager.DEFAULT_MODEL
        assert manager.cache_dir == self.temp_cache
        assert manager.max_tokens == 1024  # Updated from 512
        assert manager.temperature == 0.3

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        custom_model = "custom/model"
        custom_tokens = 1024
        custom_temp = 0.7

        manager = LLMManager(
            model_id=custom_model,
            cache_dir=self.temp_cache,
            max_tokens=custom_tokens,
            temperature=custom_temp
        )

        assert manager.model_id == custom_model
        assert manager.max_tokens == custom_tokens
        assert manager.temperature == custom_temp

    def test_model_info(self):
        """Test getting model information"""
        manager = LLMManager(cache_dir=self.temp_cache)
        info = manager.get_model_info()

        assert "model_id" in info
        assert "cache_dir" in info
        assert "max_tokens" in info
        assert "temperature" in info
        assert "loaded" in info
        assert "provider" in info
        assert info["loaded"] is False
        assert info["provider"] == "local"

    @pytest.mark.slow
    @pytest.mark.network
    @patch('diffsense.llm_manager.hf_hub_download')
    @patch('diffsense.llm_manager.Llama')
    def test_model_loading_success(self, mock_llama, mock_download):
        """Test successful model loading"""
        # Mock the download
        mock_download.return_value = str(self.temp_cache / "model.gguf")

        # Mock the Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model

        manager = LLMManager(cache_dir=self.temp_cache)

        # This should not raise an exception
        try:
            model = manager._load_model()
            assert model == mock_model
        except ModelError:
            pytest.fail("Model loading should succeed with mocked dependencies")

    @patch('diffsense.llm_manager.hf_hub_download')
    def test_model_loading_failure(self, mock_download):
        """Test model loading failure handling"""
        # Mock download failure
        mock_download.side_effect = Exception("Download failed")

        manager = LLMManager(cache_dir=self.temp_cache)

        with pytest.raises(ModelError):
            manager._load_model()

    def test_hardware_detection(self):
        """Test hardware parameter detection"""
        manager = LLMManager(cache_dir=self.temp_cache)
        params = manager._get_model_parameters()

        assert isinstance(params, dict)
        assert "n_ctx" in params
        assert "n_threads" in params
        assert "verbose" in params
        assert params["verbose"] is False

    def test_cuda_detection(self):
        """Test CUDA availability detection"""
        manager = LLMManager(cache_dir=self.temp_cache)

        # This should not raise an exception
        try:
            cuda_available = manager._cuda_available()
            assert isinstance(cuda_available, bool)
        except Exception:
            pytest.fail("CUDA detection should not raise exceptions")

    def test_prompt_building(self):
        """Test analysis prompt building"""
        manager = LLMManager(cache_dir=self.temp_cache)

        # Create sample diff blocks
        engine = DiffEngine()
        blocks = engine.compute_diff("old content", "new content")

        prompt = manager._build_analysis_prompt(blocks)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "**PRIMARY CHANGE:**" in prompt
        assert "**SECONDARY CHANGES:**" in prompt
        assert "**SUMMARY:**" in prompt

    def test_prompt_building_with_context(self):
        """Test analysis prompt building with full context"""
        manager = LLMManager(cache_dir=self.temp_cache)

        # Create sample diff blocks
        engine = DiffEngine()
        blocks = engine.compute_diff("old content", "new content")

        prompt = manager._build_analysis_prompt(
            blocks,
            original_content="old content",
            modified_content="new content",
            file1_name="old.txt",
            file2_name="new.txt"
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "FULL FILE CONTEXT:" in prompt
        assert "--- Original File (old.txt) ---" in prompt
        assert "--- Modified File (new.txt) ---" in prompt

    def test_token_estimation(self):
        """Test token count estimation."""
        manager = LLMManager(cache_dir=self.temp_cache)

        # Test token estimation
        text = "This is a test string with multiple words"
        tokens = manager._estimate_token_count(text)

        # Should be approximately 1/4 of character count
        expected = len(text) // 4
        assert tokens == expected

    def test_large_context_fallback(self):
        """Test fallback for large contexts that exceed token limits"""
        manager = LLMManager(cache_dir=self.temp_cache)

        # Create large content that would exceed token limits (need >20000 chars for >5000 tokens)
        large_content = "This is a very long line with many words to create large content " * 400  # ~26000 chars
        engine = DiffEngine()
        blocks = engine.compute_diff("small", "changed")

        prompt = manager._build_analysis_prompt(
            blocks,
            original_content=large_content,
            modified_content=large_content,
            file1_name="large.txt",
            file2_name="large.txt"
        )

        # Should fallback to smart context, not full context
        assert "FULL FILE CONTEXT:" not in prompt
        assert "RELEVANT CONTEXT" in prompt or "CHANGES TO ANALYZE:" in prompt

    def test_smart_context_extraction(self):
        """Test smart context extraction around changes"""
        manager = LLMManager(cache_dir=self.temp_cache)

        # Create content with clear structure
        content = "\n".join([f"line {i}" for i in range(1, 21)])  # 20 lines

        # Create mock blocks (simulate changes around line 10)
        from diffsense.diff_engine import DiffBlock, LineDiff, ChangeType

        block = DiffBlock(
            old_start=10, old_count=1, new_start=10, new_count=1,
            lines=[LineDiff(ChangeType.REPLACE, "line 10", "changed line 10", 10, 10, [])]
        )

        smart_context = manager._extract_smart_context(content, [block], context_lines=3)

        # Should include lines around the change
        assert "line 7" in smart_context or "line 8" in smart_context  # Some context before
        assert "line 10" in smart_context  # The changed line
        assert "line 12" in smart_context or "line 13" in smart_context  # Some context after
        assert "   7:" in smart_context or "   8:" in smart_context  # Line numbers included

    @patch('diffsense.llm_manager.LLMManager._load_model')
    def test_analyze_empty_diff(self, mock_load):
        """Test analysis of empty diff"""
        manager = LLMManager(cache_dir=self.temp_cache)

        result = manager.analyze_diff([])
        assert result == "No changes detected in the provided files"

    @patch('diffsense.llm_manager.LLMManager._load_model')
    @patch('diffsense.llm_manager.LLMManager._run_inference')
    def test_analyze_diff_success(self, mock_inference, mock_load):
        """Test successful diff analysis"""
        # Mock model and inference
        mock_model = Mock()
        mock_load.return_value = mock_model
        mock_inference.return_value = "This is a test analysis"

        manager = LLMManager(cache_dir=self.temp_cache)

        # Create sample diff
        engine = DiffEngine()
        blocks = engine.compute_diff("old", "new")

        result = manager.analyze_diff(blocks)
        assert result == "This is a test analysis"

    def test_cache_clearing(self):
        """Test cache directory clearing."""
        # Create a test file in cache
        test_file = self.temp_cache / "test_model.gguf"
        test_file.write_text("test content")

        manager = LLMManager(cache_dir=self.temp_cache)
        manager.clear_cache()

        assert self.temp_cache.exists()
        assert not test_file.exists()

    @pytest.mark.network
    @patch('diffsense.llm_manager.HfApi')
    def test_find_model_filename(self, mock_api):
        """Test finding appropriate model filename"""
        # Create mock file objects that simulate HuggingFace model files
        mock_file_1 = Mock()
        mock_file_1.rfilename = "model.Q4_K_M.gguf"

        mock_file_2 = Mock()
        mock_file_2.rfilename = "model.Q8_0.gguf"

        mock_file_3 = Mock()
        mock_file_3.rfilename = "config.json"

        # Mock model_info response
        mock_model_info = Mock()
        mock_model_info.siblings = [mock_file_1, mock_file_2, mock_file_3]

        # Mock HfApi instance
        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = mock_model_info
        mock_api.return_value = mock_api_instance

        manager = LLMManager(model_id="test/model", cache_dir=self.temp_cache)
        filename = manager._find_model_filename()

        assert filename == "model.Q4_K_M.gguf"  # Should prefer Q4_K_M

    @patch('diffsense.llm_manager.HfApi')
    def test_find_model_filename_no_gguf(self, mock_api):
        """Test filename finding when no GGUF files exist"""
        # Create mock file objects with no GGUF files
        mock_file_1 = Mock()
        mock_file_1.rfilename = "pytorch_model.bin"

        mock_file_2 = Mock()
        mock_file_2.rfilename = "config.json"

        # Mock model_info response
        mock_model_info = Mock()
        mock_model_info.siblings = [mock_file_1, mock_file_2]

        # Mock HfApi instance
        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = mock_model_info
        mock_api.return_value = mock_api_instance

        manager = LLMManager(model_id="test/model", cache_dir=self.temp_cache)

        with pytest.raises(ModelError):
            manager._find_model_filename()


    def test_get_model_parameters_safe(self):
        """Test hardware parameter detection without external dependencies"""
        manager = LLMManager(cache_dir=self.temp_cache)

        with patch('platform.system', return_value='Linux'):
            with patch('platform.machine', return_value='x86_64'):
                with patch('psutil.cpu_count', return_value=4):
                    with patch('psutil.virtual_memory') as mock_memory:
                        mock_memory.return_value.total = 8 * 1024 ** 3  # 8GB

                        params = manager._get_model_parameters()

                        assert isinstance(params, dict)
                        assert "n_ctx" in params
                        assert "n_threads" in params
                        assert params["n_threads"] <= 8  # Reasonable limit

    def test_anthropic_model_initialization(self):
        """Test initialization with Anthropic model"""
        mock_anthropic = MagicMock()

        with patch.dict(os.environ, {"DIFFSENSE_ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'anthropic': mock_anthropic}):
                manager = LLMManager(model_id="anthropic", cache_dir=self.temp_cache)
                assert manager.model_id == "anthropic"

                info = manager.get_model_info()
                assert info["provider"] == "anthropic"
                assert info["api_key_set"] is True

    def test_openai_model_initialization(self):
        """Test initialization with OpenAI model"""
        mock_openai = MagicMock()

        with patch.dict(os.environ, {"DIFFSENSE_OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {'openai': mock_openai}):
                manager = LLMManager(model_id="openai", cache_dir=self.temp_cache)
                assert manager.model_id == "openai"

                info = manager.get_model_info()
                assert info["provider"] == "openai"
                assert info["api_key_set"] is True

    def test_remote_model_without_api_key(self):
        """Test remote model initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            manager = LLMManager(model_id="anthropic", cache_dir=self.temp_cache)

            with pytest.raises(ModelError, match="Anthropic API key not found"):
                manager.analyze_diff([DiffBlock(1, 1, 1, 1, [])])

    @patch('diffsense.model_providers.AnthropicProvider')
    def test_analyze_diff_with_anthropic(self, mock_anthropic_provider):
        """Test diff analysis with Anthropic provider"""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_instance.is_available.return_value = True
        mock_provider_instance.generate.return_value = ("Anthropic analysis", {"input": 100, "output": 50})
        mock_anthropic_provider.return_value = mock_provider_instance

        with patch.dict(os.environ, {"DIFFSENSE_ANTHROPIC_API_KEY": "test-key"}):
            manager = LLMManager(model_id="anthropic", cache_dir=self.temp_cache)

            # Create sample diff
            engine = DiffEngine()
            blocks = engine.compute_diff("old", "new")

            result = manager.analyze_diff(blocks)
            assert result == "Anthropic analysis"
            mock_provider_instance.generate.assert_called_once()

    @patch('diffsense.model_providers.OpenAIProvider')
    def test_analyze_diff_with_openai(self, mock_openai_provider):
        """Test diff analysis with OpenAI provider"""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_instance.is_available.return_value = True
        mock_provider_instance.generate.return_value = ("OpenAI analysis", {"input": 80, "output": 30})
        mock_openai_provider.return_value = mock_provider_instance

        with patch.dict(os.environ, {"DIFFSENSE_OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(model_id="openai", cache_dir=self.temp_cache)

            # Create sample diff
            engine = DiffEngine()
            blocks = engine.compute_diff("old", "new")

            result = manager.analyze_diff(blocks)
            assert result == "OpenAI analysis"
            mock_provider_instance.generate.assert_called_once()

    @patch('diffsense.model_providers.LocalModelProvider')
    def test_resolve_conflict_success(self, mock_provider_class):
        """Test successful conflict resolution"""
        # Setup mock provider
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = (
            "CONFIDENCE: HIGH\nRESOLUTION:\nresolved content\nEND_RESOLUTION\nEXPLANATION: Test explanation\n",
            {"input": 200, "output": 100}
        )
        mock_provider_class.return_value = mock_provider

        # Setup mock model
        mock_model = Mock()
        manager = LLMManager(cache_dir=self.temp_cache)
        manager._model_instance = mock_model

        # Create test conflict
        conflict = ConflictSection(
            start_line=1,
            end_line=5,
            current_content="current",
            incoming_content="incoming",
            current_marker="<<<<<<< HEAD",
            incoming_marker=">>>>>>> branch",
            separator_line=3
        )

        file_context = {
            "file_language": "python",
            "total_lines": 100,
            "total_conflicts": 1,
            "conflicts_context": [{
                "conflict_lines": "1-5",
                "context_before": "before",
                "context_after": "after",
                "current_branch_info": "HEAD",
                "incoming_branch_info": "branch"
            }]
        }

        resolution, tokens = manager.resolve_conflict(conflict, file_context)

        assert isinstance(resolution, ConflictResolution)
        assert resolution.resolved_content == "resolved content"
        assert resolution.explanation == "Test explanation"
        assert resolution.confidence == ConflictResolutionConfidence.HIGH
        assert tokens == 300

    @patch('diffsense.model_providers.LocalModelProvider')
    def test_resolve_conflict_with_alternatives(self, mock_provider_class):
        """Test conflict resolution with alternatives"""
        # Setup mock provider with medium confidence response
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = (
            """CONFIDENCE: MEDIUM
RESOLUTION:
main resolution
END_RESOLUTION
EXPLANATION: Main explanation
ALTERNATIVE_1:
alternative resolution
END_ALTERNATIVE
ALTERNATIVE_EXPLANATION_1: Alternative explanation
""",
            {"input": 250, "output": 150}
        )
        mock_provider_class.return_value = mock_provider

        mock_model = Mock()
        manager = LLMManager(cache_dir=self.temp_cache)
        manager._model_instance = mock_model

        conflict = ConflictSection(
            start_line=1,
            end_line=5,
            current_content="current",
            incoming_content="incoming",
            current_marker="<<<<<<< HEAD",
            incoming_marker=">>>>>>> branch",
            separator_line=3
        )

        file_context = {"conflicts_context": [{"conflict_lines": "1-5"}]}

        resolution, tokens = manager.resolve_conflict(conflict, file_context)

        assert resolution.confidence == ConflictResolutionConfidence.MEDIUM
        assert resolution.alternative_resolutions is not None
        assert len(resolution.alternative_resolutions) == 1
        assert resolution.alternative_resolutions[0]["resolution"] == "alternative resolution"

    def test_build_conflict_resolution_prompt(self):
        """Test building conflict resolution prompt"""
        manager = LLMManager(cache_dir=self.temp_cache)

        conflict = ConflictSection(
            start_line=10,
            end_line=15,
            current_content="current code",
            incoming_content="incoming code",
            current_marker="<<<<<<< HEAD",
            incoming_marker=">>>>>>> feature",
            separator_line=12
        )

        file_context = {
            "file_language": "python",
            "total_lines": 200,
            "total_conflicts": 2,
            "conflicts_context": [
                {"conflict_lines": "10-15", "context_before": "def func():", "context_after": "return"}
            ]
        }

        prompt = manager._build_conflict_resolution_prompt(conflict, file_context)

        assert "You are resolving a Git merge conflict" in prompt
        assert "Language: python" in prompt
        assert "current code" in prompt
        assert "incoming code" in prompt
        assert "def func():" in prompt
        assert "CONFIDENCE: [HIGH|MEDIUM|LOW]" in prompt

    def test_build_conflict_prompt_with_additional_context(self):
        """Test building prompt with additional context files"""
        manager = LLMManager(cache_dir=self.temp_cache)

        conflict = Mock(start_line=1, end_line=5, current_content="", incoming_content="")
        file_context = {"conflicts_context": [{"conflict_lines": "1-5"}]}
        additional_context = {
            "model.py": "class Model:\n    pass",
            "utils.py": "def helper():\n    return True"
        }

        prompt = manager._build_conflict_resolution_prompt(
            conflict, file_context, additional_context, "test.py"
        )

        assert "ADDITIONAL CONTEXT FILES:" in prompt
        assert "--- model.py ---" in prompt
        assert "class Model:" in prompt
        assert "--- utils.py ---" in prompt

    def test_parse_conflict_resolution_success(self):
        """Test parsing well-formatted resolution response"""
        manager = LLMManager(cache_dir=self.temp_cache)

        response = """CONFIDENCE: HIGH
RESOLUTION:
def fixed_function():
    return True
END_RESOLUTION
EXPLANATION: Fixed the function to return True
"""

        conflict = Mock()
        resolution = manager._parse_conflict_resolution(response, conflict)

        assert resolution.confidence == ConflictResolutionConfidence.HIGH
        assert resolution.resolved_content == "def fixed_function():\n    return True"
        assert resolution.explanation == "Fixed the function to return True"
        assert resolution.alternative_resolutions is None

    def test_parse_conflict_resolution_malformed(self):
        """Test parsing malformed resolution response"""
        manager = LLMManager(cache_dir=self.temp_cache)

        response = """This is a malformed response without proper format.
Just some text about the resolution."""

        conflict = Mock(incoming_content="fallback content")
        resolution = manager._parse_conflict_resolution(response, conflict)

        # Should fallback gracefully
        assert resolution.confidence == ConflictResolutionConfidence.LOW
        assert resolution.explanation == "No explanation provided"

    def test_parse_conflict_resolution_code_block_fallback(self):
        """Test parsing with code block fallback"""
        manager = LLMManager(cache_dir=self.temp_cache)

        response = """CONFIDENCE: MEDIUM
Here's the resolution:
```python
def resolved():
    return 42
```
EXPLANATION: Returns the answer"""

        conflict = Mock()
        resolution = manager._parse_conflict_resolution(response, conflict)

        assert "def resolved():" in resolution.resolved_content
        assert "return 42" in resolution.resolved_content

    def test_token_tracking_in_resolve_conflict(self):
        """Test that token usage is properly tracked"""
        manager = LLMManager(cache_dir=self.temp_cache)

        # Mock provider that returns token usage
        mock_provider = Mock()
        mock_provider.generate.return_value = (
            "CONFIDENCE: HIGH\nRESOLUTION:\ntest\nEND_RESOLUTION\nEXPLANATION: test",
            {"input": 123, "output": 45}
        )
        manager._provider = mock_provider

        conflict = Mock(start_line=1, end_line=2, current_content="", incoming_content="")
        file_context = {"conflicts_context": [{"conflict_lines": "1-2"}]}

        resolution, tokens = manager.resolve_conflict(conflict, file_context)

        assert tokens == 168  # 123 + 45
        assert manager._last_token_usage["input"] == 123
        assert manager._last_token_usage["output"] == 45
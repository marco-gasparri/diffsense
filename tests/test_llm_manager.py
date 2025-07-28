"""
Tests for the LLM manager functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from diffsense.llm_manager import LLMManager
from diffsense.diff_engine import DiffEngine
from diffsense.exceptions import ModelError


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
        assert info["loaded"] is False

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
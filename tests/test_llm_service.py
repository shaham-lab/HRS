"""
Unit tests for LLM service module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from LLM.llm_service import sanitize_input, validate_severity, get_provider


class TestSanitizeInput(unittest.TestCase):
    """Test cases for input sanitization."""
    
    def test_sanitize_normal_input(self):
        """Test sanitization of normal text input."""
        result = sanitize_input("I have a headache")
        self.assertEqual(result, "I have a headache")
    
    def test_sanitize_empty_input(self):
        """Test sanitization of empty input."""
        result = sanitize_input("")
        self.assertEqual(result, "")
    
    def test_sanitize_none_input(self):
        """Test sanitization of None input."""
        result = sanitize_input(None)
        self.assertEqual(result, "")
    
    def test_sanitize_removes_control_characters(self):
        """Test that control characters are removed."""
        # Test with various control characters
        text_with_controls = "Hello\x00World\x1f"
        result = sanitize_input(text_with_controls)
        self.assertEqual(result, "HelloWorld")
    
    def test_sanitize_respects_max_length(self):
        """Test that input is truncated to max length."""
        long_text = "a" * 2000
        result = sanitize_input(long_text, max_length=100)
        self.assertEqual(len(result), 100)
    
    def test_sanitize_default_max_length(self):
        """Test that default max length is 1000."""
        long_text = "a" * 2000
        result = sanitize_input(long_text)
        self.assertEqual(len(result), 1000)
    
    def test_sanitize_strips_whitespace(self):
        """Test that leading and trailing whitespace is stripped."""
        result = sanitize_input("  test  ")
        self.assertEqual(result, "test")
    
    def test_sanitize_preserves_internal_whitespace(self):
        """Test that internal whitespace is preserved."""
        result = sanitize_input("test   string")
        self.assertEqual(result, "test   string")


class TestValidateSeverity(unittest.TestCase):
    """Test cases for severity validation."""
    
    def test_validate_severity_valid_number(self):
        """Test validation of valid severity numbers."""
        for severity in range(1, 11):
            with self.subTest(severity=severity):
                result = validate_severity(severity)
                self.assertEqual(result, severity)
    
    def test_validate_severity_valid_string(self):
        """Test validation of valid severity as string."""
        result = validate_severity("5")
        self.assertEqual(result, 5)
    
    def test_validate_severity_below_range(self):
        """Test that severity below 1 defaults to 5."""
        result = validate_severity(0)
        self.assertEqual(result, 5)
        
        result = validate_severity(-5)
        self.assertEqual(result, 5)
    
    def test_validate_severity_above_range(self):
        """Test that severity above 10 defaults to 5."""
        result = validate_severity(11)
        self.assertEqual(result, 5)
        
        result = validate_severity(100)
        self.assertEqual(result, 5)
    
    def test_validate_severity_invalid_string(self):
        """Test that invalid string defaults to 5."""
        result = validate_severity("invalid")
        self.assertEqual(result, 5)
    
    def test_validate_severity_none(self):
        """Test that None defaults to 5."""
        result = validate_severity(None)
        self.assertEqual(result, 5)
    
    def test_validate_severity_float(self):
        """Test that float values are handled correctly."""
        result = validate_severity(5.7)
        self.assertEqual(result, 5)


class TestGetProvider(unittest.TestCase):
    """Test cases for provider factory function."""
    
    @patch('LLM.llm_service.GeminiProvider')
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    def test_get_provider_gemini_default(self, mock_gemini):
        """Test that Gemini is the default provider."""
        mock_instance = Mock()
        mock_gemini.return_value = mock_instance
        
        with patch.dict('os.environ', {}, clear=True):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                provider = get_provider()
                mock_gemini.assert_called_once_with('test_key')
                mock_instance.initialize.assert_called_once()
    
    @patch('LLM.llm_service.GeminiProvider')
    @patch.dict('os.environ', {'LLM_PROVIDER': 'gemini', 'GEMINI_API_KEY': 'test_key'})
    def test_get_provider_gemini_explicit(self, mock_gemini):
        """Test explicit Gemini provider selection."""
        mock_instance = Mock()
        mock_gemini.return_value = mock_instance
        
        provider = get_provider('gemini')
        mock_gemini.assert_called_once_with('test_key')
        mock_instance.initialize.assert_called_once()
    
    @patch('LLM.llm_service.OpenAIProvider')
    @patch.dict('os.environ', {'LLM_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test_key'})
    def test_get_provider_openai(self, mock_openai):
        """Test OpenAI provider selection."""
        mock_instance = Mock()
        mock_openai.return_value = mock_instance
        
        provider = get_provider('openai')
        mock_openai.assert_called_once_with('test_key')
        mock_instance.initialize.assert_called_once()
    
    def test_get_provider_invalid(self):
        """Test that invalid provider raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_provider('invalid_provider')
        
        self.assertIn('Unknown provider', str(context.exception))
    
    @patch('LLM.llm_service.GeminiProvider')
    @patch.dict('os.environ', {'LLM_PROVIDER': 'GEMINI', 'GEMINI_API_KEY': 'test_key'})
    def test_get_provider_case_insensitive(self, mock_gemini):
        """Test that provider name is case-insensitive."""
        mock_instance = Mock()
        mock_gemini.return_value = mock_instance
        
        provider = get_provider('GEMINI')
        mock_gemini.assert_called_once()


if __name__ == '__main__':
    unittest.main()

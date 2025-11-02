"""
Gemini Provider Module for Health Recommendation System.

This module defines the Google Gemini LLM provider implementation.
"""

import os
from llm_provider import LLMProvider
from llm_constants import (
    DEMO_MODE_RESPONSE,
    ERROR_RESPONSE_TEMPLATE,
    ERROR_EMPTY_RESPONSE,
    ERROR_INIT_TEMPLATE,
    ERROR_GENERATION_TEMPLATE
)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""
    
    # Model configuration as class constant
    DEFAULT_MODEL = 'gemini-1.5-flash'
    PROVIDER_NAME = 'Gemini'
    
    def initialize(self) -> bool:
        """Initialize the Gemini client."""
        if not self.api_key or self.api_key == 'your_gemini_api_key_here':
            return False
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            # Use environment variable for model if provided, otherwise use default
            model_name = os.getenv('GEMINI_MODEL', self.DEFAULT_MODEL)
            self._client = genai.GenerativeModel(model_name)
            return True
        except Exception as e:
            print(ERROR_INIT_TEMPLATE.format(provider_name=self.PROVIDER_NAME, error=str(e)))
            return False
    
    def generate_response(self, prompt: str, system_message: str) -> str:
        """Generate a response using Google Gemini."""
        if not self.is_available():
            return self._demo_response(prompt)
        
        try:
            # Combine system message and prompt for Gemini
            full_prompt = f"{system_message}\n\n{prompt}"
            
            response = self._client.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 500,
                }
            )
            
            # Check if response has text content
            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                print("Warning: {} response did not contain text content".format(self.PROVIDER_NAME))
                return ERROR_EMPTY_RESPONSE
        except Exception as e:
            print(ERROR_GENERATION_TEMPLATE.format(provider_name=self.PROVIDER_NAME, error=str(e)))
            return ERROR_RESPONSE_TEMPLATE.format(provider_name=self.PROVIDER_NAME)
    
    def _demo_response(self, prompt: str) -> str:
        """Return a demo response when API key is not configured."""
        return DEMO_MODE_RESPONSE.format(provider_name=self.PROVIDER_NAME)

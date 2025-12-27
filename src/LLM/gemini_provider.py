"""
Gemini Provider Module for Health Recommendation System.

This module defines the Google Gemini LLM provider implementation.
"""

import os
import google.generativeai as genai
from .llm_provider import LLMProvider
from .llm_constants import (
    DEMO_MODE_RESPONSE,
    ERROR_RESPONSE_TEMPLATE,
    ERROR_EMPTY_RESPONSE,
    ERROR_INIT_TEMPLATE,
    ERROR_GENERATION_TEMPLATE
)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""
    
    # Model configuration as class constant
    DEFAULT_MODEL = 'gemini-2.5-flash'
    PROVIDER_NAME = 'Gemini'
    
    def initialize(self) -> bool:
        """Initialize the Gemini client."""
        if not self.api_key or self.api_key == 'your_gemini_api_key_here':
            return False
        
        try:
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
            return self._demo_response()
        
        try:
            # Combine system message and prompt for Gemini
            full_prompt = f"{system_message}\n\n{prompt}"

            # file: gemini_provider.py
            from typing import Any, cast

            gen_config = cast(Any, {
                'temperature': 0.7,
                'max_output_tokens': 4096,
            })

            response = self._client.generate_content(
                full_prompt,
                generation_config=gen_config
            )

            # Check for blocked or filtered responses
            if response.candidates:
                candidate = response.candidates[0]
                # Check if response was blocked
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    # finish_reason values: STOP (normal), MAX_TOKENS, SAFETY, RECITATION, OTHER
                    if finish_reason not in [1, 'STOP']:  # 1 is the enum value for STOP
                        print(f"Warning: {self.PROVIDER_NAME} response blocked with finish_reason: {finish_reason}")
                        if hasattr(candidate, 'safety_ratings'):
                            print(f"Safety ratings: {candidate.safety_ratings}")
                        return ERROR_EMPTY_RESPONSE
            
            # Try to access text content safely
            try:
                if response.text:
                    return response.text
            except ValueError as ve:
                # This exception is raised when response.text is accessed but no valid parts exist
                print(f"Warning: {self.PROVIDER_NAME} response did not contain valid text content: {str(ve)}")
                return ERROR_EMPTY_RESPONSE
            
            # Fallback if no text found
            print("Warning: {} response did not contain text content".format(self.PROVIDER_NAME))
            return ERROR_EMPTY_RESPONSE
        except Exception as e:
            print(ERROR_GENERATION_TEMPLATE.format(provider_name=self.PROVIDER_NAME, error=str(e)))
            return ERROR_RESPONSE_TEMPLATE.format(provider_name=self.PROVIDER_NAME)
    
    def _demo_response(self) -> str:
        """Return a demo response when API key is not configured."""
        return DEMO_MODE_RESPONSE.format(provider_name=self.PROVIDER_NAME)

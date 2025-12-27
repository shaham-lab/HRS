"""
OpenAI Provider Module for Health Recommendation System.

This module defines the OpenAI LLM provider implementation.
"""

import os
from .llm_provider import LLMProvider
from .llm_constants import (
    DEMO_MODE_RESPONSE,
    ERROR_RESPONSE_TEMPLATE,
    ERROR_INIT_TEMPLATE,
    ERROR_GENERATION_TEMPLATE
)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    # Model configuration as class constant
    DEFAULT_MODEL = 'gpt-3.5-turbo'
    PROVIDER_NAME = 'OpenAI'
    
    def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        if not self.api_key or self.api_key == 'your_openai_api_key_here':
            return False
        
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            return True
        except Exception as e:
            print(ERROR_INIT_TEMPLATE.format(provider_name=self.PROVIDER_NAME, error=str(e)))
            return False
    
    def generate_response(self, prompt: str, system_message: str) -> str:
        """Generate a response using OpenAI GPT."""
        if not self.is_available():
            return self._demo_response()
        
        try:
            from typing import Iterable, Any, cast
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            messages_typed = cast(Iterable[Any], messages)
            
            # Use environment variable for model if provided, otherwise use default
            model_name = os.getenv('OPENAI_MODEL', self.DEFAULT_MODEL)
            
            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages_typed,
                temperature=0.7,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            return content if content is not None else ERROR_RESPONSE_TEMPLATE.format(provider_name=self.PROVIDER_NAME)
        except Exception as e:
            print(ERROR_GENERATION_TEMPLATE.format(provider_name=self.PROVIDER_NAME, error=str(e)))
            return ERROR_RESPONSE_TEMPLATE.format(provider_name=self.PROVIDER_NAME)
    
    def _demo_response(self) -> str:
        """Return a demo response when API key is not configured."""
        return DEMO_MODE_RESPONSE.format(provider_name=self.PROVIDER_NAME)

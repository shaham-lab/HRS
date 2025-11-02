"""
LLM Provider Module for Health Recommendation System.

This module defines the abstract LLMProvider class and concrete implementations
for different LLM providers (OpenAI, Gemini).
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the provider with an API key.
        
        Args:
            api_key: Optional API key for the provider. If None, the provider
                    should handle it appropriately (e.g., demo mode).
        """
        self.api_key = api_key
        self._client = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the provider client.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, system_message: str) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The user prompt/question.
            system_message: The system message to set context.
            
        Returns:
            str: The generated response from the LLM.
        """
        pass
    
    def is_available(self) -> bool:
        """Check if the provider is available (has valid credentials).
        
        Returns:
            bool: True if provider is available, False otherwise.
        """
        return self._client is not None


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    # Model configuration as class constant
    DEFAULT_MODEL = 'gpt-3.5-turbo'
    
    def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        if not self.api_key or self.api_key == 'your_openai_api_key_here':
            return False
        
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, system_message: str) -> str:
        """Generate a response using OpenAI GPT."""
        if not self.is_available():
            return self._demo_response(prompt)
        
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
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating OpenAI response: {str(e)}")
            return "Unable to generate health recommendations at this time. Please try again later or ensure your OpenAI API key is configured correctly."
    
    def _demo_response(self, prompt: str) -> str:
        """Return a demo response when API key is not configured."""
        return f"""**DEMO MODE - No OpenAI API key configured**

Based on your query, this is a demonstration response. To get real AI-powered health recommendations, please set up your OpenAI API key in the .env file.

**General Recommendations:**
1. Monitor your symptoms and note any changes
2. Stay hydrated and get adequate rest
3. Maintain a healthy diet
4. Take over-the-counter medications as appropriate

**When to Seek Medical Attention:**
- If symptoms persist for more than a few days
- If symptoms worsen or become severe
- If you experience any concerning or unusual symptoms

**Important:** This demo response is not actual medical advice. For real health recommendations, configure your OpenAI API key. Always consult with a qualified healthcare professional for proper diagnosis and treatment.
"""


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""
    
    # Model configuration as class constant
    DEFAULT_MODEL = 'gemini-1.5-flash'
    
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
            print(f"Error initializing Gemini client: {str(e)}")
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
                print("Warning: Gemini response did not contain text content")
                return "Unable to generate health recommendations at this time. The AI service returned an empty response. Please try again."
        except Exception as e:
            print(f"Error generating Gemini response: {str(e)}")
            return "Unable to generate health recommendations at this time. Please try again later or ensure your Gemini API key is configured correctly."
    
    def _demo_response(self, prompt: str) -> str:
        """Return a demo response when API key is not configured."""
        return f"""**DEMO MODE - No Gemini API key configured**

Based on your query, this is a demonstration response. To get real AI-powered health recommendations, please set up your Gemini API key in the .env file.

**General Recommendations:**
1. Monitor your symptoms and note any changes
2. Stay hydrated and get adequate rest
3. Maintain a healthy diet
4. Take over-the-counter medications as appropriate

**When to Seek Medical Attention:**
- If symptoms persist for more than a few days
- If symptoms worsen or become severe
- If you experience any concerning or unusual symptoms

**Important:** This demo response is not actual medical advice. For real health recommendations, configure your Gemini API key. Always consult with a qualified healthcare professional for proper diagnosis and treatment.
"""


def get_provider(provider_name: str = None) -> LLMProvider:
    """Factory function to get the appropriate LLM provider.
    
    Args:
        provider_name: Name of the provider ('openai' or 'gemini'). 
                      If None, will read from environment variable LLM_PROVIDER.
                      Defaults to 'gemini' if not specified.
    
    Returns:
        LLMProvider: An instance of the requested provider.
    """
    if provider_name is None:
        provider_name = os.getenv('LLM_PROVIDER', 'gemini').lower()
    
    provider_name = provider_name.lower()
    
    if provider_name == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        provider = OpenAIProvider(api_key)
    elif provider_name == 'gemini':
        api_key = os.getenv('GEMINI_API_KEY')
        provider = GeminiProvider(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported providers: openai, gemini")
    
    provider.initialize()
    return provider

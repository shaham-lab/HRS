"""
LLM Provider Module for Health Recommendation System.

This module provides a central import point for all LLM provider classes.
For backward compatibility, it re-exports all provider classes from their
individual modules.
"""

import os
from llm_provider import LLMProvider
from openai_provider import OpenAIProvider
from gemini_provider import GeminiProvider


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

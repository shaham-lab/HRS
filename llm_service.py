"""
LLM Service Module for Health Recommendation System.

This module handles all interactions with LLM providers, including:
- Provider initialization and selection
- Input sanitization and validation
- Prompt construction
- API calls to generate health recommendations
"""

import os
import re
from dotenv import load_dotenv
from llm_provider import LLMProvider
from openai_provider import OpenAIProvider
from gemini_provider import GeminiProvider

load_dotenv()


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


# Initialize LLM provider (defaults to Gemini)
provider = get_provider()


def sanitize_input(text, max_length=1000):
    """Sanitize user input to prevent injection attacks."""
    if not text:
        return ""
    # Remove any potential control characters and limit length
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text[:max_length].strip()


def validate_severity(severity):
    """Validate severity input."""
    try:
        severity_int = int(severity)
        if 1 <= severity_int <= 10:
            return severity_int
        return 5  # Default to middle if out of range
    except (ValueError, TypeError):
        return 5  # Default to middle if invalid


def get_health_recommendation(symptoms, duration, severity, additional_info):
    """Get health recommendations from the configured LLM provider."""
    # Sanitize all inputs
    symptoms = sanitize_input(symptoms, 500)
    duration = sanitize_input(duration, 100)
    additional_info = sanitize_input(additional_info, 500)
    severity = validate_severity(severity)
    
    # Construct the prompt
    prompt = f"""You are a helpful medical assistant. Based on the following patient information, provide general health recommendations and suggestions. Remember to always advise consulting a healthcare professional.

Symptoms: {symptoms}
Duration: {duration}
Severity: {severity}/10
Additional Information: {additional_info if additional_info else 'None provided'}

Please provide:
1. A brief assessment of the symptoms
2. Possible causes (general information only)
3. Self-care recommendations
4. When to seek immediate medical attention
5. General lifestyle advice

Keep the response clear, concise, and easy to understand."""

    system_message = "You are a helpful medical assistant who provides general health information and recommendations. Always remind users to consult healthcare professionals for proper diagnosis and treatment."
    
    # Get response from provider
    return provider.generate_response(prompt, system_message)

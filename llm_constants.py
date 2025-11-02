"""
LLM Constants Module for Health Recommendation System.

This module contains all generic prompt and reply texts used across LLM providers.
"""

# Demo mode response template
DEMO_MODE_RESPONSE = """**DEMO MODE - No {provider_name} API key configured**

Based on your query, this is a demonstration response. To get real AI-powered health recommendations, please set up your {provider_name} API key in the .env file.

**General Recommendations:**
1. Monitor your symptoms and note any changes
2. Stay hydrated and get adequate rest
3. Maintain a healthy diet
4. Take over-the-counter medications as appropriate

**When to Seek Medical Attention:**
- If symptoms persist for more than a few days
- If symptoms worsen or become severe
- If you experience any concerning or unusual symptoms

**Important:** This demo response is not actual medical advice. For real health recommendations, configure your {provider_name} API key. Always consult with a qualified healthcare professional for proper diagnosis and treatment.
"""

# Error messages
ERROR_RESPONSE_TEMPLATE = "Unable to generate health recommendations at this time. Please try again later or ensure your {provider_name} API key is configured correctly."
ERROR_EMPTY_RESPONSE = "Unable to generate health recommendations at this time. The AI service returned an empty response. Please try again."

# Initialization error messages
ERROR_INIT_TEMPLATE = "Error initializing {provider_name} client: {error}"

# Generation error messages
ERROR_GENERATION_TEMPLATE = "Error generating {provider_name} response: {error}"

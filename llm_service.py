"""
LLM Service Module for Health Recommendation System.

This module handles all interactions with the OpenAI API, including:
- Client initialization
- Input sanitization and validation
- Prompt construction
- API calls to generate health recommendations
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import Iterable, Any, cast

load_dotenv()

# Initialize OpenAI client (will be None if no API key is set)
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key) if api_key and api_key != 'your_openai_api_key_here' else None


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
    """Get health recommendations from OpenAI."""
    # Sanitize all inputs
    symptoms = sanitize_input(symptoms, 500)
    duration = sanitize_input(duration, 100)
    additional_info = sanitize_input(additional_info, 500)
    severity = validate_severity(severity)
    
    # Check if OpenAI client is initialized
    if client is None:
        # Return a demo response if no API key is configured
        return f"""**DEMO MODE - No OpenAI API key configured**

Based on your symptoms: {symptoms}

**Assessment:**
This is a demonstration response. To get real AI-powered health recommendations, please set up your OpenAI API key in the .env file.

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
    
    try:
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

        messages = [
            {"role": "system",
             "content": "You are a helpful medical assistant who provides general health information and recommendations. Always remind users to consult healthcare professionals for proper diagnosis and treatment."},
            {"role": "user", "content": prompt}
        ]
        messages_typed = cast(Iterable[Any], messages)
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_typed,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        # Log the error for debugging but don't expose details to users
        print(f"Error getting recommendation: {str(e)}")
        return "Unable to generate health recommendations at this time. Please try again later or ensure your OpenAI API key is configured correctly."

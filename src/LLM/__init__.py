"""
LLM Module for Health Recommendation System.

This module contains all LLM-related functionality including:
- LLM providers (OpenAI, Gemini)
- LLM service and integration
- RAG (Retrieval-Augmented Generation) service
- Medical knowledge base
"""

from .llm_provider import LLMProvider
from .llm_service import get_health_recommendation, get_provider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from src.RAG.rag_service import RAGService, get_rag_service
from src.RAG.medical_knowledge import MEDICAL_DOCUMENTS

__all__ = [
    'LLMProvider',
    'get_health_recommendation',
    'get_provider',
    'OpenAIProvider',
    'GeminiProvider',
    'RAGService',
    'get_rag_service',
    'MEDICAL_DOCUMENTS',
]

"""
RAG Module for Health Recommendation System.

This module contains RAG (Retrieval-Augmented Generation) functionality including:
- RAG service for document retrieval and context augmentation
- Medical knowledge base
"""

from .rag_service import RAGService, get_rag_service, RAG_PROMPT_TEMPLATE
from .medical_knowledge import MEDICAL_DOCUMENTS

__all__ = [
    'RAGService',
    'get_rag_service',
    'RAG_PROMPT_TEMPLATE',
    'MEDICAL_DOCUMENTS',
]

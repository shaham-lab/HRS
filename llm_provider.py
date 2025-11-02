"""
LLM Provider Base Class Module for Health Recommendation System.

This module defines the abstract LLMProvider base class.
"""

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

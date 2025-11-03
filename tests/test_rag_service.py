"""
Unit tests for RAG service module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
from RAG.rag_service import RAGService, get_rag_service, RAG_PROMPT_TEMPLATE


class TestRAGService(unittest.TestCase):
    """Test cases for RAG service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rag_service = RAGService()
    
    def test_rag_service_initialization(self):
        """Test that RAG service initializes correctly."""
        self.assertIsNotNone(self.rag_service)
        self.assertEqual(self.rag_service.collection_name, "medical_knowledge")
        self.assertIsNone(self.rag_service.client)
        self.assertIsNone(self.rag_service.collection)
        self.assertIsNone(self.rag_service.embedding_model)
        self.assertFalse(self.rag_service.initialized)
    
    def test_rag_service_custom_collection_name(self):
        """Test RAG service with custom collection name."""
        custom_service = RAGService(collection_name="test_collection")
        self.assertEqual(custom_service.collection_name, "test_collection")
    
    @patch('RAG.rag_service.chromadb.PersistentClient')
    @patch('RAG.rag_service.SentenceTransformer')
    @patch('os.makedirs')
    def test_initialize_success(self, mock_makedirs, mock_transformer, mock_chromadb):
        """Test successful initialization of RAG service."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 10  # Pretend we have documents
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client
        mock_transformer.return_value = Mock()
        
        # Initialize service
        result = self.rag_service.initialize()
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(self.rag_service.initialized)
        mock_makedirs.assert_called_once()
    
    @patch('RAG.rag_service.chromadb.PersistentClient')
    @patch('RAG.rag_service.SentenceTransformer')
    def test_initialize_model_loading_failure(self, mock_transformer, mock_chromadb):
        """Test initialization fails gracefully when model loading fails."""
        # Setup mocks to fail on model loading
        mock_transformer.side_effect = Exception("Model loading failed")
        
        # Initialize service
        result = self.rag_service.initialize()
        
        # Verify graceful failure
        self.assertFalse(result)
        self.assertFalse(self.rag_service.initialized)
    
    @patch('RAG.rag_service.chromadb.PersistentClient')
    @patch('RAG.rag_service.SentenceTransformer')
    def test_initialize_chromadb_failure(self, mock_transformer, mock_chromadb):
        """Test initialization fails gracefully when ChromaDB fails."""
        # Setup mocks to fail on ChromaDB
        mock_chromadb.side_effect = Exception("ChromaDB connection failed")
        
        # Initialize service
        result = self.rag_service.initialize()
        
        # Verify graceful failure
        self.assertFalse(result)
        self.assertFalse(self.rag_service.initialized)
    
    def test_retrieve_context_when_not_initialized(self):
        """Test that retrieve_relevant_context returns empty string when not initialized."""
        result = self.rag_service.retrieve_relevant_context("test query")
        self.assertEqual(result, "")
    
    def test_augment_prompt_when_not_initialized(self):
        """Test that augment_prompt returns original prompt when not initialized."""
        original_prompt = "Test prompt"
        result = self.rag_service.augment_prompt_with_context(
            original_prompt, "symptoms"
        )
        self.assertEqual(result, original_prompt)
    
    @patch.object(RAGService, 'retrieve_relevant_context')
    def test_augment_prompt_with_context(self, mock_retrieve):
        """Test prompt augmentation with retrieved context."""
        # Setup
        self.rag_service.initialized = True
        mock_retrieve.return_value = "Medical context here"
        original_prompt = "Original prompt"
        
        # Execute
        result = self.rag_service.augment_prompt_with_context(
            original_prompt, "headache"
        )
        
        # Verify
        self.assertIn("Medical context here", result)
        self.assertIn("Original prompt", result)
        self.assertIn("MEDICAL REFERENCE INFORMATION", result)
        mock_retrieve.assert_called_once_with("headache", top_k=3)
    
    @patch.object(RAGService, 'retrieve_relevant_context')
    def test_augment_prompt_no_context_retrieved(self, mock_retrieve):
        """Test that original prompt is returned when no context is retrieved."""
        # Setup
        self.rag_service.initialized = True
        mock_retrieve.return_value = ""  # No context
        original_prompt = "Original prompt"
        
        # Execute
        result = self.rag_service.augment_prompt_with_context(
            original_prompt, "symptoms"
        )
        
        # Verify original prompt is returned unchanged
        self.assertEqual(result, original_prompt)


class TestGetRAGService(unittest.TestCase):
    """Test cases for get_rag_service function."""
    
    def setUp(self):
        """Reset global RAG service instance before each test."""
        import RAG.rag_service
        RAG.rag_service._rag_service = None
    
    @patch.dict('os.environ', {'RAG_ENABLED': 'false'})
    def test_get_rag_service_disabled(self):
        """Test that None is returned when RAG is disabled."""
        result = get_rag_service()
        self.assertIsNone(result)
    
    @patch.dict('os.environ', {'RAG_ENABLED': 'true'})
    @patch('RAG.rag_service.RAGService')
    def test_get_rag_service_enabled(self, mock_rag_service):
        """Test that RAG service is created when enabled."""
        mock_instance = Mock()
        mock_instance.initialize.return_value = True
        mock_rag_service.return_value = mock_instance
        
        result = get_rag_service()
        
        self.assertIsNotNone(result)
        mock_rag_service.assert_called_once()
        mock_instance.initialize.assert_called_once()
    
    @patch.dict('os.environ', {'RAG_ENABLED': 'true'})
    @patch('RAG.rag_service.RAGService')
    def test_get_rag_service_initialization_failure(self, mock_rag_service):
        """Test that None is returned when RAG service initialization fails."""
        mock_instance = Mock()
        mock_instance.initialize.return_value = False
        mock_rag_service.return_value = mock_instance
        
        result = get_rag_service()
        
        self.assertIsNone(result)
    
    @patch.dict('os.environ', {'RAG_ENABLED': '1'})
    @patch('RAG.rag_service.RAGService')
    def test_get_rag_service_enabled_numeric(self, mock_rag_service):
        """Test that '1' is treated as enabled."""
        mock_instance = Mock()
        mock_instance.initialize.return_value = True
        mock_rag_service.return_value = mock_instance
        
        result = get_rag_service()
        
        self.assertIsNotNone(result)
    
    @patch.dict('os.environ', {'RAG_ENABLED': 'yes'})
    @patch('RAG.rag_service.RAGService')
    def test_get_rag_service_enabled_yes(self, mock_rag_service):
        """Test that 'yes' is treated as enabled."""
        mock_instance = Mock()
        mock_instance.initialize.return_value = True
        mock_rag_service.return_value = mock_instance
        
        result = get_rag_service()
        
        self.assertIsNotNone(result)
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('RAG.rag_service.RAGService')
    def test_get_rag_service_default_enabled(self, mock_rag_service):
        """Test that RAG is enabled by default when env var is not set."""
        mock_instance = Mock()
        mock_instance.initialize.return_value = True
        mock_rag_service.return_value = mock_instance
        
        result = get_rag_service()
        
        self.assertIsNotNone(result)


class TestRAGPromptTemplate(unittest.TestCase):
    """Test cases for RAG prompt template."""
    
    def test_prompt_template_exists(self):
        """Test that RAG_PROMPT_TEMPLATE is defined."""
        self.assertIsNotNone(RAG_PROMPT_TEMPLATE)
    
    def test_prompt_template_has_placeholders(self):
        """Test that template has required placeholders."""
        self.assertIn('{context}', RAG_PROMPT_TEMPLATE)
        self.assertIn('{original_prompt}', RAG_PROMPT_TEMPLATE)
    
    def test_prompt_template_formatting(self):
        """Test that template can be formatted correctly."""
        context = "Test context"
        original_prompt = "Test prompt"
        
        result = RAG_PROMPT_TEMPLATE.format(
            context=context,
            original_prompt=original_prompt
        )
        
        self.assertIn(context, result)
        self.assertIn(original_prompt, result)
        self.assertNotIn('{context}', result)
        self.assertNotIn('{original_prompt}', result)


if __name__ == '__main__':
    unittest.main()

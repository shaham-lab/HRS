"""
Unit tests for medical knowledge base module.
"""

import unittest
from src.RAG.medical_knowledge import MEDICAL_DOCUMENTS


class TestMedicalKnowledge(unittest.TestCase):
    """Test cases for the medical knowledge base."""
    
    def test_medical_documents_exists(self):
        """Test that MEDICAL_DOCUMENTS is defined and accessible."""
        self.assertIsNotNone(MEDICAL_DOCUMENTS)
    
    def test_medical_documents_is_list(self):
        """Test that MEDICAL_DOCUMENTS is a list."""
        self.assertIsInstance(MEDICAL_DOCUMENTS, list)
    
    def test_medical_documents_not_empty(self):
        """Test that MEDICAL_DOCUMENTS contains documents."""
        self.assertGreater(len(MEDICAL_DOCUMENTS), 0)
    
    def test_all_documents_are_strings(self):
        """Test that all documents in MEDICAL_DOCUMENTS are strings."""
        for doc in MEDICAL_DOCUMENTS:
            self.assertIsInstance(doc, str)
    
    def test_documents_not_empty_strings(self):
        """Test that no document is an empty string."""
        for doc in MEDICAL_DOCUMENTS:
            self.assertTrue(len(doc.strip()) > 0)
    
    def test_documents_contain_medical_terms(self):
        """Test that documents contain relevant medical information."""
        # Combine all documents into one string for checking
        all_docs = " ".join(MEDICAL_DOCUMENTS).lower()
        
        # Check for presence of common medical terms
        medical_terms = [
            'symptoms', 'diagnosis', 'treatment', 'test', 'fever',
            'pain', 'headache', 'chest', 'blood', 'recommended'
        ]
        
        found_terms = [term for term in medical_terms if term in all_docs]
        # At least 5 of these terms should be present
        self.assertGreaterEqual(len(found_terms), 5)
    
    def test_specific_medical_conditions_covered(self):
        """Test that specific medical conditions are covered in the knowledge base."""
        all_docs = " ".join(MEDICAL_DOCUMENTS).lower()
        
        # Check for specific conditions that should be in the knowledge base
        conditions = ['headache', 'fever', 'chest pain', 'diabetes', 'hypertension']
        
        for condition in conditions:
            with self.subTest(condition=condition):
                self.assertIn(condition, all_docs,
                            f"Expected '{condition}' to be in medical knowledge base")
    
    def test_documents_have_reasonable_length(self):
        """Test that documents have a reasonable length (not too short, not too long)."""
        for i, doc in enumerate(MEDICAL_DOCUMENTS):
            with self.subTest(doc_index=i):
                doc_length = len(doc.strip())
                self.assertGreater(doc_length, 50,
                                 f"Document {i} is too short ({doc_length} chars)")
                self.assertLess(doc_length, 5000,
                              f"Document {i} is too long ({doc_length} chars)")


if __name__ == '__main__':
    unittest.main()

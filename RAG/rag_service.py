"""
RAG (Retrieval-Augmented Generation) Service Module.

This module handles the RAG functionality for the Health Recommendation System,
including document indexing, retrieval, and context augmentation for LLM queries.
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from .medical_knowledge import MEDICAL_DOCUMENTS


# Prompt template for RAG-augmented queries
RAG_PROMPT_TEMPLATE = """Use the following medical knowledge as reference to provide accurate recommendations:

MEDICAL REFERENCE INFORMATION:
{context}

---

{original_prompt}

Please provide recommendations based on both the medical reference information above and your medical knowledge."""


class RAGService:
    """Service for managing RAG operations."""
    
    def __init__(self, collection_name: str = "medical_knowledge"):
        """Initialize the RAG service.
        
        Args:
            collection_name: Name of the ChromaDB collection to use.
        """
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the RAG service components.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            # Initialize ChromaDB client with persistent storage
            persist_directory = os.getenv('CHROMADB_PATH', './chromadb_data')
            
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize embedding model
            # Try to load the model, handle offline scenarios
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            try:
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as model_error:
                print(f"Warning: Could not load embedding model '{model_name}': {str(model_error)}")
                print("RAG service will not be available. The system will function without RAG.")
                return False
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Medical knowledge base for health recommendations"}
                )
                print(f"Created new collection: {self.collection_name}")
                # Index the medical documents
                self._index_documents()
            
            self.initialized = True
            print("RAG service initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing RAG service: {str(e)}")
            print("The system will continue to function without RAG.")
            self.initialized = False
            return False
    
    def _index_documents(self) -> None:
        """Index medical documents into the vector store."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        try:
            # Check if collection already has documents
            if self.collection.count() > 0:
                print(f"Collection already contains {self.collection.count()} documents")
                return
            
            # Generate embeddings for all documents
            # Note: This is done synchronously on first initialization
            # The embeddings are cached in the persistent database for subsequent runs
            print("Generating embeddings for medical knowledge base...")
            embeddings = self.embedding_model.encode(
                MEDICAL_DOCUMENTS,
                convert_to_numpy=True,
                show_progress_bar=False  # Disable progress bar to avoid console clutter
            )
            
            # Add documents to collection
            ids = [f"doc_{i}" for i in range(len(MEDICAL_DOCUMENTS))]
            self.collection.add(
                documents=MEDICAL_DOCUMENTS,
                embeddings=embeddings.tolist(),
                ids=ids
            )
            
            print(f"Indexed {len(MEDICAL_DOCUMENTS)} medical documents")
        except Exception as e:
            print(f"Error indexing medical documents into vector store: {str(e)}")
            raise
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant medical context for a query.
        
        Args:
            query: The query text (e.g., patient symptoms).
            top_k: Number of top relevant documents to retrieve.
            
        Returns:
            str: Concatenated relevant medical context.
        """
        if not self.initialized or not self.collection:
            return ""
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True
            )
            
            # Retrieve top-k relevant documents
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Extract and format documents
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0]  # Get first query's results
                context = "\n\n".join(documents)
                return context
            
            return ""
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return ""
    
    def augment_prompt_with_context(
        self,
        original_prompt: str,
        symptoms: str,
        top_k: int = 3
    ) -> str:
        """Augment the original prompt with retrieved medical context.
        
        Args:
            original_prompt: The original LLM prompt.
            symptoms: Patient symptoms for context retrieval.
            top_k: Number of relevant documents to retrieve.
            
        Returns:
            str: Augmented prompt with medical context.
        """
        if not self.initialized:
            # If RAG is not initialized, return original prompt
            return original_prompt
        
        # Retrieve relevant context
        context = self.retrieve_relevant_context(symptoms, top_k=top_k)
        
        if not context:
            # If no context retrieved, return original prompt
            return original_prompt
        
        # Augment prompt with context using template
        augmented_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            original_prompt=original_prompt
        )
        
        return augmented_prompt


# Global RAG service instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> Optional[RAGService]:
    """Get or create the global RAG service instance.
    
    Returns:
        RAGService or None: The RAG service instance if enabled, None otherwise.
    """
    global _rag_service
    
    # Check if RAG is enabled via environment variable
    rag_enabled = os.getenv('RAG_ENABLED', 'true').lower() in ('true', '1', 'yes')
    
    if not rag_enabled:
        return None
    
    if _rag_service is None:
        _rag_service = RAGService()
        if not _rag_service.initialize():
            print("Warning: RAG service failed to initialize. Running without RAG.")
            _rag_service = None
    
    return _rag_service

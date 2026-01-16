"""
ChromaDB-based vector store for FAQ documents.
"""
import json
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB-backed vector store with semantic search."""

    def __init__(self, persist_directory: str = None):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Optional directory to persist data (None for in-memory)
        """
        # Use in-memory mode by default
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="faq_documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self.documents: List[Dict[str, str]] = []

    def load_documents(self, doc_path: str):
        """
        Load FAQ documents from JSON file into ChromaDB.

        Args:
            doc_path: Path to the FAQ documents JSON file
        """
        print(f"Loading documents from {doc_path}")

        # Load documents
        with open(doc_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        print(f"Loaded {len(self.documents)} documents")

        # Clear existing collection if it has data
        if self.collection.count() > 0:
            # Delete the collection and recreate it
            self.client.delete_collection(name="faq_documents")
            self.collection = self.client.create_collection(
                name="faq_documents",
                metadata={"hnsw:space": "cosine"}
            )

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for doc in self.documents:
            ids.append(doc['id'])
            # Combine title and content for better semantic search
            documents.append(f"{doc['title']}\n{doc['content']}")
            metadatas.append({
                'title': doc['title'],
                'category': doc['category'],
                'content': doc['content']
            })

        # Add to ChromaDB (it auto-generates embeddings)
        print("Adding documents to ChromaDB (generating embeddings)...")
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Added {len(ids)} documents to ChromaDB")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents most similar to the query using ChromaDB.

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of documents with similarity scores, sorted by relevance
        """
        if self.collection.count() == 0:
            raise ValueError("Vector store not initialized. Call load_documents() first.")

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]

            # Convert distance to similarity score (cosine distance to similarity)
            similarity_score = 1 - distance

            formatted_results.append({
                'id': doc_id,
                'title': metadata['title'],
                'category': metadata['category'],
                'content': metadata['content'],
                'similarity_score': float(similarity_score),
                'rank': i + 1
            })

        return formatted_results

    def get_document(self, doc_id: str) -> Dict[str, str]:
        """
        Retrieve a specific document by ID.

        Args:
            doc_id: Document ID to retrieve

        Returns:
            Document dictionary or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result['ids']:
                metadata = result['metadatas'][0]
                return {
                    'id': doc_id,
                    'title': metadata['title'],
                    'category': metadata['category'],
                    'content': metadata['content']
                }
        except Exception:
            pass
        return None

    def search_by_category(self, category: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Get documents from a specific category.

        Args:
            category: Category name to filter by
            top_k: Maximum number of documents to return

        Returns:
            List of documents in the category
        """
        results = self.collection.get(
            where={"category": category},
            limit=top_k
        )

        formatted_results = []
        for i in range(len(results['ids'])):
            doc_id = results['ids'][i]
            metadata = results['metadatas'][i]
            formatted_results.append({
                'id': doc_id,
                'title': metadata['title'],
                'category': metadata['category'],
                'content': metadata['content']
            })

        return formatted_results

    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories in the corpus."""
        # Get all documents
        all_docs = self.collection.get()
        categories = set()
        for metadata in all_docs['metadatas']:
            categories.add(metadata.get('category', 'unknown'))
        return sorted(list(categories))


# Global vector store instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """
    Get or create the global vector store instance.

    Returns:
        Initialized vector store
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        # Load documents from default path
        default_path = Path(__file__).parent.parent.parent / "data" / "faq_docs.json"
        if default_path.exists():
            _vector_store.load_documents(str(default_path))
        else:
            print(f"Warning: FAQ documents not found at {default_path}")
    return _vector_store


def initialize_vector_store(doc_path: str, persist_directory: str = None) -> VectorStore:
    """
    Initialize vector store with documents from specified path.

    Args:
        doc_path: Path to FAQ documents JSON
        persist_directory: Optional directory to persist ChromaDB data

    Returns:
        Initialized vector store
    """
    global _vector_store
    _vector_store = VectorStore(persist_directory=persist_directory)
    _vector_store.load_documents(doc_path)
    return _vector_store

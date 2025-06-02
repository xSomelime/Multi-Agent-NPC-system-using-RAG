#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) System for NPC Knowledge
Implements vector-based knowledge retrieval with strict relevance filtering
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logging.error(f"Required packages missing: {e}")
    logging.error("Install with: pip install scikit-learn sentence-transformers")
    raise

@dataclass
class KnowledgeChunk:
    """Represents a chunk of knowledge with metadata"""
    id: str
    content: str
    topic: str
    domain: str  # stable_hand, trainer, behaviourist, rival
    confidence: str  # high, medium, low
    source: str
    keywords: List[str]
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    """Result from RAG retrieval with confidence and source attribution"""
    chunk: KnowledgeChunk
    similarity_score: float
    relevance_score: float  # Adjusted score after filtering
    is_relevant: bool

class RAGKnowledgeBase:
    """
    Vector-based knowledge base using scikit-learn for efficient retrieval
    Implements strict relevance filtering to prevent hallucination
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 relevance_threshold: float = 0.4,
                 max_results: int = 5):
        """
        Initialize RAG system with sentence transformer and scikit-learn index
        
        Args:
            model_name: Sentence transformer model for embeddings
            relevance_threshold: Minimum similarity score for relevance
            max_results: Maximum number of results to return
        """
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold
        self.max_results = max_results
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logging.error(f"Failed to load embedding model {model_name}: {e}")
            raise
        
        # Initialize scikit-learn index
        self.index = NearestNeighbors(n_neighbors=max_results*2, metric='cosine', algorithm='brute')
        self.knowledge_chunks: List[KnowledgeChunk] = []
        self.domain_filters: Dict[str, List[int]] = {}  # Domain to chunk indices mapping
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.is_fitted = False
        
        logging.info(f"RAG system initialized with {model_name}, embedding dim: {self.embedding_dim}")
    
    def load_knowledge_from_json(self, knowledge_dir: str = "src/knowledge") -> int:
        """
        Load knowledge from JSON files and create embeddings
        
        Args:
            knowledge_dir: Directory containing knowledge JSON files
            
        Returns:
            Number of chunks loaded
        """
        knowledge_files = [
            "stable_hand_knowledge.json",
            "trainer_knowledge.json", 
            "behaviourist_knowledge.json",
            "rival_knowledge.json"
        ]
        
        total_chunks = 0
        
        for filename in knowledge_files:
            filepath = Path(knowledge_dir) / filename
            
            if not filepath.exists():
                logging.warning(f"Knowledge file not found: {filepath}")
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                domain = data.get('domain', 'unknown')
                chunks_loaded = self._process_knowledge_data(data, domain)
                total_chunks += chunks_loaded
                
                logging.info(f"Loaded {chunks_loaded} chunks from {filename}")
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logging.error(f"Error loading {filepath}: {e}")
                continue
        
        if total_chunks > 0:
            self._build_sklearn_index()
            logging.info(f"RAG knowledge base built with {total_chunks} total chunks")
        
        return total_chunks
    
    def _process_knowledge_data(self, data: Dict[str, Any], domain: str) -> int:
        """Process knowledge data from JSON and create chunks"""
        chunks_added = 0
        
        # Process knowledge chunks
        for chunk_data in data.get('knowledge_chunks', []):
            chunk = KnowledgeChunk(
                id=chunk_data.get('id', f"{domain}_{chunks_added}"),
                content=chunk_data.get('content', ''),
                topic=chunk_data.get('topic', 'general'),
                domain=domain,
                confidence=chunk_data.get('confidence', 'medium'),
                source=chunk_data.get('source', 'unknown'),
                keywords=data.get('expertise_keywords', [])
            )
            
            # Skip empty content
            if not chunk.content.strip():
                continue
                
            # Create embedding
            try:
                embedding = self.embedding_model.encode(chunk.content, convert_to_numpy=True)
                chunk.embedding = embedding
                
                self.knowledge_chunks.append(chunk)
                chunks_added += 1
                
                # Update domain filter mapping
                if domain not in self.domain_filters:
                    self.domain_filters[domain] = []
                self.domain_filters[domain].append(len(self.knowledge_chunks) - 1)
                
            except Exception as e:
                logging.error(f"Error creating embedding for chunk {chunk.id}: {e}")
                continue
        
        return chunks_added
    
    def _build_sklearn_index(self):
        """Build scikit-learn index from embeddings"""
        if not self.knowledge_chunks:
            logging.warning("No knowledge chunks to index")
            return
            
        # Collect all embeddings
        self.embeddings_matrix = np.array([chunk.embedding for chunk in self.knowledge_chunks])
        
        # Fit the index
        self.index.fit(self.embeddings_matrix)
        self.is_fitted = True
        
        logging.info(f"Scikit-learn index built with {len(self.knowledge_chunks)} vectors")
    
    def retrieve(self, 
                 query: str, 
                 domain_filter: Optional[str] = None,
                 require_keywords: bool = True) -> List[RetrievalResult]:
        """
        Retrieve relevant knowledge chunks for a query
        
        Args:
            query: Search query
            domain_filter: Specific domain to search (stable_hand, trainer, etc.)
            require_keywords: Whether to boost results matching query keywords
            
        Returns:
            List of relevant knowledge chunks with similarity scores
        """
        if not self.knowledge_chunks or not self.is_fitted:
            logging.warning("No knowledge chunks loaded or index not fitted")
            return []
        
        # Create query embedding
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1)
        except Exception as e:
            logging.error(f"Error creating query embedding: {e}")
            return []
        
        # Search using scikit-learn
        distances, indices = self.index.kneighbors(query_embedding, n_neighbors=min(self.max_results * 2, len(self.knowledge_chunks)))
        
        results = []
        query_keywords = set(query.lower().split())
        
        for distance, idx in zip(distances[0], indices[0]):
            chunk = self.knowledge_chunks[idx]
            
            # Apply domain filter if specified
            if domain_filter and chunk.domain != domain_filter:
                continue
            
            # Convert distance to similarity (cosine distance -> cosine similarity)
            similarity = 1 - distance
            
            # Calculate relevance score with keyword boosting
            relevance_score = float(similarity)
            
            if require_keywords:
                # Boost score if query keywords match chunk keywords or content
                keyword_matches = 0
                chunk_text = (chunk.content + " " + " ".join(chunk.keywords)).lower()
                
                for keyword in query_keywords:
                    if keyword in chunk_text or keyword in chunk.topic.lower():
                        keyword_matches += 1
                
                if keyword_matches > 0:
                    keyword_boost = min(0.2, keyword_matches * 0.05)
                    relevance_score += keyword_boost
            
            # Apply relevance threshold
            is_relevant = relevance_score >= self.relevance_threshold
            
            result = RetrievalResult(
                chunk=chunk,
                similarity_score=float(similarity),
                relevance_score=relevance_score,
                is_relevant=is_relevant
            )
            
            results.append(result)
        
        # Filter and sort by relevance
        relevant_results = [r for r in results if r.is_relevant]
        relevant_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return relevant_results[:self.max_results]
    
    def get_domain_expertise(self, domain: str) -> List[str]:
        """Get expertise keywords for a specific domain"""
        domain_chunks = [chunk for chunk in self.knowledge_chunks if chunk.domain == domain]
        
        if not domain_chunks:
            return []
        
        # Combine all keywords from domain chunks
        all_keywords = set()
        for chunk in domain_chunks:
            all_keywords.update(chunk.keywords)
        
        return sorted(list(all_keywords))
    
    def has_knowledge_about(self, topic: str, domain: Optional[str] = None, min_confidence: float = 0.3) -> bool:
        """
        Check if knowledge base has information about a topic
        
        Args:
            topic: Topic to check
            domain: Specific domain to check
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if relevant knowledge exists
        """
        results = self.retrieve(topic, domain_filter=domain)
        
        for result in results:
            if result.relevance_score >= min_confidence:
                return True
        
        return False
    
    def get_fallback_response(self, topic: str, domain: str) -> str:
        """
        Generate appropriate fallback response when no relevant knowledge found
        Following .cursorrules: fallback to "I don't know" rather than false information
        """
        domain_expertise = self.get_domain_expertise(domain)
        
        if domain_expertise:
            return f"I don't have specific information about {topic}, but I can help with {', '.join(domain_expertise[:3])} and related topics."
        else:
            return f"I don't have information about {topic} right now."
    
    def save_index(self, filepath: str):
        """Save scikit-learn index and metadata to disk"""
        try:
            # Save index and metadata
            index_data = {
                'knowledge_chunks': self.knowledge_chunks,
                'domain_filters': self.domain_filters,
                'model_name': self.model_name,
                'relevance_threshold': self.relevance_threshold,
                'embeddings_matrix': self.embeddings_matrix,
                'is_fitted': self.is_fitted
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(index_data, f)
                
            logging.info(f"RAG index saved to {filepath}.pkl")
            
        except Exception as e:
            logging.error(f"Error saving RAG index: {e}")
    
    def load_index(self, filepath: str) -> bool:
        """Load scikit-learn index and metadata from disk"""
        try:
            with open(f"{filepath}.pkl", 'rb') as f:
                index_data = pickle.load(f)
            
            self.knowledge_chunks = index_data['knowledge_chunks']
            self.domain_filters = index_data['domain_filters']
            self.model_name = index_data['model_name']
            self.relevance_threshold = index_data['relevance_threshold']
            self.embeddings_matrix = index_data.get('embeddings_matrix')
            self.is_fitted = index_data.get('is_fitted', False)
            
            # Rebuild index if we have embeddings
            if self.embeddings_matrix is not None and self.is_fitted:
                self.index.fit(self.embeddings_matrix)
            
            logging.info(f"RAG index loaded from {filepath}.pkl")
            return True
            
        except Exception as e:
            logging.error(f"Error loading RAG index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        stats = {
            'total_chunks': len(self.knowledge_chunks),
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'relevance_threshold': self.relevance_threshold,
            'domains': {}
        }
        
        for domain, indices in self.domain_filters.items():
            stats['domains'][domain] = {
                'chunk_count': len(indices),
                'topics': list(set(self.knowledge_chunks[i].topic for i in indices))
            }
        
        return stats

# Global RAG instance - lazy loaded
_rag_instance: Optional[RAGKnowledgeBase] = None

def get_rag_system() -> RAGKnowledgeBase:
    """Get or create global RAG system instance"""
    global _rag_instance
    
    if _rag_instance is None:
        _rag_instance = RAGKnowledgeBase()
        
        # Try to load existing index first
        index_path = "data/rag_index"
        if not _rag_instance.load_index(index_path):
            # Build new index from knowledge files
            chunks_loaded = _rag_instance.load_knowledge_from_json()
            if chunks_loaded > 0:
                # Save for future use
                os.makedirs("data", exist_ok=True)
                _rag_instance.save_index(index_path)
    
    return _rag_instance

if __name__ == "__main__":
    # Test the RAG system
    logging.basicConfig(level=logging.INFO)
    
    rag = RAGKnowledgeBase()
    chunks_loaded = rag.load_knowledge_from_json()
    
    print(f"Loaded {chunks_loaded} knowledge chunks")
    print(f"RAG system stats: {rag.get_stats()}")
    
    # Test queries
    test_queries = [
        "horse feeding schedule",
        "saddle fitting problems", 
        "competition preparation",
        "horse behavior analysis"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rag.retrieve(query)
        
        for result in results:
            print(f"  - {result.chunk.domain}: {result.chunk.content[:100]}... (score: {result.relevance_score:.3f})") 
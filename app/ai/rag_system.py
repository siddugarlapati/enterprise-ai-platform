import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import hashlib
import json

from app.config.settings import settings

logger = logging.getLogger(__name__)

class RAGSystem:
    """Retrieval Augmented Generation system for knowledge-based Q&A."""
    
    def __init__(self):
        self.embedding_model = None
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_size = 512
        self.chunk_overlap = 50
        
    def _get_embedding_model(self) -> SentenceTransformer:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self.embedding_model
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            
            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": metadata or {},
                "start_idx": i,
                "end_idx": min(i + self.chunk_size, len(words))
            })
        
        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the knowledge base."""
        model = self._get_embedding_model()
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            chunks = self._chunk_text(content, metadata)
            self.documents.extend(chunks)
        
        # Generate embeddings for all chunks
        texts = [d["content"] for d in self.documents]
        self.embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        logger.info(f"Added {len(documents)} documents, total chunks: {len(self.documents)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        if not self.documents or self.embeddings is None:
            return []
        
        model = self._get_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(similarities[idx]),
                "chunk_id": self.documents[idx]["id"]
            })
        
        return results

    def generate_context(self, query: str, top_k: int = 5, max_context_length: int = 2000) -> str:
        """Generate context from retrieved documents."""
        retrieved = self.retrieve(query, top_k)
        
        context_parts = []
        current_length = 0
        
        for doc in retrieved:
            content = doc["content"]
            if current_length + len(content) > max_context_length:
                break
            context_parts.append(content)
            current_length += len(content)
        
        return "\n\n".join(context_parts)

    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a RAG query and return answer with sources."""
        retrieved_docs = self.retrieve(query, top_k)
        context = self.generate_context(query, top_k)
        
        # Simple extractive answer (in production, use an LLM)
        answer = self._extract_answer(query, context)
        
        avg_confidence = np.mean([d["score"] for d in retrieved_docs]) if retrieved_docs else 0.0
        
        return {
            "query": query,
            "answer": answer,
            "sources": retrieved_docs,
            "confidence": float(avg_confidence),
            "context_used": context[:500] + "..." if len(context) > 500 else context
        }

    def _extract_answer(self, query: str, context: str) -> str:
        """Extract answer from context (simplified)."""
        if not context:
            return "No relevant information found in the knowledge base."
        
        # Return most relevant sentence
        sentences = context.split('.')
        if sentences:
            model = self._get_embedding_model()
            query_emb = model.encode([query])[0]
            sent_embs = model.encode(sentences)
            
            similarities = cosine_similarity([query_emb], sent_embs)[0]
            best_idx = np.argmax(similarities)
            
            return sentences[best_idx].strip() + "."
        
        return context[:500]

    def clear(self):
        """Clear the knowledge base."""
        self.documents = []
        self.embeddings = None
        logger.info("Knowledge base cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

    def export_index(self) -> Dict[str, Any]:
        """Export the index for persistence."""
        return {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }

    def import_index(self, data: Dict[str, Any]):
        """Import a previously exported index."""
        self.documents = data.get("documents", [])
        embeddings = data.get("embeddings")
        self.embeddings = np.array(embeddings) if embeddings else None
        logger.info(f"Imported index with {len(self.documents)} documents")

# Singleton instance
rag_system = RAGSystem()

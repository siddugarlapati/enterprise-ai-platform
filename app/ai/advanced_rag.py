"""
Advanced RAG System with re-ranking, multi-query, and hybrid search.
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio

from app.ai.vector_store import vector_store
from app.ai.llm_chat_engine import llm_chat_engine
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)


class AdvancedRAG:
    """Advanced Retrieval-Augmented Generation system."""
    
    def __init__(self):
        self.embedding_model = None
        self.reranker = None
        self.chunk_size = 512
        self.chunk_overlap = 50
        
    def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model
    
    def _get_reranker(self) -> CrossEncoder:
        """Lazy load cross-encoder for re-ranking."""
        if self.reranker is None:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return self.reranker
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with parent-child tracking.
        """
        import hashlib
        
        words = text.split()
        chunks = []
        parent_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            
            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    **(metadata or {}),
                    "parent_id": parent_id,
                    "chunk_index": len(chunks),
                    "start_idx": i,
                    "end_idx": min(i + self.chunk_size, len(words))
                }
            })
        
        return chunks
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add documents to vector store with chunking.
        
        Args:
            documents: List of {"content": str, "metadata": dict}
            collection: Optional collection name
        """
        model = self._get_embedding_model()
        
        all_chunks = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        texts = [chunk["content"] for chunk in all_chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Add to vector store
        doc_ids = vector_store.add_documents(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            collection=collection
        )
        
        logger.info(f"Added {len(documents)} documents ({len(all_chunks)} chunks)")
        
        return {
            "documents_added": len(documents),
            "chunks_created": len(all_chunks),
            "doc_ids": doc_ids
        }
    
    async def multi_query_retrieval(
        self,
        query: str,
        num_queries: int = 3,
        top_k: int = 5,
        collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple query variations and retrieve from each.
        """
        # Generate query variations using LLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates alternative phrasings of questions."
            },
            {
                "role": "user",
                "content": f"Generate {num_queries} different ways to ask this question: '{query}'\n\nReturn only the questions, one per line."
            }
        ]
        
        try:
            result = await llm_chat_engine.chat_completion(messages, max_tokens=200)
            variations = [query] + result["content"].strip().split('\n')[:num_queries]
        except Exception as e:
            logger.warning(f"Failed to generate query variations: {e}")
            variations = [query]
        
        # Retrieve for each variation
        model = self._get_embedding_model()
        all_results = []
        seen_ids = set()
        
        for var in variations:
            query_embedding = model.encode([var])[0].tolist()
            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                collection=collection
            )
            
            for result in results:
                if result["id"] not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result["id"])
        
        return all_results[:top_k * 2]  # Return more for re-ranking
    
    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using cross-encoder.
        """
        if not results:
            return []
        
        reranker = self._get_reranker()
        
        # Prepare pairs for re-ranking
        pairs = [[query, result["content"]] for result in results]
        
        # Get re-ranking scores
        scores = reranker.predict(pairs)
        
        # Sort by score
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results[:top_k]
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search.
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic search (0-1)
            collection: Collection name
        """
        model = self._get_embedding_model()
        
        # Semantic search
        query_embedding = model.encode([query])[0].tolist()
        semantic_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            collection=collection
        )
        
        # Keyword search (simple TF-IDF-like scoring)
        query_terms = set(query.lower().split())
        for result in semantic_results:
            content_terms = set(result["content"].lower().split())
            keyword_score = len(query_terms & content_terms) / len(query_terms) if query_terms else 0
            
            # Combine scores
            result["hybrid_score"] = (
                semantic_weight * result["score"] +
                (1 - semantic_weight) * keyword_score
            )
        
        # Sort by hybrid score
        semantic_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return semantic_results[:top_k]
    
    async def query_with_citations(
        self,
        query: str,
        top_k: int = 5,
        use_multi_query: bool = True,
        use_reranking: bool = True,
        collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query with answer generation and source citations.
        
        Args:
            query: User question
            top_k: Number of sources to retrieve
            use_multi_query: Use multi-query retrieval
            use_reranking: Use re-ranking
            collection: Collection name
        """
        # Retrieve documents
        if use_multi_query:
            results = await self.multi_query_retrieval(query, top_k=top_k, collection=collection)
        else:
            model = self._get_embedding_model()
            query_embedding = model.encode([query])[0].tolist()
            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                collection=collection
            )
        
        # Re-rank if enabled
        if use_reranking and results:
            results = self.rerank_results(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        
        # Build context
        context_parts = []
        citations = []
        
        for i, result in enumerate(results):
            citation_id = i + 1
            context_parts.append(f"[{citation_id}] {result['content']}")
            citations.append({
                "id": citation_id,
                "content": result["content"],
                "metadata": result["metadata"],
                "score": result.get("rerank_score", result.get("score", 0))
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with LLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Always cite your sources using [number] notation."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a detailed answer with citations."
            }
        ]
        
        try:
            result = await llm_chat_engine.chat_completion(messages, max_tokens=500)
            answer = result["content"]
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            answer = "I couldn't generate an answer. Please try again."
            tokens_used = 0
        
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "tokens_used": tokens_used,
            "retrieval_method": "multi_query" if use_multi_query else "standard",
            "reranked": use_reranking
        }
    
    async def hypothetical_document_embeddings(
        self,
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        HyDE: Generate hypothetical answer and use it for retrieval.
        """
        # Generate hypothetical answer
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Generate a detailed answer to the question."
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        try:
            result = await llm_chat_engine.chat_completion(messages, max_tokens=300)
            hypothetical_doc = result["content"]
        except Exception as e:
            logger.warning(f"Failed to generate hypothetical document: {e}")
            hypothetical_doc = query
        
        # Use hypothetical document for retrieval
        model = self._get_embedding_model()
        query_embedding = model.encode([hypothetical_doc])[0].tolist()
        
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            collection=collection
        )
        
        return results
    
    def get_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return vector_store.get_stats(collection=collection)


# Singleton instance
advanced_rag = AdvancedRAG()

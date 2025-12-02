"""
Vector Database Manager supporting Qdrant, Chroma, and Pinecone.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import hashlib
import uuid

from app.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreProvider(str, Enum):
    QDRANT = "qdrant"
    CHROMA = "chroma"
    PINECONE = "pinecone"


class VectorStore:
    """Unified vector database interface."""
    
    def __init__(self, provider: VectorStoreProvider = VectorStoreProvider.QDRANT):
        self.provider = provider
        self._client = None
        self._collection_name = "ai_platform_vectors"
        
    def _get_client(self):
        """Lazy load vector store client."""
        if self._client is None:
            if self.provider == VectorStoreProvider.QDRANT:
                self._client = self._init_qdrant()
            elif self.provider == VectorStoreProvider.CHROMA:
                self._client = self._init_chroma()
            elif self.provider == VectorStoreProvider.PINECONE:
                self._client = self._init_pinecone()
        return self._client
    
    def _init_qdrant(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            url = getattr(settings, 'QDRANT_URL', 'http://localhost:6333')
            api_key = getattr(settings, 'QDRANT_API_KEY', None)
            
            client = QdrantClient(url=url, api_key=api_key)
            
            # Create collection if not exists
            collections = client.get_collections().collections
            if not any(c.name == self._collection_name for c in collections):
                client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {self._collection_name}")
            
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            return None
    
    def _init_chroma(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            
            persist_dir = getattr(settings, 'CHROMA_PERSIST_DIR', './chroma_db')
            client = chromadb.PersistentClient(path=persist_dir)
            
            logger.info("ChromaDB client initialized")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return None
    
    def _init_pinecone(self):
        """Initialize Pinecone client."""
        try:
            import pinecone
            
            api_key = getattr(settings, 'PINECONE_API_KEY', None)
            environment = getattr(settings, 'PINECONE_ENV', 'us-west1-gcp')
            
            if not api_key:
                raise ValueError("PINECONE_API_KEY not configured")
            
            pinecone.init(api_key=api_key, environment=environment)
            
            # Create index if not exists
            if self._collection_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self._collection_name,
                    dimension=384,
                    metric='cosine'
                )
                logger.info(f"Created Pinecone index: {self._collection_name}")
            
            return pinecone.Index(self._collection_name)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            return None
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        collection: Optional[str] = None
    ) -> List[str]:
        """
        Add documents with embeddings to vector store.
        
        Args:
            documents: List of {"content": str, "metadata": dict}
            embeddings: List of embedding vectors
            collection: Optional collection name
            
        Returns:
            List of document IDs
        """
        collection = collection or self._collection_name
        client = self._get_client()
        
        if not client:
            raise ValueError("Vector store client not initialized")
        
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings length mismatch")
        
        if self.provider == VectorStoreProvider.QDRANT:
            return self._add_qdrant(documents, embeddings, collection)
        elif self.provider == VectorStoreProvider.CHROMA:
            return self._add_chroma(documents, embeddings, collection)
        elif self.provider == VectorStoreProvider.PINECONE:
            return self._add_pinecone(documents, embeddings, collection)
    
    def _add_qdrant(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        collection: str
    ) -> List[str]:
        """Add to Qdrant."""
        from qdrant_client.models import PointStruct
        
        points = []
        doc_ids = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {})
                }
            ))
        
        self._client.upsert(collection_name=collection, points=points)
        logger.info(f"Added {len(points)} documents to Qdrant")
        
        return doc_ids
    
    def _add_chroma(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        collection: str
    ) -> List[str]:
        """Add to ChromaDB."""
        coll = self._client.get_or_create_collection(name=collection)
        
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        contents = [doc.get("content", "") for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        coll.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(doc_ids)} documents to ChromaDB")
        return doc_ids
    
    def _add_pinecone(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        collection: str
    ) -> List[str]:
        """Add to Pinecone."""
        vectors = []
        doc_ids = []
        
        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            vectors.append((
                doc_id,
                embedding,
                {
                    "content": doc.get("content", ""),
                    **doc.get("metadata", {})
                }
            ))
        
        self._client.upsert(vectors=vectors)
        logger.info(f"Added {len(vectors)} documents to Pinecone")
        
        return doc_ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_metadata: Metadata filters
            collection: Collection name
            
        Returns:
            List of {"id": str, "score": float, "content": str, "metadata": dict}
        """
        collection = collection or self._collection_name
        client = self._get_client()
        
        if not client:
            raise ValueError("Vector store client not initialized")
        
        if self.provider == VectorStoreProvider.QDRANT:
            return self._search_qdrant(query_embedding, top_k, filter_metadata, collection)
        elif self.provider == VectorStoreProvider.CHROMA:
            return self._search_chroma(query_embedding, top_k, filter_metadata, collection)
        elif self.provider == VectorStoreProvider.PINECONE:
            return self._search_pinecone(query_embedding, top_k, filter_metadata, collection)
    
    def _search_qdrant(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]],
        collection: str
    ) -> List[Dict[str, Any]]:
        """Search Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)
        
        results = self._client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter
        )
        
        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {})
            }
            for hit in results
        ]
    
    def _search_chroma(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]],
        collection: str
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB."""
        coll = self._client.get_collection(name=collection)
        
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        output = []
        for i in range(len(results['ids'][0])):
            output.append({
                "id": results['ids'][0][i],
                "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
            })
        
        return output
    
    def _search_pinecone(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]],
        collection: str
    ) -> List[Dict[str, Any]]:
        """Search Pinecone."""
        results = self._client.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_metadata,
            include_metadata=True
        )
        
        return [
            {
                "id": match.id,
                "score": float(match.score),
                "content": match.metadata.get("content", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "content"}
            }
            for match in results.matches
        ]
    
    def delete_documents(
        self,
        doc_ids: List[str],
        collection: Optional[str] = None
    ) -> bool:
        """Delete documents by IDs."""
        collection = collection or self._collection_name
        client = self._get_client()
        
        if not client:
            return False
        
        try:
            if self.provider == VectorStoreProvider.QDRANT:
                self._client.delete(collection_name=collection, points_selector=doc_ids)
            elif self.provider == VectorStoreProvider.CHROMA:
                coll = self._client.get_collection(name=collection)
                coll.delete(ids=doc_ids)
            elif self.provider == VectorStoreProvider.PINECONE:
                self._client.delete(ids=doc_ids)
            
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def get_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """Get collection statistics."""
        collection = collection or self._collection_name
        client = self._get_client()
        
        if not client:
            return {}
        
        try:
            if self.provider == VectorStoreProvider.QDRANT:
                info = self._client.get_collection(collection_name=collection)
                return {
                    "provider": "qdrant",
                    "collection": collection,
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count
                }
            elif self.provider == VectorStoreProvider.CHROMA:
                coll = self._client.get_collection(name=collection)
                return {
                    "provider": "chroma",
                    "collection": collection,
                    "count": coll.count()
                }
            elif self.provider == VectorStoreProvider.PINECONE:
                stats = self._client.describe_index_stats()
                return {
                    "provider": "pinecone",
                    "collection": collection,
                    "total_vector_count": stats.total_vector_count
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Singleton instance
vector_store = VectorStore()

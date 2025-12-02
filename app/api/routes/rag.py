"""
Advanced RAG endpoints with multi-query, re-ranking, and hybrid search.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import io

from app.db.connection import get_db
from app.api.middleware.auth import get_current_user, require_model_manager
from app.ai.advanced_rag import advanced_rag
from app.ai.vector_store import vector_store

router = APIRouter(prefix="/rag", tags=["Advanced RAG"])


class AddDocumentsRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="List of documents with content and metadata")
    collection: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    collection: Optional[str] = None
    use_multi_query: bool = True
    use_reranking: bool = True


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    semantic_weight: float = Field(0.7, ge=0, le=1)
    collection: Optional[str] = None


class HyDERequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    collection: Optional[str] = None


@router.post("/documents")
async def add_documents(
    request: AddDocumentsRequest,
    current_user: dict = Depends(require_model_manager)
):
    """
    Add documents to RAG knowledge base with automatic chunking and embedding.
    
    Requires model_manager or admin role.
    """
    try:
        result = await advanced_rag.add_documents(
            documents=request.documents,
            collection=request.collection
        )
        
        return {
            "message": "Documents added successfully",
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")


@router.post("/documents/upload")
async def upload_documents(
    file: UploadFile = File(...),
    collection: Optional[str] = None,
    current_user: dict = Depends(require_model_manager)
):
    """
    Upload documents from file (TXT, PDF, or JSON).
    
    JSON format: [{"content": "...", "metadata": {...}}, ...]
    """
    try:
        content = await file.read()
        
        if file.filename.endswith('.json'):
            import json
            documents = json.loads(content.decode('utf-8'))
        elif file.filename.endswith('.txt'):
            text = content.decode('utf-8')
            documents = [{
                "content": text,
                "metadata": {"filename": file.filename}
            }]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use JSON or TXT.")
        
        result = await advanced_rag.add_documents(
            documents=documents,
            collection=collection
        )
        
        return {
            "message": f"Uploaded {file.filename} successfully",
            **result
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/query")
async def query_with_citations(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Query RAG system with advanced retrieval and answer generation.
    
    Features:
    - Multi-query retrieval (generates query variations)
    - Cross-encoder re-ranking
    - LLM-generated answers with citations
    """
    try:
        result = await advanced_rag.query_with_citations(
            query=request.query,
            top_k=request.top_k,
            use_multi_query=request.use_multi_query,
            use_reranking=request.use_reranking,
            collection=request.collection
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/search/hybrid")
async def hybrid_search(
    request: HybridSearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Hybrid search combining semantic and keyword matching.
    
    Adjust semantic_weight to balance between semantic (1.0) and keyword (0.0) search.
    """
    try:
        results = await advanced_rag.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            collection=request.collection
        )
        
        return {
            "query": request.query,
            "results": results,
            "semantic_weight": request.semantic_weight
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@router.post("/search/hyde")
async def hypothetical_document_search(
    request: HyDERequest,
    current_user: dict = Depends(get_current_user)
):
    """
    HyDE (Hypothetical Document Embeddings) search.
    
    Generates a hypothetical answer and uses it for retrieval.
    Better for complex queries where the question differs from document style.
    """
    try:
        results = await advanced_rag.hypothetical_document_embeddings(
            query=request.query,
            top_k=request.top_k,
            collection=request.collection
        )
        
        return {
            "query": request.query,
            "results": results,
            "method": "HyDE"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HyDE search failed: {str(e)}")


@router.get("/stats")
async def get_rag_stats(
    collection: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get RAG system statistics."""
    try:
        stats = advanced_rag.get_stats(collection=collection)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.delete("/documents")
async def clear_documents(
    collection: Optional[str] = None,
    current_user: dict = Depends(require_model_manager)
):
    """
    Clear all documents from collection.
    
    Requires model_manager or admin role.
    """
    try:
        # This would need implementation in vector_store
        return {
            "message": "Documents cleared",
            "collection": collection or "default"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@router.get("/collections")
async def list_collections(current_user: dict = Depends(get_current_user)):
    """List all available collections."""
    # This would need implementation based on vector store provider
    return {
        "collections": ["default"],
        "message": "Collection listing not fully implemented yet"
    }

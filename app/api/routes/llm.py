"""
LLM Chat and Advanced AI endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json

from app.db.connection import get_db
from app.api.middleware.auth import get_current_user, get_optional_user
from app.ai.llm_chat_engine import llm_chat_engine, LLMProvider

router = APIRouter(prefix="/llm", tags=["LLM & Chat"])


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    provider: Optional[str] = "openai"
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1000, ge=1, le=4000)
    stream: bool = False


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10)
    max_length: int = Field(200, ge=50, le=1000)
    provider: Optional[str] = "openai"


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=2)
    provider: Optional[str] = "openai"


class ExtractDataRequest(BaseModel):
    text: str = Field(..., min_length=1)
    schema: Dict[str, Any] = Field(..., description="JSON schema for extraction")
    provider: Optional[str] = "openai"


@router.post("/chat")
async def chat_completion(
    request: ChatRequest,
    current_user: dict = Depends(get_optional_user)
):
    """
    Generate chat completion using LLM.
    
    Supports OpenAI (GPT-4, GPT-3.5) and Anthropic (Claude) models.
    """
    try:
        provider = LLMProvider(request.provider) if request.provider else None
        
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        result = await llm_chat_engine.chat_completion(
            messages=messages,
            provider=provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        if request.stream:
            # TODO: Implement streaming response
            raise HTTPException(status_code=501, detail="Streaming not yet implemented")
        
        return {
            "response": result["content"],
            "model": result["model"],
            "usage": result["usage"],
            "finish_reason": result["finish_reason"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


@router.post("/summarize")
async def summarize_text(
    request: SummarizeRequest,
    current_user: dict = Depends(get_optional_user)
):
    """
    Summarize long text using LLM.
    
    Results are cached for 1 hour.
    """
    from app.api.utils.cache import cache_manager
    import hashlib
    
    # Check cache
    cache_key = f"summarize:{hashlib.md5(request.text.encode()).hexdigest()[:16]}"
    cached = await cache_manager.get(cache_key)
    if cached:
        return cached
    
    try:
        provider = LLMProvider(request.provider) if request.provider else None
        
        summary = await llm_chat_engine.summarize(
            text=request.text,
            max_length=request.max_length,
            provider=provider
        )
        
        result = {
            "original_length": len(request.text),
            "summary": summary,
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(request.text), 2)
        }
        
        # Cache result
        await cache_manager.set(cache_key, result, ttl=3600)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/translate")
async def translate_text(
    request: TranslateRequest,
    current_user: dict = Depends(get_optional_user)
):
    """
    Translate text to target language.
    
    Results are cached for 2 hours.
    """
    from app.api.utils.cache import cache_manager
    import hashlib
    
    # Check cache
    cache_key = f"translate:{request.target_language}:{hashlib.md5(request.text.encode()).hexdigest()[:16]}"
    cached = await cache_manager.get(cache_key)
    if cached:
        return cached
    
    try:
        provider = LLMProvider(request.provider) if request.provider else None
        
        translation = await llm_chat_engine.translate(
            text=request.text,
            target_language=request.target_language,
            provider=provider
        )
        
        result = {
            "original": request.text,
            "translation": translation,
            "target_language": request.target_language
        }
        
        # Cache result
        await cache_manager.set(cache_key, result, ttl=7200)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@router.post("/extract")
async def extract_structured_data(
    request: ExtractDataRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Extract structured data from text using function calling.
    
    Requires authentication.
    """
    try:
        provider = LLMProvider(request.provider) if request.provider else None
        
        extracted = await llm_chat_engine.extract_structured_data(
            text=request.text,
            schema=request.schema,
            provider=provider
        )
        
        return {
            "extracted_data": extracted,
            "schema": request.schema
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/models")
async def list_available_models(provider: str = "openai"):
    """List available models for a provider."""
    try:
        provider_enum = LLMProvider(provider)
        models = llm_chat_engine.get_available_models(provider_enum)
        
        return {
            "provider": provider,
            "models": models
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")


class EmbeddingsRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model: str = "text-embedding-3-small"


@router.post("/embeddings")
async def generate_embeddings(
    request: EmbeddingsRequest,
    current_user: dict = Depends(get_optional_user)
):
    """
    Generate embeddings using OpenAI.
    
    Maximum 100 texts per request.
    """
    try:
        embeddings = await llm_chat_engine.generate_embeddings(request.texts, model=request.model)
        
        return {
            "embeddings": embeddings,
            "model": request.model,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

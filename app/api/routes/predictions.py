from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import time

from app.db.connection import get_db
from app.db.crud import transaction_crud
from app.schemas.schemas import (
    SentimentRequest, SentimentResponse,
    NERRequest, NERResponse,
    ClassificationRequest, ClassificationResponse,
    SimilarityRequest, SimilarityResponse,
    EmbeddingRequest, EmbeddingResponse,
    RAGRequest, RAGResponse,
    TransactionResponse
)
from app.api.middleware.auth import get_current_user, get_optional_user
from app.ai.prediction_engine import prediction_engine
from app.models.database import AITransaction

router = APIRouter(prefix="/predictions", tags=["AI Predictions"])

async def log_transaction(
    db: AsyncSession,
    user_id: int,
    input_text: str,
    output_text: str,
    model_name: str,
    task_type: str,
    confidence: float,
    processing_time: float
):
    """Log prediction transaction to database."""
    transaction = AITransaction(
        user_id=user_id,
        input_text=input_text[:1000],
        output_text=output_text[:1000],
        model_name=model_name,
        task_type=task_type,
        confidence_score=confidence,
        processing_time=processing_time
    )
    db.add(transaction)
    await db.commit()

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_optional_user)
):
    """Analyze sentiment of text."""
    result = await prediction_engine.predict("sentiment", {"text": request.text})
    
    if current_user and current_user.get("user_id"):
        background_tasks.add_task(
            log_transaction, db, current_user["user_id"],
            request.text, result["sentiment"], "sentiment-model",
            "sentiment", result["confidence"], result["latency_ms"]
        )
    
    return SentimentResponse(
        text=request.text,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        scores=result["scores"]
    )

@router.post("/ner", response_model=NERResponse)
async def extract_entities(
    request: NERRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_optional_user)
):
    """Extract named entities from text."""
    result = await prediction_engine.predict("ner", {"text": request.text})
    
    return NERResponse(
        text=request.text,
        entities=result["entities"]
    )

@router.post("/classify", response_model=ClassificationResponse)
async def classify_text(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_optional_user)
):
    """Zero-shot text classification."""
    result = await prediction_engine.predict("classification", {
        "text": request.text,
        "labels": request.labels
    })
    
    return ClassificationResponse(
        text=request.text,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        all_scores=result["all_scores"]
    )

@router.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute semantic similarity between two texts."""
    result = await prediction_engine.predict("similarity", {
        "text1": request.text1,
        "text2": request.text2
    })
    
    return SimilarityResponse(
        text1=request.text1,
        text2=request.text2,
        similarity_score=result["similarity_score"]
    )

@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for texts."""
    result = await prediction_engine.predict("embedding", {"texts": request.texts})
    
    return EmbeddingResponse(
        embeddings=result["embeddings"],
        dimensions=result["dimensions"]
    )

@router.post("/rag", response_model=RAGResponse)
async def rag_query(
    request: RAGRequest,
    current_user: dict = Depends(get_current_user)
):
    """Query the RAG system."""
    result = await prediction_engine.predict("rag", {
        "query": request.query,
        "top_k": request.top_k
    })
    
    return RAGResponse(
        query=request.query,
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

@router.post("/batch")
async def batch_predictions(
    task_type: str,
    payloads: List[dict],
    current_user: dict = Depends(get_current_user)
):
    """Process batch predictions."""
    if len(payloads) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 items per batch")
    
    results = await prediction_engine.batch_predict(task_type, payloads)
    return {"results": results, "count": len(results)}

@router.get("/history", response_model=List[TransactionResponse])
async def get_prediction_history(
    skip: int = 0,
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's prediction history."""
    transactions = await transaction_crud.get_by_user(
        db, current_user["user_id"], skip=skip, limit=limit
    )
    return transactions

"""
Batch processing endpoints for large-scale AI operations.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from app.db.connection import get_db, get_redis
from app.api.middleware.auth import get_current_user
from app.tasks.ai_tasks import (
    batch_sentiment_analysis,
    batch_ner_extraction,
    generate_embeddings_async
)

router = APIRouter(prefix="/batch", tags=["Batch Processing"])


class BatchJob(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    items_count: int
    callback_url: Optional[str] = None


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    callback_url: Optional[str] = None


class BatchNERRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    callback_url: Optional[str] = None


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    callback_url: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    task_type: str = Field(..., description="sentiment, ner, classification, etc.")
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)
    callback_url: Optional[str] = None


@router.post("/sentiment", response_model=BatchJob)
async def batch_sentiment(
    request: BatchSentimentRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Process batch sentiment analysis.
    
    Maximum 1000 texts per batch. Results are processed asynchronously.
    """
    job_id = str(uuid.uuid4())
    
    # Store job metadata
    redis = await get_redis()
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "items_count": len(request.texts),
        "user_id": current_user.get("user_id"),
        "callback_url": request.callback_url
    }
    
    await redis.setex(
        f"batch_job:{job_id}",
        86400,  # 24 hours
        str(job_data)
    )
    
    # Queue task
    task = batch_sentiment_analysis.apply_async(
        args=[request.texts],
        task_id=job_id
    )
    
    return BatchJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow(),
        items_count=len(request.texts),
        callback_url=request.callback_url
    )


@router.post("/ner", response_model=BatchJob)
async def batch_ner(
    request: BatchNERRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Process batch named entity recognition.
    
    Maximum 1000 texts per batch.
    """
    job_id = str(uuid.uuid4())
    
    redis = await get_redis()
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "items_count": len(request.texts),
        "user_id": current_user.get("user_id"),
        "callback_url": request.callback_url
    }
    
    await redis.setex(f"batch_job:{job_id}", 86400, str(job_data))
    
    task = batch_ner_extraction.apply_async(
        args=[request.texts],
        task_id=job_id
    )
    
    return BatchJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow(),
        items_count=len(request.texts),
        callback_url=request.callback_url
    )


@router.post("/embeddings", response_model=BatchJob)
async def batch_embeddings(
    request: BatchEmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate embeddings for batch of texts.
    
    Maximum 1000 texts per batch.
    """
    job_id = str(uuid.uuid4())
    
    redis = await get_redis()
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "items_count": len(request.texts),
        "user_id": current_user.get("user_id"),
        "callback_url": request.callback_url
    }
    
    await redis.setex(f"batch_job:{job_id}", 86400, str(job_data))
    
    task = generate_embeddings_async.apply_async(
        args=[request.texts],
        task_id=job_id
    )
    
    return BatchJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow(),
        items_count=len(request.texts),
        callback_url=request.callback_url
    )


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get batch job status and results.
    
    Returns job metadata and results if completed.
    """
    from celery.result import AsyncResult
    
    # Get job from Celery
    task = AsyncResult(job_id)
    
    status_map = {
        "PENDING": "pending",
        "STARTED": "processing",
        "SUCCESS": "completed",
        "FAILURE": "failed",
        "RETRY": "retrying"
    }
    
    response = {
        "job_id": job_id,
        "status": status_map.get(task.state, "unknown"),
        "state": task.state
    }
    
    if task.state == "SUCCESS":
        response["results"] = task.result
        response["completed_at"] = datetime.utcnow().isoformat()
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
    elif task.state == "STARTED":
        response["progress"] = task.info.get("progress", 0) if isinstance(task.info, dict) else 0
    
    return response


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """
    List batch jobs for current user.
    
    Filter by status: pending, processing, completed, failed
    """
    redis = await get_redis()
    
    # Scan for user's jobs
    pattern = f"batch_job:*"
    jobs = []
    
    async for key in redis.scan_iter(match=pattern):
        job_data = await redis.get(key)
        if job_data:
            try:
                import ast
                job = ast.literal_eval(job_data)
                if job.get("user_id") == current_user.get("user_id"):
                    if status is None or job.get("status") == status:
                        jobs.append(job)
            except:
                pass
    
    # Sort by created_at descending
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return {
        "jobs": jobs[:limit],
        "count": len(jobs)
    }


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Cancel a pending or running batch job.
    """
    from celery.result import AsyncResult
    
    task = AsyncResult(job_id)
    
    if task.state in ["PENDING", "STARTED"]:
        task.revoke(terminate=True)
        
        redis = await get_redis()
        await redis.delete(f"batch_job:{job_id}")
        
        return {
            "message": "Job cancelled",
            "job_id": job_id
        }
    
    return {
        "message": f"Job cannot be cancelled (status: {task.state})",
        "job_id": job_id
    }


@router.post("/predictions", response_model=BatchJob)
async def batch_predictions(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Generic batch prediction endpoint.
    
    Supports any task type with custom payloads.
    """
    job_id = str(uuid.uuid4())
    
    redis = await get_redis()
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "task_type": request.task_type,
        "created_at": datetime.utcnow().isoformat(),
        "items_count": len(request.items),
        "user_id": current_user.get("user_id"),
        "callback_url": request.callback_url
    }
    
    await redis.setex(f"batch_job:{job_id}", 86400, str(job_data))
    
    # Store items temporarily
    await redis.setex(
        f"batch_items:{job_id}",
        3600,  # 1 hour
        str(request.items)
    )
    
    # Process in background
    from app.ai.prediction_engine import prediction_engine
    
    async def process_batch():
        try:
            results = await prediction_engine.batch_predict(
                request.task_type,
                request.items
            )
            
            # Store results
            await redis.setex(
                f"batch_results:{job_id}",
                86400,
                str(results)
            )
            
            # Update status
            job_data["status"] = "completed"
            await redis.setex(f"batch_job:{job_id}", 86400, str(job_data))
            
            # Call webhook if provided
            if request.callback_url:
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.post(
                        request.callback_url,
                        json={"job_id": job_id, "status": "completed", "results": results}
                    )
        except Exception as e:
            job_data["status"] = "failed"
            job_data["error"] = str(e)
            await redis.setex(f"batch_job:{job_id}", 86400, str(job_data))
    
    background_tasks.add_task(process_batch)
    
    return BatchJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow(),
        items_count=len(request.items),
        callback_url=request.callback_url
    )

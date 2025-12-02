from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import pandas as pd
import io

from app.db.connection import get_db
from app.db.crud import model_crud
from app.schemas.schemas import ModelCreate, ModelResponse, ModelMetrics
from app.api.middleware.auth import get_current_user, require_model_manager
from app.ai.ml_pipeline import ml_pipeline
from app.ai.rag_system import rag_system
from app.models.database import PredictionModel

router = APIRouter(prefix="/models", tags=["Model Management"])

@router.get("/", response_model=List[ModelResponse])
async def list_models(
    skip: int = 0,
    limit: int = 20,
    deployed_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """List all models."""
    if deployed_only:
        models = await model_crud.get_deployed(db)
    else:
        models = await model_crud.get_multi(db, skip=skip, limit=limit)
    return models

@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Get model details."""
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.post("/", response_model=ModelResponse)
async def create_model(
    model_data: ModelCreate,
    current_user: dict = Depends(require_model_manager),
    db: AsyncSession = Depends(get_db)
):
    """Register a new model."""
    existing = await model_crud.get_by_name(db, model_data.model_name)
    if existing:
        raise HTTPException(status_code=400, detail="Model name already exists")
    
    model_dict = {
        "model_name": model_data.model_name,
        "model_type": model_data.model_type,
        "description": model_data.description,
        "config": model_data.config,
        "version": "1.0.0"
    }
    
    model = await model_crud.create(db, model_dict)
    await db.commit()
    return model

@router.post("/{model_id}/train")
async def train_model(
    model_id: int,
    file: UploadFile = File(...),
    target_column: str = "target",
    test_size: float = 0.2,
    current_user: dict = Depends(require_model_manager),
    db: AsyncSession = Depends(get_db)
):
    """Train a model with uploaded data."""
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Read uploaded CSV
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
    
    # Preprocess and train
    X, y, features = ml_pipeline.preprocess_data(df, target_column)
    result = ml_pipeline.train_model(
        X, y,
        model_type=model.model_type,
        model_name=model.model_name,
        test_size=test_size
    )
    
    # Update model metrics
    await model_crud.update(db, model_id, {
        "accuracy": result["metrics"]["accuracy"],
        "precision_score": result["metrics"]["precision"],
        "recall_score": result["metrics"]["recall"],
        "f1_score": result["metrics"]["f1_score"]
    })
    await db.commit()
    
    return {
        "message": "Model trained successfully",
        "metrics": result["metrics"],
        "run_id": result["run_id"]
    }

@router.post("/{model_id}/deploy")
async def deploy_model(
    model_id: int,
    current_user: dict = Depends(require_model_manager),
    db: AsyncSession = Depends(get_db)
):
    """Deploy a model for inference."""
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.model_name not in ml_pipeline.list_models():
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    await model_crud.update(db, model_id, {"deployed": True})
    await db.commit()
    
    return {"message": f"Model {model.model_name} deployed successfully"}

@router.delete("/{model_id}/deploy")
async def undeploy_model(
    model_id: int,
    current_user: dict = Depends(require_model_manager),
    db: AsyncSession = Depends(get_db)
):
    """Undeploy a model."""
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    await model_crud.update(db, model_id, {"deployed": False})
    await db.commit()
    
    return {"message": f"Model {model.model_name} undeployed"}

@router.get("/{model_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_id: int, db: AsyncSession = Depends(get_db)):
    """Get model performance metrics."""
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelMetrics(
        accuracy=model.accuracy or 0,
        precision=model.precision_score or 0,
        recall=model.recall_score or 0,
        f1_score=model.f1_score or 0
    )

@router.post("/rag/documents")
async def add_rag_documents(
    documents: List[dict],
    current_user: dict = Depends(require_model_manager)
):
    """Add documents to RAG knowledge base."""
    rag_system.add_documents(documents)
    return {
        "message": f"Added {len(documents)} documents",
        "stats": rag_system.get_stats()
    }

@router.delete("/rag/documents")
async def clear_rag_documents(current_user: dict = Depends(require_model_manager)):
    """Clear RAG knowledge base."""
    rag_system.clear()
    return {"message": "Knowledge base cleared"}

@router.get("/loaded")
async def get_loaded_models():
    """Get list of models loaded in memory."""
    return {
        "ml_models": ml_pipeline.list_models(),
        "rag_stats": rag_system.get_stats()
    }

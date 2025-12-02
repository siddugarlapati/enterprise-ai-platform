from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime, timedelta

from app.db.connection import get_db
from app.db.crud import user_crud, model_crud, transaction_crud
from app.schemas.schemas import UserResponse, ModelResponse, PaginatedResponse
from app.api.middleware.auth import require_admin, get_current_user
from app.ai.prediction_engine import prediction_engine
from app.models.database import User, AITransaction, PredictionModel

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 20,
    active_only: bool = False,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all users (admin only)."""
    if active_only:
        users = await user_crud.get_active_users(db, skip=skip, limit=limit)
    else:
        users = await user_crud.get_multi(db, skip=skip, limit=limit)
    return users

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get user details (admin only)."""
    user = await user_crud.get(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update user role (admin only)."""
    valid_roles = ["admin", "user", "analyst", "model_manager"]
    if role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
    
    user = await user_crud.get(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await user_crud.update(db, user_id, {"role": role})
    await db.commit()
    
    return {"message": f"User role updated to {role}"}

@router.put("/users/{user_id}/status")
async def toggle_user_status(
    user_id: int,
    is_active: bool,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Enable/disable user (admin only)."""
    user = await user_crud.get(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await user_crud.update(db, user_id, {"is_active": is_active})
    await db.commit()
    
    status = "enabled" if is_active else "disabled"
    return {"message": f"User {status}"}

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete user (admin only)."""
    if user_id == current_user["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    deleted = await user_crud.delete(db, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    
    await db.commit()
    return {"message": "User deleted"}

@router.get("/stats")
async def get_system_stats(
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get system statistics (admin only)."""
    # User stats
    total_users = await user_crud.count(db)
    active_users = await db.execute(select(func.count()).select_from(User).where(User.is_active == True))
    
    # Transaction stats
    total_transactions = await transaction_crud.count(db)
    
    # Model stats
    total_models = await model_crud.count(db)
    deployed_models = len(await model_crud.get_deployed(db))
    
    # Recent activity (last 24h)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_transactions = await db.execute(
        select(func.count()).select_from(AITransaction).where(AITransaction.created_at >= yesterday)
    )
    
    # Prediction engine metrics
    engine_metrics = prediction_engine.get_metrics()
    
    return {
        "users": {
            "total": total_users,
            "active": active_users.scalar()
        },
        "transactions": {
            "total": total_transactions,
            "last_24h": recent_transactions.scalar()
        },
        "models": {
            "total": total_models,
            "deployed": deployed_models
        },
        "engine": engine_metrics
    }

@router.get("/transactions")
async def list_all_transactions(
    skip: int = 0,
    limit: int = 50,
    user_id: Optional[int] = None,
    task_type: Optional[str] = None,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all transactions (admin only)."""
    query = select(AITransaction)
    
    if user_id:
        query = query.where(AITransaction.user_id == user_id)
    if task_type:
        query = query.where(AITransaction.task_type == task_type)
    
    query = query.order_by(AITransaction.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    transactions = result.scalars().all()
    
    return transactions

@router.post("/reset-metrics")
async def reset_engine_metrics(current_user: dict = Depends(require_admin)):
    """Reset prediction engine metrics (admin only)."""
    prediction_engine.reset_metrics()
    return {"message": "Metrics reset successfully"}

@router.get("/export/transactions")
async def export_transactions(
    format: str = Query("json", enum=["json", "csv"]),
    days: int = 7,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Export transactions (admin only)."""
    since = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(AITransaction).where(AITransaction.created_at >= since)
    )
    transactions = result.scalars().all()
    
    if format == "csv":
        import csv
        import io
        from fastapi.responses import StreamingResponse
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "user_id", "task_type", "model_name", "confidence", "processing_time", "created_at"])
        
        for t in transactions:
            writer.writerow([t.id, t.user_id, t.task_type, t.model_name, t.confidence_score, t.processing_time, t.created_at])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=transactions_{days}d.csv"}
        )
    
    return [
        {
            "id": t.id,
            "user_id": t.user_id,
            "task_type": t.task_type,
            "model_name": t.model_name,
            "confidence": t.confidence_score,
            "processing_time": t.processing_time,
            "created_at": t.created_at.isoformat()
        }
        for t in transactions
    ]

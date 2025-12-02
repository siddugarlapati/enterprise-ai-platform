from celery import shared_task
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)


@shared_task
def cleanup_old_transactions(days: int = 90) -> Dict[str, Any]:
    """Clean up old transaction records."""
    from app.db.connection import get_sync_db
    from app.models.database import AITransaction
    from sqlalchemy import delete
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_sync_db() as db:
        result = db.execute(
            delete(AITransaction).where(AITransaction.created_at < cutoff_date)
        )
        deleted_count = result.rowcount
        db.commit()
    
    logger.info(f"Cleaned up {deleted_count} old transactions")
    return {"deleted_count": deleted_count, "cutoff_date": cutoff_date.isoformat()}


@shared_task
def update_model_metrics() -> Dict[str, Any]:
    """Update model performance metrics."""
    from app.ai.prediction_engine import prediction_engine
    
    metrics = prediction_engine.get_metrics()
    logger.info(f"Updated model metrics: {metrics['total_requests']} total requests")
    
    return metrics


@shared_task
def system_health_check() -> Dict[str, Any]:
    """Perform system health check."""
    from sqlalchemy import text
    
    health = {"timestamp": datetime.utcnow().isoformat(), "checks": {}}
    
    # Check database
    try:
        from app.db.connection import sync_engine
        with sync_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health["checks"]["database"] = "healthy"
    except Exception as e:
        health["checks"]["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        import redis
        from app.config.settings import settings
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        health["checks"]["redis"] = "healthy"
    except Exception as e:
        health["checks"]["redis"] = f"unhealthy: {str(e)}"
    
    # Check ML models
    try:
        from app.ai.ml_pipeline import ml_pipeline
        health["checks"]["ml_models"] = f"loaded: {len(ml_pipeline.list_models())}"
    except Exception as e:
        health["checks"]["ml_models"] = f"error: {str(e)}"
    
    all_healthy = all("healthy" in str(v) or "loaded" in str(v) for v in health["checks"].values())
    health["status"] = "healthy" if all_healthy else "degraded"
    
    logger.info(f"Health check: {health['status']}")
    return health


@shared_task
def backup_model_cache() -> Dict[str, Any]:
    """Backup model cache to storage."""
    import os
    import shutil
    from app.config.settings import settings
    
    backup_dir = f"{settings.MODEL_CACHE_DIR}_backup_{datetime.utcnow().strftime('%Y%m%d')}"
    
    if os.path.exists(settings.MODEL_CACHE_DIR):
        shutil.copytree(settings.MODEL_CACHE_DIR, backup_dir, dirs_exist_ok=True)
        logger.info(f"Model cache backed up to {backup_dir}")
        return {"backup_path": backup_dir, "status": "success"}
    
    return {"status": "skipped", "reason": "No model cache found"}


@shared_task
def generate_usage_report(days: int = 7) -> Dict[str, Any]:
    """Generate usage report."""
    from app.db.connection import get_sync_db
    from app.models.database import AITransaction, User
    from sqlalchemy import func, select
    
    since = datetime.utcnow() - timedelta(days=days)
    
    with get_sync_db() as db:
        # Transaction stats
        total_transactions = db.execute(
            select(func.count()).select_from(AITransaction).where(AITransaction.created_at >= since)
        ).scalar()
        
        # By task type
        task_stats = db.execute(
            select(AITransaction.task_type, func.count())
            .where(AITransaction.created_at >= since)
            .group_by(AITransaction.task_type)
        ).all()
        
        # Active users
        active_users = db.execute(
            select(func.count(func.distinct(AITransaction.user_id)))
            .where(AITransaction.created_at >= since)
        ).scalar()
    
    report = {
        "period_days": days,
        "total_transactions": total_transactions,
        "active_users": active_users,
        "by_task_type": dict(task_stats),
        "generated_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Generated usage report: {total_transactions} transactions in {days} days")
    return report


@shared_task
def cleanup_expired_api_keys() -> Dict[str, Any]:
    """Clean up expired API keys."""
    from app.db.connection import get_sync_db
    from app.models.database import APIKey
    from sqlalchemy import delete
    
    now = datetime.utcnow()
    
    with get_sync_db() as db:
        result = db.execute(
            delete(APIKey).where(APIKey.expires_at < now)
        )
        deleted_count = result.rowcount
        db.commit()
    
    logger.info(f"Cleaned up {deleted_count} expired API keys")
    return {"deleted_count": deleted_count}


@shared_task
def aggregate_daily_stats() -> Dict[str, Any]:
    """Aggregate daily statistics."""
    from app.db.connection import get_sync_db
    from app.models.database import AITransaction
    from sqlalchemy import func, select
    
    yesterday = datetime.utcnow().date() - timedelta(days=1)
    start = datetime.combine(yesterday, datetime.min.time())
    end = datetime.combine(yesterday, datetime.max.time())
    
    with get_sync_db() as db:
        stats = db.execute(
            select(
                func.count().label('total'),
                func.avg(AITransaction.processing_time).label('avg_processing_time'),
                func.avg(AITransaction.confidence_score).label('avg_confidence')
            )
            .where(AITransaction.created_at.between(start, end))
        ).first()
    
    return {
        "date": yesterday.isoformat(),
        "total_transactions": stats.total if stats else 0,
        "avg_processing_time": float(stats.avg_processing_time) if stats and stats.avg_processing_time else 0,
        "avg_confidence": float(stats.avg_confidence) if stats and stats.avg_confidence else 0
    }

from celery import Celery
from celery.schedules import crontab
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "ai_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.ai_tasks", "app.tasks.maintenance_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3000,
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    result_expires=86400,
)

# Scheduled tasks
celery_app.conf.beat_schedule = {
    "cleanup-old-transactions": {
        "task": "app.tasks.maintenance_tasks.cleanup_old_transactions",
        "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    "update-model-metrics": {
        "task": "app.tasks.maintenance_tasks.update_model_metrics",
        "schedule": crontab(minute="*/30"),  # Every 30 minutes
    },
    "health-check": {
        "task": "app.tasks.maintenance_tasks.system_health_check",
        "schedule": crontab(minute="*/5"),  # Every 5 minutes
    },
}

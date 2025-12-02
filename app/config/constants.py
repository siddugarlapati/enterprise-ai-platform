# Role definitions
class Roles:
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    MODEL_MANAGER = "model_manager"

ROLE_PERMISSIONS = {
    Roles.ADMIN: ["read", "write", "delete", "manage_users", "manage_models", "view_analytics"],
    Roles.MODEL_MANAGER: ["read", "write", "manage_models", "view_analytics"],
    Roles.ANALYST: ["read", "view_analytics"],
    Roles.USER: ["read", "write"],
}

# Model types
class ModelTypes:
    SENTIMENT = "sentiment"
    NER = "ner"
    CLASSIFICATION = "classification"
    SIMILARITY = "similarity"
    EMBEDDING = "embedding"

# Task statuses
class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Cache keys
CACHE_KEYS = {
    "user": "user:{user_id}",
    "model": "model:{model_id}",
    "prediction": "prediction:{prediction_id}",
    "rate_limit": "rate_limit:{client_id}",
}

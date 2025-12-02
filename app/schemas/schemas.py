from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
class TaskType(str, Enum):
    SENTIMENT = "sentiment"
    NER = "ner"
    CLASSIFICATION = "classification"
    SIMILARITY = "similarity"
    EMBEDDING = "embedding"
    RAG = "rag"

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    MODEL_MANAGER = "model_manager"

# Auth Schemas
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: int
    exp: datetime
    role: str

# AI Request/Response Schemas
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

class NEREntity(BaseModel):
    entity: str
    label: str
    start: int
    end: int
    confidence: float

class NERResponse(BaseModel):
    text: str
    entities: List[NEREntity]

class ClassificationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    labels: List[str] = Field(..., min_items=2)

class ClassificationResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    all_scores: Dict[str, float]

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    similarity_score: float
    text1: str
    text2: str

class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int

class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    context_window: int = Field(default=512, ge=128, le=2048)

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

# Model Schemas
class ModelCreate(BaseModel):
    model_name: str
    model_type: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class ModelResponse(BaseModel):
    id: int
    model_name: str
    model_type: str
    description: Optional[str]
    accuracy: Optional[float]
    version: str
    deployed: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[List[List[int]]] = None

# Transaction Schemas
class TransactionResponse(BaseModel):
    id: int
    input_text: str
    output_text: str
    model_name: str
    task_type: str
    confidence_score: Optional[float]
    processing_time: float
    created_at: datetime
    
    class Config:
        from_attributes = True

# Pagination
class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int

# Health Check
class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    redis: str
    models_loaded: int
    uptime_seconds: float

# API Key
class APIKeyCreate(BaseModel):
    name: str
    scopes: List[str] = ["read"]
    expires_days: Optional[int] = 30

class APIKeyResponse(BaseModel):
    id: int
    name: str
    key: str  # Only returned on creation
    scopes: List[str]
    expires_at: Optional[datetime]
    created_at: datetime

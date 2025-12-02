from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AITransaction(Base):
    __tablename__ = "ai_transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    input_text = Column(String)
    output_text = Column(String)
    model_name = Column(String)
    confidence_score = Column(Float)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionModel(Base):
    __tablename__ = "prediction_models"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True)
    model_type = Column(String)
    accuracy = Column(Float)
    version = Column(String)
    deployed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
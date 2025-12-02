from celery import shared_task
import logging
from typing import List, Dict, Any

from app.ai.prediction_engine import prediction_engine
from app.ai.ml_pipeline import ml_pipeline
from app.ai.rag_system import rag_system

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def batch_sentiment_analysis(self, texts: List[str]) -> List[Dict[str, Any]]:
    """Process batch sentiment analysis."""
    try:
        results = []
        for text in texts:
            result = prediction_engine._handle_sentiment({"text": text})
            results.append(result)
        return results
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        self.retry(exc=e, countdown=60)

@shared_task(bind=True, max_retries=3)
def batch_ner_extraction(self, texts: List[str]) -> List[Dict[str, Any]]:
    """Process batch NER extraction."""
    try:
        results = []
        for text in texts:
            result = prediction_engine._handle_ner({"text": text})
            results.append(result)
        return results
    except Exception as e:
        logger.error(f"Batch NER extraction failed: {e}")
        self.retry(exc=e, countdown=60)

@shared_task(bind=True, max_retries=3)
def train_model_async(
    self,
    model_name: str,
    model_type: str,
    data: List[Dict[str, Any]],
    target_column: str
) -> Dict[str, Any]:
    """Train model asynchronously."""
    try:
        import pandas as pd
        df = pd.DataFrame(data)
        
        X, y, features = ml_pipeline.preprocess_data(df, target_column)
        result = ml_pipeline.train_model(X, y, model_type=model_type, model_name=model_name)
        
        logger.info(f"Model {model_name} trained successfully")
        return result
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        self.retry(exc=e, countdown=120)

@shared_task(bind=True)
def index_documents_async(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Index documents for RAG system asynchronously."""
    try:
        rag_system.add_documents(documents)
        stats = rag_system.get_stats()
        logger.info(f"Indexed {len(documents)} documents")
        return stats
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        raise

@shared_task(bind=True, max_retries=3)
def generate_embeddings_async(self, texts: List[str]) -> Dict[str, Any]:
    """Generate embeddings asynchronously."""
    try:
        result = prediction_engine._handle_embedding({"texts": texts})
        return result
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        self.retry(exc=e, countdown=60)

@shared_task
def process_large_dataset(
    dataset_id: str,
    task_type: str,
    batch_size: int = 100
) -> Dict[str, Any]:
    """Process large dataset in batches."""
    # This would typically load data from storage
    logger.info(f"Processing dataset {dataset_id} with task {task_type}")
    
    return {
        "dataset_id": dataset_id,
        "task_type": task_type,
        "status": "completed",
        "processed_batches": 0
    }

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.ai.llm_engine import llm_engine
from app.ai.ml_pipeline import ml_pipeline
from app.ai.rag_system import rag_system
from app.config.settings import settings

logger = logging.getLogger(__name__)

class PredictionEngine:
    """Unified prediction engine for all AI tasks."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._request_count = 0
        self._total_latency = 0.0
        self._start_time = datetime.utcnow()
    
    async def predict(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Route prediction request to appropriate handler."""
        start_time = time.time()
        
        handlers = {
            "sentiment": self._handle_sentiment,
            "ner": self._handle_ner,
            "classification": self._handle_classification,
            "similarity": self._handle_similarity,
            "embedding": self._handle_embedding,
            "rag": self._handle_rag,
            "ml_predict": self._handle_ml_predict,
        }
        
        handler = handlers.get(task_type)
        if not handler:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Run in thread pool for CPU-bound tasks
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, handler, payload)
        
        latency = time.time() - start_time
        self._request_count += 1
        self._total_latency += latency
        
        result["latency_ms"] = round(latency * 1000, 2)
        result["task_type"] = task_type
        result["timestamp"] = datetime.utcnow().isoformat()
        
        return result

    def _handle_sentiment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        result = llm_engine.analyze_sentiment(text)
        return {
            "text": text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "scores": result["scores"]
        }

    def _handle_ner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        entities = llm_engine.extract_entities(text)
        return {
            "text": text,
            "entities": entities,
            "entity_count": len(entities)
        }

    def _handle_classification(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        labels = payload.get("labels", [])
        result = llm_engine.classify_text(text, labels)
        return {
            "text": text,
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"]
        }

    def _handle_similarity(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text1 = payload.get("text1", "")
        text2 = payload.get("text2", "")
        score = llm_engine.compute_similarity(text1, text2)
        return {
            "text1": text1,
            "text2": text2,
            "similarity_score": score
        }

    def _handle_embedding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        texts = payload.get("texts", [])
        embeddings = llm_engine.get_embeddings(texts)
        return {
            "embeddings": embeddings.tolist(),
            "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
            "count": len(texts)
        }

    def _handle_rag(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        top_k = payload.get("top_k", 5)
        result = rag_system.query(query, top_k)
        return result

    def _handle_ml_predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        model_name = payload.get("model_name", "default")
        features = payload.get("features", [])
        
        X = np.array(features).reshape(1, -1) if isinstance(features[0], (int, float)) else np.array(features)
        predictions, probabilities = ml_pipeline.predict(model_name, X)
        
        return {
            "model_name": model_name,
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None
        }

    async def batch_predict(self, task_type: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple predictions in batch."""
        tasks = [self.predict(task_type, payload) for payload in payloads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "index": i,
                    "task_type": task_type
                })
            else:
                result["index"] = i
                processed_results.append(result)
        
        return processed_results

    def get_metrics(self) -> Dict[str, Any]:
        """Get prediction engine metrics."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        avg_latency = self._total_latency / self._request_count if self._request_count > 0 else 0
        
        return {
            "total_requests": self._request_count,
            "average_latency_ms": round(avg_latency * 1000, 2),
            "uptime_seconds": round(uptime, 2),
            "requests_per_second": round(self._request_count / uptime, 2) if uptime > 0 else 0,
            "models_loaded": len(ml_pipeline.list_models()),
            "llm_info": llm_engine.get_model_info(),
            "rag_stats": rag_system.get_stats()
        }

    def reset_metrics(self):
        """Reset metrics counters."""
        self._request_count = 0
        self._total_latency = 0.0
        self._start_time = datetime.utcnow()

# Singleton instance
prediction_engine = PredictionEngine()

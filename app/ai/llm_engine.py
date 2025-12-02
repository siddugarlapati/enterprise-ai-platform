import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

class LLMEngine:
    """Core NLP engine with sentiment, NER, classification, and embeddings."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sentiment_pipeline = None
        self._ner_pipeline = None
        self._zero_shot_pipeline = None
        self._embedding_model = None
        self._tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self._document_embeddings = None
        self._documents = []
        
    @property
    def sentiment_pipeline(self):
        if self._sentiment_pipeline is None:
            logger.info("Loading sentiment analysis model...")
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1
            )
        return self._sentiment_pipeline
    
    @property
    def ner_pipeline(self):
        if self._ner_pipeline is None:
            logger.info("Loading NER model...")
            self._ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
        return self._ner_pipeline
    
    @property
    def zero_shot_pipeline(self):
        if self._zero_shot_pipeline is None:
            logger.info("Loading zero-shot classification model...")
            self._zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
        return self._zero_shot_pipeline
    
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            logger.info("Loading embedding model...")
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            if self.device == "cuda":
                self._embedding_model = self._embedding_model.to(self.device)
        return self._embedding_model

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        result = self.sentiment_pipeline(text[:512])[0]
        
        # Get detailed scores
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        if result["label"] == "POSITIVE":
            scores["positive"] = result["score"]
            scores["negative"] = 1 - result["score"]
        else:
            scores["negative"] = result["score"]
            scores["positive"] = 1 - result["score"]
        scores["neutral"] = 1 - abs(scores["positive"] - scores["negative"])
        
        return {
            "sentiment": result["label"].lower(),
            "confidence": result["score"],
            "scores": scores
        }

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        results = self.ner_pipeline(text[:512])
        
        entities = []
        for entity in results:
            entities.append({
                "entity": entity["word"],
                "label": entity["entity_group"],
                "start": entity["start"],
                "end": entity["end"],
                "confidence": float(entity["score"])
            })
        return entities

    def classify_text(self, text: str, labels: List[str]) -> Dict[str, Any]:
        """Zero-shot text classification."""
        result = self.zero_shot_pipeline(text[:512], labels, multi_label=False)
        
        all_scores = dict(zip(result["labels"], result["scores"]))
        
        return {
            "predicted_label": result["labels"][0],
            "confidence": result["scores"][0],
            "all_scores": all_scores
        }

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        embeddings = self.get_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Perform semantic search over documents."""
        if not documents:
            return []
        
        query_embedding = self.get_embeddings([query])[0]
        doc_embeddings = self.get_embeddings(documents)
        
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx]), documents[idx]))
        return results

    def tfidf_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """TF-IDF based search."""
        if not documents:
            return []
        
        tfidf_matrix = self._tfidf_vectorizer.fit_transform(documents + [query])
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx]), documents[idx]))
        return results

    def index_documents(self, documents: List[str]):
        """Index documents for search."""
        self._documents = documents
        self._document_embeddings = self.get_embeddings(documents)
        logger.info(f"Indexed {len(documents)} documents")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "device": self.device,
            "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
            "ner_model": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "classification_model": "facebook/bart-large-mnli",
            "embedding_model": settings.EMBEDDING_MODEL,
            "indexed_documents": len(self._documents)
        }

# Singleton instance
llm_engine = LLMEngine()

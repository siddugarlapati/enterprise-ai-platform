import pytest
import numpy as np

from app.ai.llm_engine import LLMEngine

@pytest.fixture
def llm_engine():
    return LLMEngine()

class TestSentimentAnalysis:
    def test_positive_sentiment(self, llm_engine):
        result = llm_engine.analyze_sentiment("I love this product! It's amazing and wonderful.")
        assert result["sentiment"] in ["positive", "negative"]
        assert 0 <= result["confidence"] <= 1
        assert "scores" in result

    def test_negative_sentiment(self, llm_engine):
        result = llm_engine.analyze_sentiment("This is terrible. I hate it so much.")
        assert result["sentiment"] in ["positive", "negative"]
        assert 0 <= result["confidence"] <= 1

    def test_empty_text(self, llm_engine):
        result = llm_engine.analyze_sentiment("")
        assert "sentiment" in result

class TestNER:
    def test_extract_entities(self, llm_engine):
        text = "Apple Inc. was founded by Steve Jobs in California."
        entities = llm_engine.extract_entities(text)
        assert isinstance(entities, list)
        
        # Check entity structure
        if entities:
            entity = entities[0]
            assert "entity" in entity
            assert "label" in entity
            assert "confidence" in entity

    def test_no_entities(self, llm_engine):
        text = "The quick brown fox jumps over the lazy dog."
        entities = llm_engine.extract_entities(text)
        assert isinstance(entities, list)

class TestClassification:
    def test_zero_shot_classification(self, llm_engine):
        text = "The stock market crashed today"
        labels = ["finance", "sports", "technology"]
        
        result = llm_engine.classify_text(text, labels)
        
        assert result["predicted_label"] in labels
        assert 0 <= result["confidence"] <= 1
        assert len(result["all_scores"]) == len(labels)

    def test_classification_with_many_labels(self, llm_engine):
        text = "Scientists discovered a new planet"
        labels = ["science", "politics", "entertainment", "sports", "technology"]
        
        result = llm_engine.classify_text(text, labels)
        assert result["predicted_label"] in labels

class TestEmbeddings:
    def test_single_embedding(self, llm_engine):
        texts = ["Hello world"]
        embeddings = llm_engine.get_embeddings(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0

    def test_multiple_embeddings(self, llm_engine):
        texts = ["Hello world", "Goodbye world", "Test text"]
        embeddings = llm_engine.get_embeddings(texts)
        
        assert embeddings.shape[0] == 3

    def test_embedding_consistency(self, llm_engine):
        text = "Consistent text"
        emb1 = llm_engine.get_embeddings([text])[0]
        emb2 = llm_engine.get_embeddings([text])[0]
        
        # Embeddings should be identical for same text
        np.testing.assert_array_almost_equal(emb1, emb2)

class TestSimilarity:
    def test_identical_texts(self, llm_engine):
        text = "The quick brown fox"
        similarity = llm_engine.compute_similarity(text, text)
        
        assert similarity > 0.99  # Should be very close to 1

    def test_similar_texts(self, llm_engine):
        text1 = "The cat sat on the mat"
        text2 = "A feline rested on the rug"
        
        similarity = llm_engine.compute_similarity(text1, text2)
        assert 0 < similarity < 1

    def test_different_texts(self, llm_engine):
        text1 = "Machine learning is fascinating"
        text2 = "I love pizza and pasta"
        
        similarity = llm_engine.compute_similarity(text1, text2)
        assert similarity < 0.5  # Should be relatively low

class TestSemanticSearch:
    def test_semantic_search(self, llm_engine):
        documents = [
            "Python is a programming language",
            "Machine learning uses algorithms",
            "Cats are domestic animals",
            "Neural networks are inspired by the brain"
        ]
        
        query = "artificial intelligence"
        results = llm_engine.semantic_search(query, documents, top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)

    def test_tfidf_search(self, llm_engine):
        documents = [
            "Python programming tutorial",
            "Java development guide",
            "Python machine learning basics"
        ]
        
        query = "Python"
        results = llm_engine.tfidf_search(query, documents, top_k=2)
        
        assert len(results) == 2

class TestModelInfo:
    def test_get_model_info(self, llm_engine):
        info = llm_engine.get_model_info()
        
        assert "device" in info
        assert "sentiment_model" in info
        assert "embedding_model" in info

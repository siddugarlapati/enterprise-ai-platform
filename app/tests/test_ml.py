import pytest
import numpy as np
import pandas as pd

from app.ai.ml_pipeline import MLPipeline
from app.ai.rag_system import RAGSystem

@pytest.fixture
def ml_pipeline():
    return MLPipeline()

@pytest.fixture
def rag_system():
    return RAGSystem()

@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })

class TestMLPipeline:
    def test_preprocess_data(self, ml_pipeline, sample_dataframe):
        X, y, features = ml_pipeline.preprocess_data(
            sample_dataframe, 
            target_column='target'
        )
        
        assert X.shape[0] == 100
        assert len(y) == 100
        assert len(features) == 3

    def test_train_model(self, ml_pipeline, sample_dataframe):
        X, y, _ = ml_pipeline.preprocess_data(sample_dataframe, 'target')
        
        result = ml_pipeline.train_model(
            X, y,
            model_type='random_forest',
            model_name='test_model',
            hyperparams={'n_estimators': 10, 'max_depth': 3}
        )
        
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert 0 <= result['metrics']['accuracy'] <= 1

    def test_predict(self, ml_pipeline, sample_dataframe):
        X, y, _ = ml_pipeline.preprocess_data(sample_dataframe, 'target')
        
        ml_pipeline.train_model(X, y, model_name='predict_test')
        
        predictions, probabilities = ml_pipeline.predict('predict_test', X[:5])
        
        assert len(predictions) == 5
        assert probabilities is not None
        assert probabilities.shape[0] == 5

    def test_evaluate_model(self, ml_pipeline):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = ml_pipeline.evaluate_model(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_batch_predict(self, ml_pipeline, sample_dataframe):
        X, y, features = ml_pipeline.preprocess_data(sample_dataframe, 'target')
        ml_pipeline.train_model(X, y, model_name='batch_test')
        
        results = ml_pipeline.batch_predict(
            'batch_test',
            sample_dataframe,
            feature_columns=['feature1', 'feature2', 'feature3'],
            batch_size=20
        )
        
        assert len(results) == 100
        assert all('prediction' in r for r in results)

    def test_list_models(self, ml_pipeline, sample_dataframe):
        X, y, _ = ml_pipeline.preprocess_data(sample_dataframe, 'target')
        ml_pipeline.train_model(X, y, model_name='list_test')
        
        models = ml_pipeline.list_models()
        assert 'list_test' in models

    def test_different_model_types(self, ml_pipeline, sample_dataframe):
        X, y, _ = ml_pipeline.preprocess_data(sample_dataframe, 'target')
        
        for model_type in ['random_forest', 'logistic_regression']:
            result = ml_pipeline.train_model(
                X, y,
                model_type=model_type,
                model_name=f'test_{model_type}'
            )
            assert result['metrics']['accuracy'] > 0

class TestRAGSystem:
    def test_add_documents(self, rag_system):
        documents = [
            {"content": "Python is a programming language", "metadata": {"source": "doc1"}},
            {"content": "Machine learning is a subset of AI", "metadata": {"source": "doc2"}}
        ]
        
        rag_system.add_documents(documents)
        stats = rag_system.get_stats()
        
        assert stats["total_documents"] > 0

    def test_retrieve(self, rag_system):
        documents = [
            {"content": "Python is great for data science"},
            {"content": "JavaScript is used for web development"},
            {"content": "Machine learning requires lots of data"}
        ]
        
        rag_system.add_documents(documents)
        results = rag_system.retrieve("data science programming", top_k=2)
        
        assert len(results) == 2
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)

    def test_query(self, rag_system):
        documents = [
            {"content": "The capital of France is Paris. Paris is known for the Eiffel Tower."},
            {"content": "London is the capital of the United Kingdom."}
        ]
        
        rag_system.add_documents(documents)
        result = rag_system.query("What is the capital of France?")
        
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result

    def test_clear(self, rag_system):
        documents = [{"content": "Test document"}]
        rag_system.add_documents(documents)
        
        rag_system.clear()
        stats = rag_system.get_stats()
        
        assert stats["total_documents"] == 0

    def test_export_import_index(self, rag_system):
        documents = [{"content": "Test document for export"}]
        rag_system.add_documents(documents)
        
        exported = rag_system.export_index()
        
        rag_system.clear()
        rag_system.import_index(exported)
        
        stats = rag_system.get_stats()
        assert stats["total_documents"] > 0

    def test_chunk_text(self, rag_system):
        long_text = " ".join(["word"] * 1000)
        chunks = rag_system._chunk_text(long_text)
        
        assert len(chunks) > 1
        assert all("content" in c for c in chunks)
        assert all("id" in c for c in chunks)

class TestPredictionEngine:
    @pytest.mark.asyncio
    async def test_predict_sentiment(self):
        from app.ai.prediction_engine import prediction_engine
        
        result = await prediction_engine.predict("sentiment", {"text": "I love this!"})
        
        assert "sentiment" in result
        assert "latency_ms" in result
        assert "task_type" in result

    @pytest.mark.asyncio
    async def test_batch_predict(self):
        from app.ai.prediction_engine import prediction_engine
        
        payloads = [
            {"text": "Great product!"},
            {"text": "Terrible experience"}
        ]
        
        results = await prediction_engine.batch_predict("sentiment", payloads)
        
        assert len(results) == 2

    def test_get_metrics(self):
        from app.ai.prediction_engine import prediction_engine
        
        metrics = prediction_engine.get_metrics()
        
        assert "total_requests" in metrics
        assert "average_latency_ms" in metrics
        assert "uptime_seconds" in metrics

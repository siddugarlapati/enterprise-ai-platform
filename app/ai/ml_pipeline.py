import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import logging
import os

from app.config.settings import settings

logger = logging.getLogger(__name__)

class MLPipeline:
    """ML Pipeline for training, evaluation, and inference."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess data for training."""
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle categorical features
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        
        # Scale features
        if scale:
            scaler_key = "_".join(feature_columns[:3])
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                X = self.scalers[scaler_key].fit_transform(X)
            else:
                X = self.scalers[scaler_key].transform(X)
        
        # Encode target if categorical
        if y.dtype == 'object':
            if target_column not in self.encoders:
                self.encoders[target_column] = LabelEncoder()
                y = self.encoders[target_column].fit_transform(y)
            else:
                y = self.encoders[target_column].transform(y)
        
        return np.array(X), np.array(y), feature_columns

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "random_forest",
        model_name: str = "default",
        hyperparams: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train a model with MLflow tracking."""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Select model
        model_classes = {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "logistic_regression": LogisticRegression
        }
        
        model_class = model_classes.get(model_type, RandomForestClassifier)
        params = hyperparams or {}
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("test_size", test_size)
            
            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            metrics = self.evaluate_model(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Store model
            self.models[model_name] = model
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            metrics["cv_mean"] = float(cv_scores.mean())
            metrics["cv_std"] = float(cv_scores.std())
            
            logger.info(f"Model {model_name} trained with accuracy: {metrics['accuracy']:.4f}")
            
            return {
                "model_name": model_name,
                "model_type": model_type,
                "metrics": metrics,
                "run_id": mlflow.active_run().info.run_id
            }

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

    def predict(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
        
        return predictions, probabilities

    def batch_predict(
        self, 
        model_name: str, 
        data: pd.DataFrame,
        feature_columns: List[str],
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Batch prediction for large datasets."""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            X = batch[feature_columns].values
            
            predictions, probabilities = self.predict(model_name, X)
            
            for j, pred in enumerate(predictions):
                result = {
                    "index": i + j,
                    "prediction": int(pred),
                    "confidence": float(probabilities[j].max()) if probabilities is not None else None
                }
                results.append(result)
        
        return results

    def save_model(self, model_name: str, path: str):
        """Save model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models[model_name], path)
        logger.info(f"Model {model_name} saved to {path}")

    def load_model(self, model_name: str, path: str):
        """Load model from disk."""
        self.models[model_name] = joblib.load(path)
        logger.info(f"Model {model_name} loaded from {path}")

    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get feature importance for tree-based models."""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            return {f"feature_{i}": float(imp) for i, imp in enumerate(model.feature_importances_)}
        return None

    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())

# Singleton instance
ml_pipeline = MLPipeline()

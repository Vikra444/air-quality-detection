"""
Base class for air quality prediction models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from ...utils.logger import get_logger

logger = get_logger("ml.base_model")


class BaseAirQualityModel(ABC):
    """Base class for all air quality prediction models."""
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.is_trained = False
        self.training_metrics: Dict[str, float] = {}
        self.feature_columns: Optional[list] = None
        self.created_at = datetime.now()
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        pass
    
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Validate input data."""
        if X is None or len(X) == 0:
            raise ValueError("Input data X cannot be empty")
        if y is not None and len(y) != len(X):
            raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "mape": float(mape)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "training_metrics": self.training_metrics,
            "created_at": self.created_at.isoformat(),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0
        }
    
    def save_model(self, filepath: str):
        """Save model to file."""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model from file."""
        import joblib
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


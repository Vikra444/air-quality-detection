"""
Ensemble models for air quality prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from .base_model import BaseAirQualityModel
from ...utils.logger import get_logger

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logger = get_logger("ml.ensemble")


class RandomForestAirQualityModel(BaseAirQualityModel):
    """Random Forest model for air quality prediction."""
    
    def __init__(self, model_version: str = "1.0.0", **kwargs):
        super().__init__("RandomForest", model_version)
        
        self.model_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1,
            **kwargs
        }
        
        self.model = RandomForestRegressor(**self.model_params)
        self.uncertainty_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train the Random Forest model."""
        self._validate_input(X, y)
        
        logger.info(f"Training Random Forest model with {len(X)} samples")
        
        # Train main model
        self.model.fit(X, y)
        
        # Train uncertainty estimation model
        self.uncertainty_model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
        
        # Use absolute residuals as target for uncertainty model
        y_pred = self.model.predict(X)
        residuals = np.abs(y - y_pred)
        self.uncertainty_model.fit(X, residuals)
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, y_pred)
        self.is_trained = True
        
        logger.info(f"Random Forest training completed. MAE: {self.training_metrics['mae']:.3f}")
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        predictions = self.predict(X)
        
        if self.uncertainty_model is None:
            uncertainty = np.ones(len(predictions)) * self.training_metrics.get("mae", 1.0)
        else:
            uncertainty = self.uncertainty_model.predict(X)
        
        confidence = 1.0 / (1.0 + uncertainty)
        return predictions, confidence
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from Random Forest."""
        if not self.is_trained or self.feature_columns is None:
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_columns, importances.tolist()))


class XGBoostAirQualityModel(BaseAirQualityModel):
    """XGBoost model for air quality prediction."""
    
    def __init__(self, model_version: str = "1.0.0", **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for XGBoostAirQualityModel. Install with: pip install xgboost")
        super().__init__("XGBoost", model_version)
        
        self.model_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            **kwargs
        }
        
        self.model = xgb.XGBRegressor(**self.model_params)
        self.uncertainty_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train the XGBoost model."""
        self._validate_input(X, y)
        
        logger.info(f"Training XGBoost model with {len(X)} samples")
        
        # Train main model
        if validation_data:
            X_val, y_val = validation_data
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        # Train uncertainty model
        self.uncertainty_model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        y_pred = self.model.predict(X)
        residuals = np.abs(y - y_pred)
        self.uncertainty_model.fit(X, residuals)
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, y_pred)
        self.is_trained = True
        
        logger.info(f"XGBoost training completed. MAE: {self.training_metrics['mae']:.3f}")
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        predictions = self.predict(X)
        
        if self.uncertainty_model is None:
            uncertainty = np.ones(len(predictions)) * self.training_metrics.get("mae", 1.0)
        else:
            uncertainty = self.uncertainty_model.predict(X)
        
        confidence = 1.0 / (1.0 + uncertainty)
        return predictions, confidence


class EnsembleAirQualityModel(BaseAirQualityModel):
    """Ensemble of multiple models for robust predictions."""
    
    def __init__(self, models: List[BaseAirQualityModel], model_version: str = "1.0.0"):
        super().__init__("Ensemble", model_version)
        self.models = models
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train all models in the ensemble."""
        self._validate_input(X, y)
        
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        # Train each model
        model_metrics = []
        for model in self.models:
            metrics = model.fit(X, y, validation_data)
            model_metrics.append(metrics)
        
        # Calculate weights based on inverse MAE
        maes = [m["mae"] for m in model_metrics]
        self.weights = np.array([1.0 / (mae + 1e-8) for mae in maes])
        self.weights = self.weights / self.weights.sum()
        
        # Calculate ensemble metrics
        predictions = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, predictions)
        self.is_trained = True
        
        logger.info(f"Ensemble training completed. MAE: {self.training_metrics['mae']:.3f}")
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = np.array([model.predict(X) for model in self.models])
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        predictions_list = []
        confidences_list = []
        
        for model in self.models:
            pred, conf = model.predict_with_confidence(X)
            predictions_list.append(pred)
            confidences_list.append(conf)
        
        # Weighted average of predictions
        predictions = np.average(predictions_list, axis=0, weights=self.weights)
        
        # Average confidence
        confidences = np.mean(confidences_list, axis=0)
        
        return predictions, confidences


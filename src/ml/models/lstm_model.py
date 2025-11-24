"""
LSTM model for time-series air quality prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base_model import BaseAirQualityModel
from ...utils.logger import get_logger

logger = get_logger("ml.lstm")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM model will not work.")


class LSTMAirQualityModel(BaseAirQualityModel):
    """LSTM model for time-series air quality prediction."""
    
    def __init__(self, sequence_length: int = 24, model_version: str = "1.0.0", **kwargs):
        super().__init__("LSTM", model_version)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.sequence_length = sequence_length
        self.model = None
        self.uncertainty_model = None
        self.model_params = kwargs
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train the LSTM model."""
        self._validate_input(X, y)
        
        logger.info(f"Training LSTM model with {len(X)} samples")
        
        # Reshape data for LSTM (samples, timesteps, features)
        if len(X.shape) == 2:
            # If 2D, assume we need to create sequences
            X_seq = self._create_sequences(X, self.sequence_length)
            y_seq = y[self.sequence_length:]
        else:
            X_seq = X
            y_seq = y
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        # Train model
        if validation_data:
            X_val, y_val = validation_data
            if len(X_val.shape) == 2:
                X_val_seq = self._create_sequences(X_val, self.sequence_length)
                y_val_seq = y_val[self.sequence_length:]
            else:
                X_val_seq = X_val
                y_val_seq = y_val
            
            history = self.model.fit(
                X_seq, y_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50,
                batch_size=32,
                verbose=0
            )
        else:
            history = self.model.fit(
                X_seq, y_seq,
                epochs=50,
                batch_size=32,
                verbose=0
            )
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, y_pred)
        self.is_trained = True
        
        logger.info(f"LSTM training completed. MAE: {self.training_metrics['mae']:.3f}")
        return self.training_metrics
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> np.ndarray:
        """Create sequences from time series data."""
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Reshape if needed
        if len(X.shape) == 2:
            X_seq = self._create_sequences(X, self.sequence_length)
        else:
            X_seq = X
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Flatten if needed
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        predictions = self.predict(X)
        
        # Simple confidence estimation based on training MAE
        uncertainty = np.ones(len(predictions)) * self.training_metrics.get("mae", 1.0)
        confidence = 1.0 / (1.0 + uncertainty)
        
        return predictions, confidence


class CNNLSTMAirQualityModel(BaseAirQualityModel):
    """Hybrid CNN-LSTM model for spatio-temporal prediction."""
    
    def __init__(self, sequence_length: int = 24, model_version: str = "1.0.0", **kwargs):
        super().__init__("CNN-LSTM", model_version)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN-LSTM model")
        
        self.sequence_length = sequence_length
        self.model = None
        self.model_params = kwargs
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build CNN-LSTM model architecture."""
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train the CNN-LSTM model."""
        self._validate_input(X, y)
        
        logger.info(f"Training CNN-LSTM model with {len(X)} samples")
        
        # Reshape data for CNN-LSTM (samples, timesteps, features)
        if len(X.shape) == 2:
            # If 2D, assume we need to create sequences
            X_seq = self._create_sequences(X, self.sequence_length)
            y_seq = y[self.sequence_length:]
        else:
            X_seq = X
            y_seq = y
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        # Train model
        if validation_data:
            X_val, y_val = validation_data
            if len(X_val.shape) == 2:
                X_val_seq = self._create_sequences(X_val, self.sequence_length)
                y_val_seq = y_val[self.sequence_length:]
            else:
                X_val_seq = X_val
                y_val_seq = y_val
            
            history = self.model.fit(
                X_seq, y_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50,
                batch_size=32,
                verbose=0
            )
        else:
            history = self.model.fit(
                X_seq, y_seq,
                epochs=50,
                batch_size=32,
                verbose=0
            )
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, y_pred)
        self.is_trained = True
        
        logger.info(f"CNN-LSTM training completed. MAE: {self.training_metrics['mae']:.3f}")
        return self.training_metrics
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> np.ndarray:
        """Create sequences from time series data."""
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Reshape if needed
        if len(X.shape) == 2:
            X_seq = self._create_sequences(X, self.sequence_length)
        else:
            X_seq = X
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Flatten if needed
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        predictions = self.predict(X)
        
        # Simple confidence estimation based on training MAE
        uncertainty = np.ones(len(predictions)) * self.training_metrics.get("mae", 1.0)
        confidence = 1.0 / (1.0 + uncertainty)
        
        return predictions, confidence


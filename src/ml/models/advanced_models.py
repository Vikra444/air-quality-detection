"""
Advanced ML models for air quality prediction with deep learning and enhanced features.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
import pandas as pd
from .base_model import BaseAirQualityModel
from ...utils.logger import get_logger

# Optional imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, Flatten
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

logger = get_logger("ml.advanced_models")


class AttentionLSTMAirQualityModel(BaseAirQualityModel):
    """LSTM model with attention mechanism for air quality prediction."""
    
    def __init__(self, sequence_length: int = 48, model_version: str = "2.0.0", **kwargs):
        super().__init__("AttentionLSTM", model_version)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for AttentionLSTM model")
        
        self.sequence_length = sequence_length
        self.model = None
        self.model_params = kwargs
        self.feature_columns = None
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM model with attention mechanism."""
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(64, return_sequences=True)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = Flatten()(attention)
        attention = tf.nn.softmax(attention)
        attention = tf.expand_dims(attention, axis=-1)
        
        # Apply attention
        attended = tf.multiply(lstm_out, attention)
        attended = tf.reduce_sum(attended, axis=1)
        
        # Dense layers
        dense_out = Dense(64, activation='relu')(attended)
        dense_out = Dropout(0.2)(dense_out)
        dense_out = Dense(32, activation='relu')(dense_out)
        dense_out = Dropout(0.2)(dense_out)
        outputs = Dense(1, activation='linear')(dense_out)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train the Attention LSTM model."""
        self._validate_input(X, y)
        
        logger.info(f"Training Attention LSTM model with {len(X)} samples")
        
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
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
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
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
        else:
            history = self.model.fit(
                X_seq, y_seq,
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, y_pred)
        self.is_trained = True
        
        logger.info(f"Attention LSTM training completed. MAE: {self.training_metrics['mae']:.3f}")
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
        """Make predictions with confidence intervals using Monte Carlo Dropout."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Reshape if needed
        if len(X.shape) == 2:
            X_seq = self._create_sequences(X, self.sequence_length)
        else:
            X_seq = X
        
        # Monte Carlo Dropout for uncertainty estimation
        n_samples = 100
        predictions_samples = []
        
        # Enable dropout during inference
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                layer.trainable = True
        
        for _ in range(n_samples):
            pred = self.model.predict(X_seq, verbose=0)
            predictions_samples.append(pred.flatten())
        
        # Disable dropout after inference
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                layer.trainable = False
        
        predictions_samples = np.array(predictions_samples)
        mean_predictions = np.mean(predictions_samples, axis=0)
        std_predictions = np.std(predictions_samples, axis=0)
        
        # Convert to confidence (higher confidence = lower uncertainty)
        confidence = 1.0 / (1.0 + std_predictions)
        
        return mean_predictions, confidence


class MultiTaskLSTMAirQualityModel(BaseAirQualityModel):
    """Multi-task LSTM model that predicts multiple pollutants simultaneously."""
    
    def __init__(self, sequence_length: int = 48, model_version: str = "2.0.0", **kwargs):
        super().__init__("MultiTaskLSTM", model_version)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for MultiTaskLSTM model")
        
        self.sequence_length = sequence_length
        self.model = None
        self.model_params = kwargs
        self.target_columns = None  # List of target pollutants
    
    def _build_model(self, input_shape: Tuple[int, int], n_outputs: int) -> Model:
        """Build multi-task LSTM model."""
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Shared LSTM layers
        shared = LSTM(64, return_sequences=True)(inputs)
        shared = Dropout(0.2)(shared)
        shared = LSTM(64, return_sequences=False)(shared)
        shared = Dropout(0.2)(shared)
        
        # Task-specific layers for each pollutant
        outputs = []
        for i in range(n_outputs):
            task_specific = Dense(32, activation='relu', name=f'task_{i}_dense1')(shared)
            task_specific = Dropout(0.2)(task_specific)
            task_output = Dense(1, activation='linear', name=f'task_{i}_output')(task_specific)
            outputs.append(task_output)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train the Multi-task LSTM model."""
        self._validate_input(X, y)
        
        logger.info(f"Training Multi-task LSTM model with {len(X)} samples")
        
        # Reshape data for LSTM (samples, timesteps, features)
        if len(X.shape) == 2:
            # If 2D, assume we need to create sequences
            X_seq = self._create_sequences(X, self.sequence_length)
            if len(y.shape) == 2:
                # Multi-target case
                y_seq = [y[self.sequence_length:, i] for i in range(y.shape[1])]
                self.target_columns = [f"target_{i}" for i in range(y.shape[1])]
            else:
                # Single target case
                y_seq = y[self.sequence_length:]
                self.target_columns = ["aqi"]
        else:
            X_seq = X
            y_seq = y
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        n_outputs = len(y_seq) if isinstance(y_seq, list) else 1
        self.model = self._build_model(input_shape, n_outputs)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        if validation_data:
            X_val, y_val = validation_data
            if len(X_val.shape) == 2:
                X_val_seq = self._create_sequences(X_val, self.sequence_length)
                if len(y_val.shape) == 2:
                    y_val_seq = [y_val[self.sequence_length:, i] for i in range(y_val.shape[1])]
                else:
                    y_val_seq = y_val[self.sequence_length:]
            else:
                X_val_seq = X_val
                y_val_seq = y_val
            
            history = self.model.fit(
                X_seq, y_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
        else:
            history = self.model.fit(
                X_seq, y_seq,
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, y_pred)
        self.is_trained = True
        
        logger.info(f"Multi-task LSTM training completed. MAE: {self.training_metrics['mae']:.3f}")
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
        
        # Handle multiple outputs
        if isinstance(predictions, list):
            # Combine predictions into single array
            combined_pred = np.column_stack(predictions)
            return combined_pred.flatten() if combined_pred.shape[1] == 1 else combined_pred
        else:
            # Flatten if needed
            if len(predictions.shape) > 1:
                return predictions.flatten()
            return predictions
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        predictions = self.predict(X)
        
        # Simple confidence estimation based on training MAE
        uncertainty = np.ones(len(predictions)) * self.training_metrics.get("mae", 1.0)
        confidence = 1.0 / (1.0 + uncertainty)
        
        return predictions, confidence


class TransformerAirQualityModel(BaseAirQualityModel):
    """Transformer-based model for air quality prediction."""
    
    def __init__(self, sequence_length: int = 48, model_version: str = "2.0.0", **kwargs):
        super().__init__("Transformer", model_version)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Transformer model")
        
        self.sequence_length = sequence_length
        self.model = None
        self.model_params = kwargs
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build Transformer model."""
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        pos_encoding = self._positional_encoding(positions, input_shape[1])
        x = inputs + pos_encoding
        
        # Transformer blocks
        for _ in range(2):
            # Multi-head attention
            attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
            attn_output = Dropout(0.1)(attn_output)
            out1 = LayerNormalization()(x + attn_output)
            
            # Feed forward
            ffn_output = Dense(128, activation='relu')(out1)
            ffn_output = Dropout(0.1)(ffn_output)
            ffn_output = Dense(input_shape[1])(ffn_output)
            ffn_output = Dropout(0.1)(ffn_output)
            x = LayerNormalization()(out1 + ffn_output)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _positional_encoding(self, position, d_model):
        """Generate positional encoding."""
        angle_rads = self._get_angles(position, d_model)
        
        # Apply sin to even indices
        angle_rads1 = tf.cast(angle_rads[:, 0::2], dtype=tf.float32)
        angle_rads2 = tf.cast(angle_rads[:, 1::2], dtype=tf.float32)
        
        pos_encoding = tf.concat([tf.sin(angle_rads1), tf.cos(angle_rads2)], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        
        return pos_encoding
    
    def _get_angles(self, pos, d_model):
        """Get angles for positional encoding."""
        i = tf.range(d_model, dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train the Transformer model."""
        self._validate_input(X, y)
        
        logger.info(f"Training Transformer model with {len(X)} samples")
        
        # Reshape data for Transformer (samples, timesteps, features)
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
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
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
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
        else:
            history = self.model.fit(
                X_seq, y_seq,
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, y_pred)
        self.is_trained = True
        
        logger.info(f"Transformer training completed. MAE: {self.training_metrics['mae']:.3f}")
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


# Enhanced ensemble model that includes the new advanced models
class AdvancedEnsembleAirQualityModel(BaseAirQualityModel):
    """Advanced ensemble of multiple models including deep learning models."""
    
    def __init__(self, models: List[BaseAirQualityModel], model_version: str = "2.0.0"):
        super().__init__("AdvancedEnsemble", model_version)
        self.models = models
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, float]:
        """Train all models in the advanced ensemble."""
        self._validate_input(X, y)
        
        logger.info(f"Training advanced ensemble with {len(self.models)} models")
        
        # Train each model
        model_metrics = []
        trained_models = []
        
        for model in self.models:
            try:
                metrics = model.fit(X, y, validation_data)
                model_metrics.append(metrics)
                trained_models.append(model)
                logger.info(f"Successfully trained {model.model_name}")
            except Exception as e:
                logger.warning(f"Failed to train {model.model_name}: {e}")
        
        # Update models list with only successfully trained models
        self.models = trained_models
        
        if not self.models:
            raise ValueError("No models were successfully trained")
        
        # Calculate weights based on inverse MAE
        maes = [m["mae"] for m in model_metrics]
        self.weights = np.array([1.0 / (mae + 1e-8) for mae in maes])
        self.weights = self.weights / self.weights.sum()
        
        # Calculate ensemble metrics
        predictions = self.predict(X)
        self.training_metrics = self.calculate_metrics(y, predictions)
        self.is_trained = True
        
        logger.info(f"Advanced ensemble training completed. MAE: {self.training_metrics['mae']:.3f}")
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
            try:
                pred, conf = model.predict_with_confidence(X)
                predictions_list.append(pred)
                confidences_list.append(conf)
            except Exception as e:
                logger.warning(f"Error in {model.model_name} prediction: {e}")
                # Fallback to regular prediction
                pred = model.predict(X)
                uncertainty = np.ones(len(pred)) * model.training_metrics.get("mae", 1.0)
                conf = 1.0 / (1.0 + uncertainty)
                predictions_list.append(pred)
                confidences_list.append(conf)
        
        # Weighted average of predictions
        predictions = np.average(predictions_list, axis=0, weights=self.weights)
        
        # Average confidence
        confidences = np.mean(confidences_list, axis=0)
        
        return predictions, confidences
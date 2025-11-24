"""
SHAP (SHapley Additive exPlanations) explainer for air quality models.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import json
import base64
from io import BytesIO

from ...utils.logger import get_logger

logger = get_logger("ml.explainability.shap")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class SHAPExplainer:
    """SHAP explainer for air quality prediction models."""
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained ML model (sklearn, XGBoost, or TensorFlow)
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names or [
            "PM2.5", "PM10", "NO2", "CO", "O3", "SO2",
            "temperature", "humidity", "wind_speed", "wind_direction", "pressure"
        ]
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        try:
            # Try TreeExplainer for tree-based models
            if hasattr(self.model, 'predict_proba') or hasattr(self.model, 'booster'):
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized SHAP TreeExplainer")
            # Try LinearExplainer for linear models
            elif hasattr(self.model, 'coef_'):
                self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_names))))
                logger.info("Initialized SHAP LinearExplainer")
            # Use KernelExplainer as fallback
            else:
                # For neural networks or complex models, use KernelExplainer with sample
                sample_data = np.zeros((100, len(self.feature_names)))
                self.explainer = shap.KernelExplainer(self.model.predict, sample_data)
                logger.info("Initialized SHAP KernelExplainer")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}. Using KernelExplainer as fallback.")
            sample_data = np.zeros((100, len(self.feature_names)))
            self.explainer = shap.KernelExplainer(self.model.predict, sample_data)
    
    def explain(self, X: np.ndarray, max_evals: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            X: Input features (single sample or batch)
            max_evals: Maximum evaluations for KernelExplainer
        
        Returns:
            Dictionary with SHAP values, feature importance, and visualization data
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available"}
        
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Calculate SHAP values
            if isinstance(self.explainer, shap.KernelExplainer):
                shap_values = self.explainer.shap_values(X, nsamples=max_evals)
            else:
                shap_values = self.explainer.shap_values(X)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first output
            
            # Ensure shap_values is 2D
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Calculate feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Get feature contributions for the first sample
            contributions = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(shap_values[0]):
                    contributions[feature_name] = {
                        "shap_value": float(shap_values[0][i]),
                        "importance": float(feature_importance[i])
                    }
            
            # Sort by importance
            sorted_features = sorted(
                contributions.items(),
                key=lambda x: x[1]["importance"],
                reverse=True
            )
            
            # Generate summary
            top_contributors = [
                {
                    "feature": name,
                    "shap_value": data["shap_value"],
                    "importance": data["importance"],
                    "contribution_percent": (data["importance"] / feature_importance.sum() * 100) if feature_importance.sum() > 0 else 0
                }
                for name, data in sorted_features[:5]  # Top 5 features
            ]
            
            # Generate visualization (waterfall plot as base64)
            visualization = self._generate_visualization(shap_values[0], X[0])
            
            return {
                "shap_values": shap_values[0].tolist(),
                "feature_names": self.feature_names[:len(shap_values[0])],
                "feature_importance": feature_importance.tolist(),
                "top_contributors": top_contributors,
                "visualization": visualization,
                "explanation": self._generate_text_explanation(top_contributors)
            }
        
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}", error=str(e))
            return {"error": f"Failed to generate SHAP explanation: {str(e)}"}
    
    def _generate_visualization(self, shap_values: np.ndarray, feature_values: np.ndarray) -> Dict[str, str]:
        """Generate visualization data for SHAP values."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create waterfall-style bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get top features
            feature_indices = np.argsort(np.abs(shap_values))[-10:][::-1]  # Top 10
            top_shap = shap_values[feature_indices]
            top_names = [self.feature_names[i] for i in feature_indices]
            
            # Color bars based on positive/negative contribution
            colors = ['red' if x < 0 else 'green' for x in top_shap]
            
            ax.barh(range(len(top_names)), top_shap, color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names)
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title('SHAP Feature Importance')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return {
                "type": "image/png",
                "data": image_base64,
                "format": "base64"
            }
        
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")
            return {"error": "Visualization generation failed"}
    
    def _generate_text_explanation(self, top_contributors: List[Dict]) -> str:
        """Generate human-readable text explanation."""
        if not top_contributors:
            return "No significant feature contributions identified."
        
        top_feature = top_contributors[0]
        explanation = f"The prediction is primarily influenced by {top_feature['feature']} "
        
        if top_feature['shap_value'] > 0:
            explanation += f"(increases risk by {abs(top_feature['shap_value']):.2f} AQI points). "
        else:
            explanation += f"(decreases risk by {abs(top_feature['shap_value']):.2f} AQI points). "
        
        if len(top_contributors) > 1:
            other_features = [f"{c['feature']}" for c in top_contributors[1:3]]
            explanation += f"Other significant factors include {', '.join(other_features)}."
        
        return explanation


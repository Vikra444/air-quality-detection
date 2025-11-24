"""
LIME (Local Interpretable Model-agnostic Explanations) explainer for air quality models.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import base64
from io import BytesIO

from ...utils.logger import get_logger

logger = get_logger("ml.explainability.lime")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


class LIMEExplainer:
    """LIME explainer for air quality prediction models."""
    
    def __init__(self, model, training_data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained ML model
            training_data: Training data used to train the model (for sampling)
            feature_names: List of feature names
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required. Install with: pip install lime")
        
        self.model = model
        self.feature_names = feature_names or [
            "PM2.5", "PM10", "NO2", "CO", "O3", "SO2",
            "temperature", "humidity", "wind_speed", "wind_direction", "pressure"
        ]
        self.training_data = training_data
        
        # Initialize LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=self.feature_names,
            mode='regression',
            discretize_continuous=True
        )
        logger.info("Initialized LIME explainer")
    
    def explain(self, X: np.ndarray, num_features: int = 5) -> Dict[str, Any]:
        """
        Generate LIME explanations for predictions.
        
        Args:
            X: Input features (single sample)
            num_features: Number of top features to explain
        
        Returns:
            Dictionary with LIME explanations and feature importance
        """
        if not LIME_AVAILABLE:
            return {"error": "LIME not available"}
        
        try:
            # Ensure X is 1D for single sample
            if len(X.shape) > 1:
                X = X[0]
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                X,
                self.model.predict,
                num_features=num_features,
                top_labels=1
            )
            
            # Extract feature contributions
            feature_contributions = {}
            for feature_idx, contribution in explanation.as_list():
                # Find feature index
                try:
                    feat_idx = self.feature_names.index(feature_idx)
                except ValueError:
                    # Handle cases where feature name might be different
                    feat_idx = len(feature_contributions)
                
                feature_contributions[feature_idx] = {
                    "contribution": float(contribution),
                    "importance": abs(float(contribution))
                }
            
            # Sort by importance
            sorted_features = sorted(
                feature_contributions.items(),
                key=lambda x: x[1]["importance"],
                reverse=True
            )
            
            # Get top contributors
            top_contributors = [
                {
                    "feature": name,
                    "contribution": data["contribution"],
                    "importance": data["importance"],
                    "contribution_percent": (data["importance"] / sum(abs(d["contribution"]) for d in feature_contributions.values()) * 100) if feature_contributions else 0
                }
                for name, data in sorted_features[:num_features]
            ]
            
            # Generate visualization
            visualization = self._generate_visualization(explanation)
            
            # Generate text explanation
            text_explanation = self._generate_text_explanation(top_contributors)
            
            return {
                "feature_contributions": {k: v["contribution"] for k, v in feature_contributions.items()},
                "top_contributors": top_contributors,
                "visualization": visualization,
                "explanation": text_explanation,
                "predicted_value": float(self.model.predict(X.reshape(1, -1))[0])
            }
        
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}", error=str(e))
            return {"error": f"Failed to generate LIME explanation: {str(e)}"}
    
    def _generate_visualization(self, explanation) -> Dict[str, str]:
        """Generate visualization for LIME explanation."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get feature contributions
            features = []
            contributions = []
            for feature, contribution in explanation.as_list():
                features.append(feature)
                contributions.append(contribution)
            
            # Color bars
            colors = ['red' if x < 0 else 'green' for x in contributions]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, contributions, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Contribution to Prediction')
            ax.set_title('LIME Feature Contributions')
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
            logger.warning(f"Failed to generate LIME visualization: {e}")
            return {"error": "Visualization generation failed"}
    
    def _generate_text_explanation(self, top_contributors: List[Dict]) -> str:
        """Generate human-readable text explanation."""
        if not top_contributors:
            return "No significant feature contributions identified."
        
        top_feature = top_contributors[0]
        explanation = f"Local explanation: {top_feature['feature']} "
        
        if top_feature['contribution'] > 0:
            explanation += f"increases the predicted AQI by {abs(top_feature['contribution']):.2f} points. "
        else:
            explanation += f"decreases the predicted AQI by {abs(top_feature['contribution']):.2f} points. "
        
        if len(top_contributors) > 1:
            other_features = [f"{c['feature']}" for c in top_contributors[1:3]]
            explanation += f"Other local factors: {', '.join(other_features)}."
        
        return explanation


"""
Factory for creating explainability explainers.
"""

from typing import Optional, Dict, Any, List
import numpy as np

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from ...utils.logger import get_logger

logger = get_logger("ml.explainability.factory")


class ExplainerFactory:
    """Factory for creating and managing explainability explainers."""
    
    @staticmethod
    def create_shap_explainer(model, feature_names: Optional[List[str]] = None) -> Optional[SHAPExplainer]:
        """Create a SHAP explainer."""
        try:
            return SHAPExplainer(model, feature_names)
        except Exception as e:
            logger.warning(f"Failed to create SHAP explainer: {e}")
            return None
    
    @staticmethod
    def create_lime_explainer(
        model,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[LIMEExplainer]:
        """Create a LIME explainer."""
        try:
            return LIMEExplainer(model, training_data, feature_names)
        except Exception as e:
            logger.warning(f"Failed to create LIME explainer: {e}")
            return None
    
    @staticmethod
    def explain_prediction(
        model,
        X: np.ndarray,
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Generate explanation for a prediction using specified method.
        
        Args:
            model: Trained ML model
            X: Input features (single sample)
            training_data: Training data (required for LIME)
            feature_names: List of feature names
            method: Explanation method ("shap", "lime", or "both")
        
        Returns:
            Dictionary with explanations
        """
        results = {}
        
        # SHAP explanation
        if method in ["shap", "both"]:
            try:
                shap_explainer = ExplainerFactory.create_shap_explainer(model, feature_names)
                if shap_explainer:
                    results["shap"] = shap_explainer.explain(X)
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                results["shap"] = {"error": str(e)}
        
        # LIME explanation
        if method in ["lime", "both"]:
            if training_data is None:
                results["lime"] = {"error": "Training data required for LIME"}
            else:
                try:
                    lime_explainer = ExplainerFactory.create_lime_explainer(
                        model, training_data, feature_names
                    )
                    if lime_explainer:
                        results["lime"] = lime_explainer.explain(X)
                except Exception as e:
                    logger.warning(f"LIME explanation failed: {e}")
                    results["lime"] = {"error": str(e)}
        
        return results


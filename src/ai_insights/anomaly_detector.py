"""
Anomaly detector for air quality data.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger("ai_insights.anomaly_detector")


class AnomalyDetector:
    """Detect anomalies in air quality data."""
    
    def detect_anomaly(
        self,
        current_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect anomalies in current air quality data.
        
        Args:
            current_data: Current air quality data
            historical_data: Historical data for comparison
            threshold_std: Number of standard deviations for anomaly detection
        
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            if not historical_data or len(historical_data) < 5:
                return {
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "explanation": "Insufficient historical data for anomaly detection"
                }
            
            # Extract AQI values
            aqi_values = [record.get("aqi", 0) for record in historical_data]
            current_aqi = current_data.get("aqi", 0)
            
            # Calculate statistics
            mean_aqi = np.mean(aqi_values)
            std_aqi = np.std(aqi_values)
            
            # Check if current AQI is anomaly
            z_score = (current_aqi - mean_aqi) / std_aqi if std_aqi > 0 else 0
            is_anomaly = abs(z_score) > threshold_std
            
            # Calculate confidence
            confidence = min(abs(z_score) / threshold_std, 1.0) if is_anomaly else 0.0
            
            # Determine type of anomaly
            anomaly_type = None
            if is_anomaly:
                if z_score > 0:
                    anomaly_type = "spike"  # Unusually high
                else:
                    anomaly_type = "drop"  # Unusually low
            
            return {
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type,
                "z_score": float(z_score),
                "current_aqi": current_aqi,
                "mean_aqi": float(mean_aqi),
                "std_aqi": float(std_aqi),
                "confidence": confidence,
                "explanation": self._generate_anomaly_explanation(
                    is_anomaly, anomaly_type, current_aqi, mean_aqi, z_score
                )
            }
        
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}", error=str(e))
            return {"error": str(e)}
    
    def _generate_anomaly_explanation(
        self,
        is_anomaly: bool,
        anomaly_type: Optional[str],
        current_aqi: float,
        mean_aqi: float,
        z_score: float
    ) -> str:
        """Generate human-readable anomaly explanation."""
        if not is_anomaly:
            return f"Current AQI ({current_aqi:.1f}) is within normal range (mean: {mean_aqi:.1f})."
        
        if anomaly_type == "spike":
            return f"⚠️ Anomaly detected: AQI spike to {current_aqi:.1f} (normal range: {mean_aqi:.1f} ± {abs(z_score):.1f}σ). This is unusually high."
        elif anomaly_type == "drop":
            return f"✅ Anomaly detected: AQI drop to {current_aqi:.1f} (normal range: {mean_aqi:.1f} ± {abs(z_score):.1f}σ). This is unusually low."
        else:
            return f"Anomaly detected in AQI: {current_aqi:.1f} (z-score: {z_score:.2f})"


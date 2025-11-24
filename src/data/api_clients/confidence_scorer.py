"""
Data confidence scoring for API responses.
"""

from typing import Dict, Any
from datetime import datetime, timedelta
from ...utils.logger import get_logger

logger = get_logger("api_client.confidence")


class ConfidenceScorer:
    """Calculate confidence scores for API data."""
    
    @staticmethod
    def calculate_confidence(
        data: Dict[str, Any],
        latency: float,
        api_name: str,
        consistency_score: float = 1.0
    ) -> float:
        """
        Calculate confidence score based on multiple factors.
        
        Factors:
        - Data freshness (how recent is the data)
        - API latency (response time)
        - Data consistency (completeness of fields)
        - API reliability (historical performance)
        """
        # Freshness score (0-1)
        freshness_score = ConfidenceScorer._calculate_freshness(data)
        
        # Latency score (0-1, lower latency = higher score)
        latency_score = max(0, 1 - (latency / 10.0))  # 10 seconds = 0 score
        
        # Consistency score (0-1, how complete is the data)
        consistency = consistency_score
        
        # Weighted average
        confidence = (
            freshness_score * 0.4 +
            latency_score * 0.2 +
            consistency * 0.4
        )
        
        return min(1.0, max(0.0, confidence))
    
    @staticmethod
    def _calculate_freshness(data: Dict[str, Any]) -> float:
        """Calculate freshness score based on data timestamp."""
        try:
            timestamp_str = data.get("timestamp", "")
            if not timestamp_str:
                return 0.5  # Unknown timestamp = medium confidence
            
            # Parse timestamp
            if isinstance(timestamp_str, str):
                if "T" in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.fromtimestamp(float(timestamp_str))
            else:
                timestamp = timestamp_str
            
            # Calculate age in minutes
            age_minutes = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds() / 60
            
            # Freshness score: 1.0 for < 10 min, decreasing to 0.5 for > 60 min
            if age_minutes < 10:
                return 1.0
            elif age_minutes < 30:
                return 1.0 - (age_minutes - 10) * 0.01
            elif age_minutes < 60:
                return 0.8 - (age_minutes - 30) * 0.01
            else:
                return 0.5
        
        except Exception as e:
            logger.warning(f"Error calculating freshness: {e}", error=str(e))
            return 0.5
    
    @staticmethod
    def calculate_consistency(data: Dict[str, Any]) -> float:
        """Calculate data consistency/completeness score."""
        required_fields = [
            "latitude", "longitude", "aqi", "pm25", "pm10"
        ]
        
        optional_fields = [
            "no2", "co", "o3", "so2", "temperature", "humidity",
            "wind_speed", "pressure"
        ]
        
        # Check required fields
        required_count = sum(1 for field in required_fields if field in data and data[field] is not None)
        required_score = required_count / len(required_fields)
        
        # Check optional fields
        optional_count = sum(1 for field in optional_fields if field in data and data[field] is not None)
        optional_score = optional_count / len(optional_fields) * 0.3
        
        return min(1.0, required_score + optional_score)


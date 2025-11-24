"""
Cumulative exposure tracking for users.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from ..data.models import AirQualityData
from ..data.storage import storage
from ..utils.logger import get_logger

logger = get_logger("advisory.exposure_tracker")


class ExposureTracker:
    """Track cumulative air quality exposure for users."""
    
    def __init__(self):
        self.exposure_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def track_exposure(
        self,
        user_id: str,
        location_id: str,
        aqi: float,
        timestamp: Optional[datetime] = None
    ):
        """Track exposure for a user."""
        if timestamp is None:
            timestamp = datetime.now()
        
        exposure_record = {
            "location_id": location_id,
            "aqi": aqi,
            "timestamp": timestamp.isoformat(),
            "exposure_hours": 1  # Default 1 hour exposure
        }
        
        if user_id not in self.exposure_history:
            self.exposure_history[user_id] = []
        
        self.exposure_history[user_id].append(exposure_record)
        
        # Keep only last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        self.exposure_history[user_id] = [
            record for record in self.exposure_history[user_id]
            if datetime.fromisoformat(record["timestamp"]) > cutoff_date
        ]
    
    def calculate_cumulative_exposure(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Calculate cumulative exposure over specified days."""
        if user_id not in self.exposure_history:
            return {
                "user_id": user_id,
                "days": days,
                "total_exposure": 0,
                "average_aqi": 0,
                "max_aqi": 0,
                "exposure_score": 0
            }
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_exposures = [
            record for record in self.exposure_history[user_id]
            if datetime.fromisoformat(record["timestamp"]) > cutoff_date
        ]
        
        if not recent_exposures:
            return {
                "user_id": user_id,
                "days": days,
                "total_exposure": 0,
                "average_aqi": 0,
                "max_aqi": 0,
                "exposure_score": 0
            }
        
        total_exposure = sum(record["aqi"] * record["exposure_hours"] for record in recent_exposures)
        total_hours = sum(record["exposure_hours"] for record in recent_exposures)
        average_aqi = total_exposure / total_hours if total_hours > 0 else 0
        max_aqi = max(record["aqi"] for record in recent_exposures)
        
        # Calculate exposure score (0-100, higher is worse)
        exposure_score = min(100, (average_aqi / 500) * 100)
        
        return {
            "user_id": user_id,
            "days": days,
            "total_exposure": total_exposure,
            "average_aqi": average_aqi,
            "max_aqi": max_aqi,
            "exposure_score": exposure_score,
            "exposure_count": len(recent_exposures)
        }
    
    def get_exposure_history(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get exposure history for a user."""
        if user_id not in self.exposure_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            record for record in self.exposure_history[user_id]
            if datetime.fromisoformat(record["timestamp"]) > cutoff_date
        ]


# Global exposure tracker
exposure_tracker = ExposureTracker()


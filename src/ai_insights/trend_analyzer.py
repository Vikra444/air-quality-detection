"""
Trend analyzer for air quality data.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

from ..utils.logger import get_logger

logger = get_logger("ai_insights.trend_analyzer")


class TrendAnalyzer:
    """Analyze trends in air quality data."""
    
    def analyze_trend(
        self,
        historical_data: List[Dict[str, Any]],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze trend in air quality data.
        
        Args:
            historical_data: List of historical air quality records
            current_data: Current air quality data
        
        Returns:
            Dictionary with trend analysis
        """
        try:
            if not historical_data or len(historical_data) < 2:
                return {
                    "trend": "insufficient_data",
                    "direction": "unknown",
                    "change_percent": 0.0,
                    "explanation": "Insufficient historical data for trend analysis"
                }
            
            # Extract AQI values
            aqi_values = [record.get("aqi", 0) for record in historical_data]
            aqi_values.append(current_data.get("aqi", 0))
            
            # Calculate trend
            if len(aqi_values) >= 3:
                # Use linear regression for trend
                x = np.arange(len(aqi_values))
                y = np.array(aqi_values)
                
                # Simple linear regression
                slope = np.polyfit(x, y, 1)[0]
                
                # Calculate change
                first_half = np.mean(aqi_values[:len(aqi_values)//2])
                second_half = np.mean(aqi_values[len(aqi_values)//2:])
                change_percent = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                
                # Determine direction
                if abs(slope) < 0.5:
                    direction = "stable"
                    trend = "stable"
                elif slope > 0:
                    direction = "worsening"
                    trend = "increasing"
                else:
                    direction = "improving"
                    trend = "decreasing"
                
                # Calculate rate of change
                rate_of_change = abs(slope)
                
                return {
                    "trend": trend,
                    "direction": direction,
                    "change_percent": float(change_percent),
                    "rate_of_change": float(rate_of_change),
                    "slope": float(slope),
                    "explanation": self._generate_trend_explanation(direction, change_percent, rate_of_change)
                }
            else:
                return {
                    "trend": "insufficient_data",
                    "direction": "unknown",
                    "change_percent": 0.0
                }
        
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}", error=str(e))
            return {"error": str(e)}
    
    def _generate_trend_explanation(
        self,
        direction: str,
        change_percent: float,
        rate_of_change: float
    ) -> str:
        """Generate human-readable trend explanation."""
        magnitude = "significantly" if abs(change_percent) > 20 else "moderately" if abs(change_percent) > 10 else "slightly"
        
        if direction == "improving":
            return f"Air quality is {magnitude} improving (decreased by {abs(change_percent):.1f}%). "
        elif direction == "worsening":
            return f"Air quality is {magnitude} worsening (increased by {abs(change_percent):.1f}%). "
        else:
            return f"Air quality is relatively stable (change: {change_percent:.1f}%). "


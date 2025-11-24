"""
Causal analysis for understanding pollution patterns.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ....utils.logger import get_logger

logger = get_logger("ai_insights.causal_analysis")


class CausalAnalyzer:
    """Analyze causal relationships between factors and air quality."""
    
    def analyze_causal_impact(
        self,
        current_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        factor_changes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze causal impact of factor changes on air quality.
        
        Args:
            current_data: Current air quality and weather data
            historical_data: Historical data for comparison
            factor_changes: Dictionary of factor changes (e.g., {"wind_speed": 0.2})
        
        Returns:
            Dictionary with causal impact analysis
        """
        try:
            # Calculate baseline AQI
            baseline_aqi = current_data.get("aqi", 0)
            
            # Estimate impact of each factor change
            impacts = {}
            
            # Wind speed impact
            if "wind_speed" in factor_changes:
                wind_change = factor_changes["wind_speed"]
                current_wind = current_data.get("wind_speed", 0)
                # Higher wind = lower AQI (dispersal)
                wind_impact = -wind_change * 10  # Rough estimate: 1 m/s wind = -10 AQI
                impacts["wind_speed"] = {
                    "change": wind_change,
                    "aqi_impact": wind_impact,
                    "explanation": f"Wind speed change of {wind_change:.1f} m/s would impact AQI by approximately {wind_impact:.1f} points"
                }
            
            # Temperature impact
            if "temperature" in factor_changes:
                temp_change = factor_changes["temperature"]
                # Temperature inversion (cold air trapping) increases AQI
                temp_impact = -temp_change * 0.5  # Rough estimate
                impacts["temperature"] = {
                    "change": temp_change,
                    "aqi_impact": temp_impact,
                    "explanation": f"Temperature change of {temp_change:.1f}°C would impact AQI by approximately {temp_impact:.1f} points"
                }
            
            # Traffic reduction impact
            if "traffic_reduction" in factor_changes:
                traffic_reduction = factor_changes["traffic_reduction"]
                # Reduce NO2 and CO proportionally
                no2_reduction = current_data.get("no2", 0) * traffic_reduction
                co_reduction = current_data.get("co", 0) * traffic_reduction
                # Estimate AQI impact (NO2 and CO contribute to AQI)
                traffic_impact = -(no2_reduction * 0.3 + co_reduction * 0.2)  # Rough estimate
                impacts["traffic_reduction"] = {
                    "change": traffic_reduction,
                    "aqi_impact": traffic_impact,
                    "explanation": f"Traffic reduction of {traffic_reduction*100:.1f}% would reduce AQI by approximately {abs(traffic_impact):.1f} points"
                }
            
            # Green cover impact
            if "green_cover_increase" in factor_changes:
                green_increase = factor_changes["green_cover_increase"]
                # More green cover = lower PM2.5 and PM10
                pm25_reduction = current_data.get("pm25", 0) * green_increase * 0.1  # 10% of PM reduction per 10% green increase
                pm10_reduction = current_data.get("pm10", 0) * green_increase * 0.1
                green_impact = -(pm25_reduction * 0.4 + pm10_reduction * 0.3)  # Rough estimate
                impacts["green_cover_increase"] = {
                    "change": green_increase,
                    "aqi_impact": green_impact,
                    "explanation": f"Green cover increase of {green_increase*100:.1f}% would reduce AQI by approximately {abs(green_impact):.1f} points"
                }
            
            # Total estimated impact
            total_impact = sum(imp["aqi_impact"] for imp in impacts.values())
            predicted_aqi = baseline_aqi + total_impact
            
            return {
                "baseline_aqi": baseline_aqi,
                "predicted_aqi": max(0, predicted_aqi),  # AQI can't be negative
                "total_impact": total_impact,
                "factor_impacts": impacts,
                "confidence": self._calculate_confidence(impacts)
            }
        
        except Exception as e:
            logger.error(f"Error in causal analysis: {e}", error=str(e))
            return {"error": str(e)}
    
    def _calculate_confidence(self, impacts: Dict[str, Any]) -> float:
        """Calculate confidence in causal impact estimates."""
        if not impacts:
            return 0.0
        
        # Higher confidence if we have more factors analyzed
        factor_count = len(impacts)
        base_confidence = min(factor_count * 0.2, 0.8)
        
        # Adjust based on magnitude of impacts
        total_impact_magnitude = sum(abs(imp["aqi_impact"]) for imp in impacts.values())
        if total_impact_magnitude > 20:
            base_confidence += 0.1
        
        return min(base_confidence, 0.9)


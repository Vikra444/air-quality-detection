"""
Attribution explainer to explain why AQI spiked or changed.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .source_classifier import SourceClassifier, PollutionSource
from .causal_analysis import CausalAnalyzer
from ....utils.logger import get_logger

logger = get_logger("ai_insights.attribution_explainer")


class AttributionExplainer:
    """Explain why AQI changed or spiked."""
    
    def __init__(self):
        """Initialize attribution explainer."""
        self.source_classifier = SourceClassifier()
        self.causal_analyzer = CausalAnalyzer()
    
    def explain_aqi_change(
        self,
        current_data: Dict[str, Any],
        previous_data: Optional[Dict[str, Any]] = None,
        weather_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Explain why AQI changed.
        
        Args:
            current_data: Current air quality data
            previous_data: Previous air quality data for comparison
            weather_data: Current weather data
        
        Returns:
            Dictionary with explanation of AQI change
        """
        try:
            current_aqi = current_data.get("aqi", 0)
            
            # If no previous data, classify current source
            if previous_data is None:
                source_result = self.source_classifier.classify(current_data, weather_data)
                return {
                    "current_aqi": current_aqi,
                    "change": None,
                    "primary_source": source_result["primary_source"],
                    "source_confidence": source_result["confidence"],
                    "explanation": source_result["explanation"],
                    "recommendations": self._generate_recommendations(source_result["primary_source"])
                }
            
            # Calculate change
            previous_aqi = previous_data.get("aqi", 0)
            aqi_change = current_aqi - previous_aqi
            change_percent = (aqi_change / previous_aqi * 100) if previous_aqi > 0 else 0
            
            # Classify sources for both periods
            current_source = self.source_classifier.classify(current_data, weather_data)
            previous_source = self.source_classifier.classify(previous_data, weather_data)
            
            # Identify what changed
            pollutant_changes = {}
            for pollutant in ["pm25", "pm10", "no2", "co", "o3", "so2"]:
                current_val = current_data.get(pollutant, 0)
                prev_val = previous_data.get(pollutant, 0)
                if prev_val > 0:
                    change = ((current_val - prev_val) / prev_val) * 100
                    pollutant_changes[pollutant] = {
                        "current": current_val,
                        "previous": prev_val,
                        "change_percent": change
                    }
            
            # Find biggest contributor
            biggest_change = max(
                pollutant_changes.items(),
                key=lambda x: abs(x[1]["change_percent"]),
                default=None
            )
            
            # Generate explanation
            explanation = self._generate_change_explanation(
                aqi_change,
                change_percent,
                biggest_change,
                current_source,
                previous_source,
                weather_data
            )
            
            return {
                "current_aqi": current_aqi,
                "previous_aqi": previous_aqi,
                "aqi_change": aqi_change,
                "change_percent": change_percent,
                "primary_source": current_source["primary_source"],
                "source_confidence": current_source["confidence"],
                "pollutant_changes": pollutant_changes,
                "biggest_contributor": biggest_change[0] if biggest_change else None,
                "explanation": explanation,
                "recommendations": self._generate_recommendations(current_source["primary_source"])
            }
        
        except Exception as e:
            logger.error(f"Error explaining AQI change: {e}", error=str(e))
            return {"error": str(e)}
    
    def _generate_change_explanation(
        self,
        aqi_change: float,
        change_percent: float,
        biggest_change: Optional[tuple],
        current_source: Dict[str, Any],
        previous_source: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation for AQI change."""
        if abs(change_percent) < 5:
            return "AQI has remained relatively stable with minor fluctuations."
        
        direction = "increased" if aqi_change > 0 else "decreased"
        magnitude = "significantly" if abs(change_percent) > 20 else "moderately"
        
        explanation = f"AQI {direction} {magnitude} by {abs(change_percent):.1f}% "
        
        if biggest_change:
            pollutant, change_data = biggest_change
            pollutant_name = pollutant.upper().replace("_", ".")
            change_val = change_data["change_percent"]
            explanation += f"primarily due to {pollutant_name} {direction} by {abs(change_val):.1f}%. "
        
        # Add source attribution
        if current_source["primary_source"] != previous_source["primary_source"]:
            explanation += f"Primary pollution source changed from {previous_source['primary_source']} to {current_source['primary_source']}. "
        else:
            explanation += f"Primary source remains {current_source['primary_source']}. "
        
        # Add weather context
        if weather_data:
            wind_speed = weather_data.get("wind_speed", 0)
            if wind_speed < 1.5 and aqi_change > 0:
                explanation += "Low wind speed is causing pollutant accumulation. "
            elif wind_speed > 5.0 and aqi_change < 0:
                explanation += "Higher wind speed is helping disperse pollutants. "
        
        return explanation
    
    def _generate_recommendations(self, source: str) -> List[str]:
        """Generate recommendations based on pollution source."""
        recommendations = {
            PollutionSource.TRAFFIC: [
                "Consider using public transportation or carpooling",
                "Avoid outdoor exercise during rush hours",
                "Use N95 masks if commuting by vehicle"
            ],
            PollutionSource.CROP_BURNING: [
                "Stay indoors, especially during morning hours",
                "Use air purifiers with HEPA filters",
                "Keep windows closed to prevent smoke entry"
            ],
            PollutionSource.WEATHER: [
                "Monitor weather conditions for improvement",
                "Limit outdoor activities until wind speed increases",
                "Use indoor air purifiers"
            ],
            PollutionSource.INDUSTRIAL: [
                "Report industrial emissions if excessive",
                "Stay away from industrial areas if possible",
                "Use air purifiers at home"
            ],
            PollutionSource.CONSTRUCTION: [
                "Avoid areas with active construction",
                "Wear masks if near construction sites",
                "Keep windows closed if living near construction"
            ],
            PollutionSource.UNKNOWN: [
                "Monitor air quality regularly",
                "Follow general air quality guidelines",
                "Use air purifiers as precaution"
            ]
        }
        
        return recommendations.get(source, recommendations[PollutionSource.UNKNOWN])


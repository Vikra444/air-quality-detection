"""
Source classifier to identify pollution sources (traffic, crop burning, weather, industrial).
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from ....utils.logger import get_logger

logger = get_logger("ai_insights.source_classifier")


class PollutionSource(str, Enum):
    """Pollution source types."""
    TRAFFIC = "traffic"
    CROP_BURNING = "crop_burning"
    WEATHER = "weather"
    INDUSTRIAL = "industrial"
    CONSTRUCTION = "construction"
    UNKNOWN = "unknown"


class SourceClassifier:
    """Classify pollution sources based on pollutant patterns and external factors."""
    
    def __init__(self):
        """Initialize source classifier with rule-based and ML-based patterns."""
        self.source_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for different pollution sources."""
        return {
            PollutionSource.TRAFFIC: {
                "indicators": {
                    "no2_high": True,  # High NO2 indicates traffic
                    "co_high": True,    # High CO indicates vehicular emissions
                    "pm25_moderate": True,
                    "time_of_day": ["morning", "evening"],  # Rush hours
                    "wind_speed_low": True  # Low wind = accumulation
                },
                "weights": {
                    "no2": 0.4,
                    "co": 0.3,
                    "pm25": 0.2,
                    "time_factor": 0.1
                }
            },
            PollutionSource.CROP_BURNING: {
                "indicators": {
                    "pm25_very_high": True,
                    "pm10_very_high": True,
                    "season": ["october", "november", "december"],  # Harvest season
                    "wind_direction": "northwest",  # Common in India
                    "visibility_low": True
                },
                "weights": {
                    "pm25": 0.5,
                    "pm10": 0.3,
                    "season_factor": 0.2
                }
            },
            PollutionSource.WEATHER: {
                "indicators": {
                    "wind_speed_very_low": True,  # Stagnant air
                    "temperature_inversion": True,
                    "humidity_high": True,
                    "pressure_high": True,
                    "all_pollutants_accumulated": True
                },
                "weights": {
                    "wind_speed": 0.4,
                    "temperature": 0.3,
                    "humidity": 0.2,
                    "pressure": 0.1
                }
            },
            PollutionSource.INDUSTRIAL: {
                "indicators": {
                    "so2_high": True,  # Industrial emissions
                    "pm25_high": True,
                    "no2_moderate": True,
                    "consistent_pattern": True,  # Steady emissions
                    "time_of_day": ["all"]  # 24/7 operations
                },
                "weights": {
                    "so2": 0.5,
                    "pm25": 0.3,
                    "no2": 0.2
                }
            },
            PollutionSource.CONSTRUCTION: {
                "indicators": {
                    "pm10_very_high": True,  # Dust from construction
                    "pm25_moderate": True,
                    "wind_speed_moderate": True,
                    "time_of_day": ["daytime"]
                },
                "weights": {
                    "pm10": 0.6,
                    "pm25": 0.4
                }
            }
        }
    
    def classify(
        self,
        air_quality_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify the primary pollution source.
        
        Args:
            air_quality_data: Dictionary with pollutant values
            weather_data: Optional weather data
        
        Returns:
            Dictionary with source classification and confidence
        """
        try:
            scores = {}
            
            # Extract pollutant values
            pm25 = air_quality_data.get("pm25", 0)
            pm10 = air_quality_data.get("pm10", 0)
            no2 = air_quality_data.get("no2", 0)
            co = air_quality_data.get("co", 0)
            so2 = air_quality_data.get("so2", 0)
            aqi = air_quality_data.get("aqi", 0)
            
            # Extract weather data
            wind_speed = weather_data.get("wind_speed", 0) if weather_data else 0
            wind_direction = weather_data.get("wind_direction", 0) if weather_data else 0
            temperature = weather_data.get("temperature", 0) if weather_data else 0
            humidity = weather_data.get("humidity", 0) if weather_data else 0
            pressure = weather_data.get("pressure", 0) if weather_data else 0
            
            # Get current time and season
            current_time = datetime.now()
            hour = current_time.hour
            month = current_time.month
            
            # Determine time of day
            if 6 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 18:
                time_of_day = "daytime"
            elif 18 <= hour < 22:
                time_of_day = "evening"
            else:
                time_of_day = "night"
            
            # Determine season (for India)
            if month in [10, 11, 12, 1]:
                season = "winter_harvest"
            elif month in [2, 3, 4, 5]:
                season = "summer"
            else:
                season = "monsoon"
            
            # Score each source
            for source, pattern in self.source_patterns.items():
                score = 0.0
                indicators = pattern["indicators"]
                weights = pattern["weights"]
                
                # Traffic scoring
                if source == PollutionSource.TRAFFIC:
                    if no2 > 50:  # High NO2
                        score += weights["no2"] * min(no2 / 100, 1.0)
                    if co > 1.0:  # High CO
                        score += weights["co"] * min(co / 5.0, 1.0)
                    if 30 < pm25 < 100:  # Moderate PM2.5
                        score += weights["pm25"] * 0.5
                    if time_of_day in ["morning", "evening"]:
                        score += weights["time_factor"] * 1.0
                    if wind_speed < 2.0:  # Low wind
                        score += 0.2
                
                # Crop burning scoring
                elif source == PollutionSource.CROP_BURNING:
                    if pm25 > 150:  # Very high PM2.5
                        score += weights["pm25"] * min(pm25 / 300, 1.0)
                    if pm10 > 200:  # Very high PM10
                        score += weights["pm10"] * min(pm10 / 400, 1.0)
                    if season == "winter_harvest":
                        score += weights["season_factor"] * 1.0
                    if 270 <= wind_direction <= 360 or 0 <= wind_direction <= 90:  # NW/N
                        score += 0.1
                
                # Weather scoring
                elif source == PollutionSource.WEATHER:
                    if wind_speed < 1.0:  # Very low wind
                        score += weights["wind_speed"] * 1.0
                    elif wind_speed < 2.0:
                        score += weights["wind_speed"] * 0.7
                    if temperature < 15:  # Temperature inversion conditions
                        score += weights["temperature"] * 0.5
                    if humidity > 70:
                        score += weights["humidity"] * 0.3
                    if pressure > 1020:
                        score += weights["pressure"] * 0.2
                    # If all pollutants are elevated, likely weather-related accumulation
                    if pm25 > 50 and pm10 > 80 and no2 > 30:
                        score += 0.3
                
                # Industrial scoring
                elif source == PollutionSource.INDUSTRIAL:
                    if so2 > 20:  # High SO2
                        score += weights["so2"] * min(so2 / 50, 1.0)
                    if pm25 > 100:
                        score += weights["pm25"] * min(pm25 / 200, 1.0)
                    if 30 < no2 < 80:  # Moderate NO2
                        score += weights["no2"] * 0.5
                    # Industrial sources are consistent
                    score += 0.2
                
                # Construction scoring
                elif source == PollutionSource.CONSTRUCTION:
                    if pm10 > 150:  # Very high PM10 (dust)
                        score += weights["pm10"] * min(pm10 / 300, 1.0)
                    if 50 < pm25 < 120:
                        score += weights["pm25"] * 0.5
                    if time_of_day == "daytime":
                        score += 0.2
                    if 1.0 < wind_speed < 5.0:  # Moderate wind spreads dust
                        score += 0.1
                
                scores[source.value] = min(score, 1.0)  # Cap at 1.0
            
            # Find primary source
            primary_source = max(scores.items(), key=lambda x: x[1])
            
            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                normalized_scores = {k: v / total_score for k, v in scores.items()}
            else:
                normalized_scores = {k: 0.0 for k in scores.keys()}
                primary_source = (PollutionSource.UNKNOWN.value, 0.0)
            
            return {
                "primary_source": primary_source[0],
                "confidence": float(primary_source[1]),
                "source_scores": normalized_scores,
                "all_scores": scores,
                "explanation": self._generate_explanation(
                    primary_source[0],
                    air_quality_data,
                    weather_data
                )
            }
        
        except Exception as e:
            logger.error(f"Error classifying source: {e}", error=str(e))
            return {
                "primary_source": PollutionSource.UNKNOWN.value,
                "confidence": 0.0,
                "source_scores": {},
                "error": str(e)
            }
    
    def _generate_explanation(
        self,
        source: str,
        air_quality_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation for source classification."""
        explanations = {
            PollutionSource.TRAFFIC: f"High NO2 ({air_quality_data.get('no2', 0):.1f} ppb) and CO levels indicate vehicular emissions. "
                                     f"Traffic congestion during rush hours contributes to air pollution.",
            PollutionSource.CROP_BURNING: f"Very high PM2.5 ({air_quality_data.get('pm25', 0):.1f} μg/m³) and PM10 levels suggest crop residue burning. "
                                         f"This is common during harvest season (Oct-Dec) in agricultural regions.",
            PollutionSource.WEATHER: f"Low wind speed ({weather_data.get('wind_speed', 0) if weather_data else 0:.1f} m/s) and atmospheric conditions "
                                     f"are causing pollutant accumulation. Temperature inversion may be trapping pollutants near the ground.",
            PollutionSource.INDUSTRIAL: f"Elevated SO2 ({air_quality_data.get('so2', 0):.1f} ppb) and particulate matter indicate industrial emissions. "
                                        f"Factories and power plants are likely contributing to air pollution.",
            PollutionSource.CONSTRUCTION: f"High PM10 ({air_quality_data.get('pm10', 0):.1f} μg/m³) levels suggest construction dust. "
                                         f"Active construction sites in the area are generating particulate matter.",
            PollutionSource.UNKNOWN: "Unable to determine primary pollution source with high confidence."
        }
        
        return explanations.get(source, explanations[PollutionSource.UNKNOWN])


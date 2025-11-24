"""
Data preprocessing and validation.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime
from ..utils.logger import get_logger
from ..utils.exceptions import DataValidationError
from .models import AirQualityData, AreaType
from .quality_assurance import quality_assurance

logger = get_logger("preprocessing")


class AirQualityPreprocessor:
    """Preprocessor for air quality data."""
    
    def __init__(self):
        self.pm25_threshold = 500
        self.pm10_threshold = 600
        self.no2_threshold = 400
        self.co_threshold = 50
        self.o3_threshold = 500
        self.so2_threshold = 1000
        self.quality_assurance = quality_assurance
    
    def validate_and_clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean air quality data."""
        try:
            # Remove outliers
            cleaned = self._remove_outliers(data)
            
            # Fill missing values
            filled = self._fill_missing_values(cleaned)
            
            # Validate ranges
            validated = self._validate_ranges(filled)
            
            return validated
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}", error=str(e))
            raise DataValidationError(f"Preprocessing failed: {e}")
    
    def _remove_outliers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove statistical outliers using IQR method."""
        cleaned = data.copy()
        
        pollutants = ["pm25", "pm10", "no2", "co", "o3", "so2"]
        
        for pollutant in pollutants:
            if pollutant in cleaned and cleaned[pollutant] is not None:
                value = cleaned[pollutant]
                
                # Simple threshold-based outlier removal
                threshold_map = {
                    "pm25": self.pm25_threshold,
                    "pm10": self.pm10_threshold,
                    "no2": self.no2_threshold,
                    "co": self.co_threshold,
                    "o3": self.o3_threshold,
                    "so2": self.so2_threshold
                }
                
                if value > threshold_map.get(pollutant, float('inf')):
                    logger.warning(
                        f"Outlier detected for {pollutant}: {value}",
                        pollutant=pollutant,
                        value=value
                    )
                    cleaned[pollutant] = threshold_map[pollutant]
        
        return cleaned
    
    def _fill_missing_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing values with reasonable defaults or interpolations."""
        filled = data.copy()
        
        # Fill missing pollutants with 0 (not detected)
        pollutants = ["pm25", "pm10", "no2", "co", "o3", "so2"]
        for pollutant in pollutants:
            if pollutant not in filled or filled[pollutant] is None:
                filled[pollutant] = 0.0
        
        # Fill missing weather data
        if "temperature" not in filled or filled["temperature"] is None:
            filled["temperature"] = 20.0  # Default temperature
        
        if "humidity" not in filled or filled["humidity"] is None:
            filled["humidity"] = 50.0  # Default humidity
        
        if "wind_speed" not in filled or filled["wind_speed"] is None:
            filled["wind_speed"] = 3.0  # Default wind speed
        
        if "pressure" not in filled or filled["pressure"] is None:
            filled["pressure"] = 1013.0  # Default pressure
        
        return filled
    
    def _validate_ranges(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data ranges."""
        validated = data.copy()
        
        # Validate AQI
        if "aqi" in validated:
            validated["aqi"] = max(0, min(500, validated["aqi"]))
        
        # Validate coordinates
        if "latitude" in validated:
            validated["latitude"] = max(-90, min(90, validated["latitude"]))
        
        if "longitude" in validated:
            validated["longitude"] = max(-180, min(180, validated["longitude"]))
        
        # Validate humidity
        if "humidity" in validated:
            validated["humidity"] = max(0, min(100, validated["humidity"]))
        
        # Validate wind direction
        if "wind_direction" in validated and validated["wind_direction"] is not None:
            validated["wind_direction"] = validated["wind_direction"] % 360
        
        return validated
    
    def classify_area_type(
        self,
        latitude: float,
        longitude: float,
        pm25: float,
        building_density: Optional[float] = None
    ) -> AreaType:
        """
        Classify area as urban or peri-urban.
        
        Simple heuristic-based classification.
        Can be enhanced with geospatial data.
        """
        # Simple classification based on PM2.5 levels
        # Urban areas typically have higher PM2.5
        if pm25 > 35:  # WHO guideline for urban areas
            return AreaType.URBAN
        elif pm25 > 15:
            return AreaType.PERI_URBAN
        else:
            return AreaType.RURAL
    
    def calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate data quality score using quality assurance module."""
        # Convert dict to AirQualityData for quality scoring
        try:
            aq_data = AirQualityData(
                timestamp=data.get("timestamp", datetime.now()),
                location_id=data.get("location_id", "unknown"),
                latitude=data.get("latitude", 0),
                longitude=data.get("longitude", 0),
                pm25=data.get("pm25", 0),
                pm10=data.get("pm10", 0),
                no2=data.get("no2", 0),
                co=data.get("co", 0),
                o3=data.get("o3", 0),
                so2=data.get("so2", 0),
                aqi=data.get("aqi", 0),
                primary_pollutant=data.get("primary_pollutant"),
                temperature=data.get("temperature"),
                humidity=data.get("humidity"),
                wind_speed=data.get("wind_speed"),
                wind_direction=data.get("wind_direction"),
                pressure=data.get("pressure"),
                source_api=data.get("source_api"),
                quality_score=data.get("quality_score"),
                confidence_score=data.get("confidence_score"),
                area_type=data.get("area_type")
            )
            
            return self.quality_assurance.calculate_data_quality_score(aq_data)
        except Exception as e:
            logger.warning(f"Could not calculate quality score: {e}")
            return 0.5  # Default score
    
    def normalize_to_model(self, data: Dict[str, Any]) -> AirQualityData:
        """Convert dictionary to AirQualityData model with quality assurance."""
        # Preprocess data
        cleaned = self.validate_and_clean(data)
        
        # Ensure location_id is present (required field)
        if "location_id" not in cleaned or not cleaned["location_id"]:
            cleaned["location_id"] = f"Location_{cleaned.get('latitude', 0)}_{cleaned.get('longitude', 0)}"
        
        # Calculate quality score if not present
        if "quality_score" not in cleaned:
            cleaned["quality_score"] = self.calculate_quality_score(cleaned)
        
        # Classify area type if not present
        if "area_type" not in cleaned:
            cleaned["area_type"] = self.classify_area_type(
                cleaned.get("latitude", 0),
                cleaned.get("longitude", 0),
                cleaned.get("pm25", 0)
            )
        
        # Ensure timestamp is datetime
        if "timestamp" in cleaned and isinstance(cleaned["timestamp"], str):
            cleaned["timestamp"] = datetime.fromisoformat(cleaned["timestamp"])
        elif "timestamp" not in cleaned:
            cleaned["timestamp"] = datetime.now()
        
        # Clean data using quality assurance
        try:
            aq_data = AirQualityData(**cleaned)
            cleaned_data = self.quality_assurance.clean_data(aq_data)
            
            # Validate the cleaned data
            is_valid, issues = self.quality_assurance.validate_data(cleaned_data)
            if not is_valid:
                logger.warning(f"Data validation issues after cleaning: {issues}")
            
            return cleaned_data
        except Exception as e:
            logger.error(f"Error in final data validation: {e}")
            # Fallback to original data
            return AirQualityData(**cleaned)
"""
Data validation helpers for AirGuard system.
"""

from typing import Any, Dict, Optional
from .exceptions import DataValidationError


def validate_latitude(lat: float) -> bool:
    """Validate latitude value."""
    if not isinstance(lat, (int, float)):
        raise DataValidationError(f"Latitude must be numeric, got {type(lat)}")
    if not -90 <= lat <= 90:
        raise DataValidationError(f"Latitude must be between -90 and 90, got {lat}")
    return True


def validate_longitude(lon: float) -> bool:
    """Validate longitude value."""
    if not isinstance(lon, (int, float)):
        raise DataValidationError(f"Longitude must be numeric, got {type(lon)}")
    if not -180 <= lon <= 180:
        raise DataValidationError(f"Longitude must be between -180 and 180, got {lon}")
    return True


def validate_aqi(aqi: float) -> bool:
    """Validate AQI value."""
    if not isinstance(aqi, (int, float)):
        raise DataValidationError(f"AQI must be numeric, got {type(aqi)}")
    if aqi < 0 or aqi > 500:
        raise DataValidationError(f"AQI must be between 0 and 500, got {aqi}")
    return True


def validate_pollutant_value(value: float, pollutant: str, max_value: Optional[float] = None) -> bool:
    """Validate pollutant value."""
    if not isinstance(value, (int, float)):
        raise DataValidationError(f"{pollutant} must be numeric, got {type(value)}")
    if value < 0:
        raise DataValidationError(f"{pollutant} cannot be negative, got {value}")
    if max_value and value > max_value:
        raise DataValidationError(f"{pollutant} exceeds maximum value {max_value}, got {value}")
    return True


def validate_air_quality_data(data: Dict[str, Any]) -> bool:
    """Validate complete air quality data structure."""
    required_fields = ["latitude", "longitude", "aqi", "pm25", "pm10"]
    
    for field in required_fields:
        if field not in data:
            raise DataValidationError(f"Missing required field: {field}")
    
    validate_latitude(data["latitude"])
    validate_longitude(data["longitude"])
    validate_aqi(data["aqi"])
    validate_pollutant_value(data["pm25"], "PM2.5", 500)
    validate_pollutant_value(data["pm10"], "PM10", 600)
    
    return True


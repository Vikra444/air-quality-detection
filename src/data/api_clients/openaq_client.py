"""
OpenAQ API client.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from .base_client import BaseAPIClient
from ...config.settings import settings
from ...utils.logger import get_logger
from ...utils.exceptions import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = get_logger("api_client.openaq")


class OpenAQClient(BaseAPIClient):
    """Client for OpenAQ API."""
    
    def __init__(self):
        super().__init__(
            api_key=settings.openaq_api_key,
            base_url=settings.openaq_base_url,
            timeout=30
        )
        self.min_request_interval = 2.0  # Increase interval to reduce rate limiting
    
    async def fetch_air_quality(
        self,
        latitude: float,
        longitude: float,
        radius: int = 10000,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Fetch air quality data from OpenAQ."""
        try:
            # OpenAQ uses locations endpoint with coordinates
            data = await self._make_request(
                method="GET",
                endpoint="/locations",
                params={
                    "coordinates": f"{latitude},{longitude}",
                    "radius": radius,
                    "limit": 1
                },
                headers={"X-API-Key": self.api_key} if self.api_key else None
            )
            
            # Validate response
            if not isinstance(data, dict):
                logger.warning(f"OpenAQ returned invalid data type: {type(data)}")
                return None
                
            if not data.get("results"):
                logger.warning("No OpenAQ data found for location")
                return None
            
            # Get latest measurements with better error handling
            location_id = data["results"][0]["id"]
            try:
                measurements = await self._make_request(
                    method="GET",
                    endpoint="/latest",
                    params={
                        "location_id": location_id,
                        "limit": 1
                    },
                    headers={"X-API-Key": self.api_key} if self.api_key else None
                )
                
                # Validate measurements response
                if not isinstance(measurements, dict):
                    logger.warning(f"OpenAQ measurements returned invalid data type: {type(measurements)}")
                    return None
                    
            except APIError as e:
                logger.warning(f"OpenAQ measurements API error: {e}")
                return None
            except Exception as e:
                logger.warning(f"OpenAQ measurements error: {e}")
                return None
            
            combined_data = {
                "location": data["results"][0],
                "measurements": measurements
            }
            
            return self.normalize_data(combined_data)
        
        except APIError as e:
            logger.error(f"OpenAQ API error: {e}", error=str(e))
            return None
        except Exception as e:
            logger.error(f"Error fetching OpenAQ data: {e}", error=str(e))
            return None
    
    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OpenAQ response to standard format."""
        try:
            # Validate input data
            if not isinstance(raw_data, dict):
                raise ValueError(f"Expected dict, got {type(raw_data)}")
                
            location = raw_data.get("location", {})
            measurements_data = raw_data.get("measurements", {})
            measurements = measurements_data.get("results", [])
            
            # Validate measurements
            if not isinstance(measurements, list):
                raise ValueError(f"Expected measurements list, got {type(measurements)}")
            
            # Extract coordinates
            coordinates = location.get("coordinates", {})
            latitude = coordinates.get("latitude", 0) if isinstance(coordinates, dict) else 0
            longitude = coordinates.get("longitude", 0) if isinstance(coordinates, dict) else 0
            
            # Extract pollutant values from measurements
            pollutants = {}
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    continue
                parameter = measurement.get("parameter", "").lower()
                value = measurement.get("value", 0)
                pollutants[parameter] = value
            
            # Calculate AQI from pollutants (simplified)
            aqi = self._calculate_aqi(pollutants)
            
            normalized = {
                "timestamp": measurements[0].get("date", {}).get("utc", datetime.now().isoformat()) if measurements and isinstance(measurements[0], dict) and isinstance(measurements[0].get("date", {}), dict) else datetime.now().isoformat(),
                "latitude": latitude,
                "longitude": longitude,
                "pm25": pollutants.get("pm25", 0),
                "pm10": pollutants.get("pm10", 0),
                "no2": pollutants.get("no2", 0),
                "co": pollutants.get("co", 0),
                "o3": pollutants.get("o3", 0),
                "so2": pollutants.get("so2", 0),
                "aqi": aqi,
                "primary_pollutant": max(pollutants.items(), key=lambda x: x[1])[0].upper() if pollutants else "PM2.5",
                "temperature": 0,  # OpenAQ doesn't provide weather data
                "humidity": 0,
                "pressure": 0,
                "wind_speed": 0,
                "wind_direction": 0,
                "source": "openaq",
                "quality_score": 0.75
            }
            
            return normalized
        
        except Exception as e:
            logger.error(f"Error normalizing OpenAQ data: {e}", error=str(e))
            return {}  # Return empty dict instead of raising exception
    
    def _calculate_aqi(self, pollutants: Dict[str, float]) -> float:
        """Calculate AQI from pollutant values."""
        # Simplified AQI calculation
        if not pollutants:
            return 0
        
        # Use PM2.5 as primary indicator if available
        if "pm25" in pollutants:
            pm25 = pollutants["pm25"]
            if pm25 <= 12:
                return pm25 * 4.17  # 0-50 range
            elif pm25 <= 35.4:
                return 50 + (pm25 - 12) * 1.47  # 51-100 range
            elif pm25 <= 55.4:
                return 100 + (pm25 - 35.4) * 2.5  # 101-150 range
            else:
                return 150 + (pm25 - 55.4) * 1.67  # 151+ range
        
        return 0
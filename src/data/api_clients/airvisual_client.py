"""
AirVisual (IQAir) API client.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from .base_client import BaseAPIClient
from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger("api_client.airvisual")


class AirVisualClient(BaseAPIClient):
    """Client for AirVisual (IQAir) API."""
    
    def __init__(self):
        super().__init__(
            api_key=settings.airvisual_api_key,
            base_url=settings.airvisual_base_url,
            timeout=30
        )
        self.min_request_interval = 1.0  # AirVisual rate limiting
    
    async def fetch_air_quality(
        self,
        latitude: float,
        longitude: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Fetch air quality data from AirVisual."""
        if not self.api_key:
            logger.warning("AirVisual API key not configured")
            return None
        
        try:
            # AirVisual uses nearest city endpoint
            data = await self._make_request(
                method="GET",
                endpoint="/nearest_city",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "key": self.api_key
                }
            )
            
            return self.normalize_data(data)
        
        except Exception as e:
            logger.error(f"Error fetching AirVisual data: {e}", error=str(e))
            return None
    
    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize AirVisual response to standard format."""
        try:
            data = raw_data["data"]
            pollution = data["current"]["pollution"]
            weather = data["current"]["weather"]
            location = data["location"]["coordinates"]
            
            normalized = {
                "timestamp": datetime.now().isoformat(),  # AirVisual doesn't provide exact timestamp
                "latitude": location[1],
                "longitude": location[0],
                "pm25": pollution.get("aqius", 0) * 0.5,  # Approximate conversion
                "pm10": pollution.get("aqius", 0) * 0.6,
                "no2": pollution.get("no2", 0),
                "co": pollution.get("co", 0),
                "o3": pollution.get("o3", 0),
                "so2": pollution.get("so2", 0),
                "aqi": pollution.get("aqius", 0),
                "primary_pollutant": pollution.get("mainus", "PM2.5"),
                "temperature": weather.get("tp", 0),
                "humidity": weather.get("hu", 0),
                "pressure": weather.get("pr", 0),
                "wind_speed": weather.get("ws", 0),
                "wind_direction": weather.get("wd", 0),
                "source": "airvisual",
                "quality_score": 0.85
            }
            
            return normalized
        
        except Exception as e:
            logger.error(f"Error normalizing AirVisual data: {e}", error=str(e))
            raise


"""
AQICN API client.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from .base_client import BaseAPIClient
from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger("api_client.aqicn")


class AQICNClient(BaseAPIClient):
    """Client for AQICN API."""
    
    def __init__(self):
        super().__init__(
            api_key=settings.aqicn_api_key,
            base_url=settings.aqicn_base_url,
            timeout=30
        )
        self.min_request_interval = 1.0
    
    async def fetch_air_quality(
        self,
        latitude: float,
        longitude: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Fetch air quality data from AQICN."""
        try:
            # AQICN uses feed/geo endpoint with correct parameters
            data = await self._make_request(
                method="GET",
                endpoint=f"/feed/geo:{latitude};{longitude}/",
                params={
                    "token": self.api_key or "demo"  # AQICN allows demo token
                }
            )
            
            # Check if we got valid data
            if not isinstance(data, dict):
                logger.error(f"AQICN returned invalid data type: {type(data)}")
                return None
                
            if data.get("status") != "ok":
                logger.error(f"AQICN returned error status: {data.get('status')}")
                return None
                
            return self.normalize_data(data)
        
        except Exception as e:
            logger.error(f"Error fetching AQICN data: {e}", error=str(e))
            return None
    
    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize AQICN response to standard format."""
        try:
            # Validate input data
            if not isinstance(raw_data, dict):
                raise ValueError(f"Expected dict, got {type(raw_data)}")
                
            data = raw_data.get("data", {})
            if not isinstance(data, dict):
                raise ValueError(f"Expected data dict, got {type(data)}")
                
            iaqi = data.get("iaqi", {})
            city = data.get("city", {})
            time_data = data.get("time", {})
            
            # Extract pollutant values
            pm25 = iaqi.get("pm25", {}).get("v", 0) if isinstance(iaqi.get("pm25"), dict) else 0
            pm10 = iaqi.get("pm10", {}).get("v", 0) if isinstance(iaqi.get("pm10"), dict) else 0
            no2 = iaqi.get("no2", {}).get("v", 0) if isinstance(iaqi.get("no2"), dict) else 0
            co = iaqi.get("co", {}).get("v", 0) if isinstance(iaqi.get("co"), dict) else 0
            o3 = iaqi.get("o3", {}).get("v", 0) if isinstance(iaqi.get("o3"), dict) else 0
            so2 = iaqi.get("so2", {}).get("v", 0) if isinstance(iaqi.get("so2"), dict) else 0
            
            normalized = {
                "timestamp": time_data.get("iso", datetime.now().isoformat()),
                "latitude": city.get("geo", [0, 0])[0] if isinstance(city.get("geo"), list) and len(city.get("geo", [])) >= 2 else 0,
                "longitude": city.get("geo", [0, 0])[1] if isinstance(city.get("geo"), list) and len(city.get("geo", [])) >= 2 else 0,
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
                "co": co,
                "o3": o3,
                "so2": so2,
                "aqi": data.get("aqi", 0),
                "primary_pollutant": data.get("dominentpol", "PM2.5"),
                "temperature": data.get("iaqi", {}).get("t", {}).get("v", 0) if isinstance(data.get("iaqi", {}).get("t"), dict) else 0,
                "humidity": data.get("iaqi", {}).get("h", {}).get("v", 0) if isinstance(data.get("iaqi", {}).get("h"), dict) else 0,
                "pressure": data.get("iaqi", {}).get("p", {}).get("v", 0) if isinstance(data.get("iaqi", {}).get("p"), dict) else 0,
                "wind_speed": data.get("iaqi", {}).get("w", {}).get("v", 0) if isinstance(data.get("iaqi", {}).get("w"), dict) else 0,
                "wind_direction": 0,  # AQICN doesn't provide wind direction
                "source": "aqicn",
                "quality_score": 0.8
            }
            
            return normalized
        
        except Exception as e:
            logger.error(f"Error normalizing AQICN data: {e}", error=str(e))
            raise
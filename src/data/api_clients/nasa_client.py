"""
NASA API client for satellite imagery and air quality data.
"""

import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime

from .base_client import BaseAPIClient
from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger("api_clients.nasa")


class NASAClient(BaseAPIClient):
    """NASA API client for satellite imagery and air quality data."""
    
    def __init__(self):
        """Initialize NASA client."""
        super().__init__()
        self.api_key = settings.nasa_api_key or "DEMO_KEY"  # NASA provides demo key
        self.base_url = "https://api.nasa.gov"
    
    async def fetch_air_quality(
        self,
        latitude: float,
        longitude: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch air quality data from NASA APIs.
        
        Note: NASA doesn't have direct air quality API, but we can use:
        - Earth Imagery API for visual data
        - POWER API for weather/climate data that affects air quality
        """
        try:
            # Use NASA POWER API for weather data that affects air quality
            # This provides meteorological parameters
            url = f"{self.base_url}/POWER/v1/point"
            
            params = {
                "parameters": "T2M,RH2M,WS2M,PRECTOTCORR",
                "community": "AG",
                "longitude": longitude,
                "latitude": latitude,
                "start": datetime.now().strftime("%Y%m%d"),
                "end": datetime.now().strftime("%Y%m%d"),
                "format": "JSON",
                "api_key": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Validate response structure
                        if not isinstance(data, dict):
                            logger.warning("NASA API returned invalid data structure")
                            return None
                            
                        # Extract weather data
                        if "properties" in data and "parameter" in data["properties"]:
                            params_data = data["properties"]["parameter"]
                            
                            # Get latest values with better error handling
                            temp = 20
                            humidity = 50
                            wind_speed = 3
                            precipitation = 0
                            
                            if isinstance(params_data.get("T2M"), dict) and params_data["T2M"].get("data"):
                                temp_data = params_data["T2M"]["data"]
                                if isinstance(temp_data, list) and len(temp_data) > 0:
                                    temp = temp_data[-1]
                                    
                            if isinstance(params_data.get("RH2M"), dict) and params_data["RH2M"].get("data"):
                                humidity_data = params_data["RH2M"]["data"]
                                if isinstance(humidity_data, list) and len(humidity_data) > 0:
                                    humidity = humidity_data[-1]
                                    
                            if isinstance(params_data.get("WS2M"), dict) and params_data["WS2M"].get("data"):
                                wind_data = params_data["WS2M"]["data"]
                                if isinstance(wind_data, list) and len(wind_data) > 0:
                                    wind_speed = wind_data[-1]
                                    
                            if isinstance(params_data.get("PRECTOTCORR"), dict) and params_data["PRECTOTCORR"].get("data"):
                                precip_data = params_data["PRECTOTCORR"]["data"]
                                if isinstance(precip_data, list) and len(precip_data) > 0:
                                    precipitation = precip_data[-1]
                            
                            # NASA doesn't provide direct AQI, but we can use weather data
                            # Return weather-enhanced data
                            return {
                                "timestamp": datetime.now().isoformat(),
                                "latitude": latitude,
                                "longitude": longitude,
                                "temperature": float(temp),
                                "humidity": float(humidity),
                                "wind_speed": float(wind_speed),
                                "precipitation": float(precipitation),
                                "pressure": 1013.25,  # Default
                                "source_api": "NASA_POWER",
                                "data_quality": "high"
                            }
                        else:
                            logger.info("NASA API returned no relevant weather data for location")
                            return None
                    else:
                        logger.info(f"NASA API returned status {response.status}, no data available")
                        return None
            
        except aiohttp.ClientError as e:
            logger.warning(f"NASA API network error: {e}")
            return None
        except Exception as e:
            logger.warning(f"NASA API fetch failed: {e}")
            return None
    
    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize NASA data to standard format."""
        # NASA data is already in a simplified format, so we just return it
        if not isinstance(raw_data, dict):
            return {}
        return raw_data
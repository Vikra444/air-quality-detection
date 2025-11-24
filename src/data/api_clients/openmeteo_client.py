import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime

from .base_client import BaseAPIClient
from ...utils.logger import get_logger

logger = get_logger("api_clients.openmeteo")


class OpenMeteoClient(BaseAPIClient):
    """Open-Meteo API client for weather and air quality data."""
    
    def __init__(self):
        """Initialize Open-Meteo client."""
        super().__init__()
        self.base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    async def fetch_air_quality(
        self,
        latitude: float,
        longitude: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch air quality data from Open-Meteo API.
        
        No API key required - completely free!
        """
        try:
            url = f"{self.base_url}"
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "pm2_5,pm10,nitrogen_dioxide,carbon_monoxide,ozone,sulphur_dioxide",
                "timezone": "auto",
                "forecast_days": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "hourly" in data:
                            hourly = data["hourly"]
                            
                            # Get current hour's data (latest)
                            pm25 = hourly.get("pm2_5", [0])[-1] if hourly.get("pm2_5") else 0
                            pm10 = hourly.get("pm10", [0])[-1] if hourly.get("pm10") else 0
                            no2 = hourly.get("nitrogen_dioxide", [0])[-1] if hourly.get("nitrogen_dioxide") else 0
                            co = hourly.get("carbon_monoxide", [0])[-1] if hourly.get("carbon_monoxide") else 0
                            o3 = hourly.get("ozone", [0])[-1] if hourly.get("ozone") else 0
                            so2 = hourly.get("sulphur_dioxide", [0])[-1] if hourly.get("sulphur_dioxide") else 0
                            
                            # Calculate AQI (simplified - use highest pollutant)
                            aqi = max(
                                self._pm25_to_aqi(pm25),
                                self._pm10_to_aqi(pm10),
                                self._no2_to_aqi(no2),
                                self._o3_to_aqi(o3),
                                self._so2_to_aqi(so2)
                            )
                            
                            return {
                                "timestamp": datetime.now().isoformat(),
                                "latitude": latitude,
                                "longitude": longitude,
                                "pm25": float(pm25) if pm25 else 0,
                                "pm10": float(pm10) if pm10 else 0,
                                "no2": float(no2) if no2 else 0,
                                "co": float(co) if co else 0,
                                "o3": float(o3) if o3 else 0,
                                "so2": float(so2) if so2 else 0,
                                "aqi": float(aqi),
                                "source_api": "OpenMeteo",
                                "data_quality": "high"
                            }
            
            logger.warning("Open-Meteo API returned no data")
            return None
        
        except Exception as e:
            logger.warning(f"Open-Meteo API fetch failed: {e}")
            return None
    
    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Open-Meteo data to standard format."""
        # Open-Meteo data is already in a simplified format, so we just return it
        return raw_data
    
    def _pm25_to_aqi(self, pm25: float) -> float:
        """Convert PM2.5 to AQI."""
        if pm25 <= 12:
            return (pm25 / 12) * 50
        elif pm25 <= 35.4:
            return 50 + ((pm25 - 12) / (35.4 - 12)) * 50
        elif pm25 <= 55.4:
            return 100 + ((pm25 - 35.4) / (55.4 - 35.4)) * 50
        elif pm25 <= 150.4:
            return 150 + ((pm25 - 55.4) / (150.4 - 55.4)) * 100
        elif pm25 <= 250.4:
            return 200 + ((pm25 - 150.4) / (250.4 - 150.4)) * 100
        else:
            return 300 + ((pm25 - 250.4) / (350.4 - 250.4)) * 200
    
    def _pm10_to_aqi(self, pm10: float) -> float:
        """Convert PM10 to AQI."""
        if pm10 <= 54:
            return (pm10 / 54) * 50
        elif pm10 <= 154:
            return 50 + ((pm10 - 54) / (154 - 54)) * 50
        elif pm10 <= 254:
            return 100 + ((pm10 - 154) / (254 - 154)) * 50
        elif pm10 <= 354:
            return 150 + ((pm10 - 254) / (354 - 254)) * 100
        elif pm10 <= 424:
            return 200 + ((pm10 - 354) / (424 - 354)) * 100
        else:
            return 300 + ((pm10 - 424) / (504 - 424)) * 200
    
    def _no2_to_aqi(self, no2: float) -> float:
        """Convert NO2 (ppb) to AQI."""
        # Convert ppb to AQI (simplified)
        return no2 * 2.0  # Rough conversion
    
    def _o3_to_aqi(self, o3: float) -> float:
        """Convert O3 (ppb) to AQI."""
        return o3 * 1.2  # Rough conversion
    
    def _so2_to_aqi(self, so2: float) -> float:
        """Convert SO2 (ppb) to AQI."""
        return so2 * 1.5  # Rough conversion
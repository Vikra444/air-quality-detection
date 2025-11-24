"""
OpenWeatherMap Air Pollution API client.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from .base_client import BaseAPIClient
from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger("api_client.openweather")


class OpenWeatherClient(BaseAPIClient):
    """Client for OpenWeatherMap Air Pollution API."""
    
    def __init__(self):
        super().__init__(
            api_key=settings.openweather_api_key,
            base_url=settings.openweather_base_url,
            timeout=30
        )
        self.min_request_interval = 0.1  # OpenWeather allows faster requests
    
    async def fetch_air_quality(
        self,
        latitude: float,
        longitude: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Fetch air quality data from OpenWeatherMap."""
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not configured")
            return None
        
        try:
            # Fetch air pollution data
            pollution_data = await self._make_request(
                method="GET",
                endpoint="/air_pollution",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "appid": self.api_key
                }
            )
            
            # Fetch weather data for additional context
            weather_data = await self._make_request(
                method="GET",
                endpoint="/weather",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "appid": self.api_key,
                    "units": "metric"
                }
            )
            
            # Combine and normalize
            combined_data = {
                "pollution": pollution_data,
                "weather": weather_data
            }
            
            return self.normalize_data(combined_data)
        
        except Exception as e:
            logger.error(f"Error fetching OpenWeatherMap data: {e}", error=str(e))
            return None
    
    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OpenWeatherMap response to standard format."""
        try:
            pollution = raw_data["pollution"]["list"][0]
            components = pollution["components"]
            main = pollution["main"]
            weather = raw_data["weather"]
            
            # Calculate AQI based on actual pollutant values instead of fixed mapping
            # Using EPA AQI calculation method
            aqi = self._calculate_aqi_from_components(components)
            
            normalized = {
                "timestamp": datetime.fromtimestamp(pollution["dt"]).isoformat(),
                "latitude": weather["coord"]["lat"],
                "longitude": weather["coord"]["lon"],
                "pm25": components.get("pm2_5", 0),
                "pm10": components.get("pm10", 0),
                "no2": components.get("no2", 0),
                "co": components.get("co", 0) / 1000,  # Convert from μg/m³ to ppm
                "o3": components.get("o3", 0),
                "so2": components.get("so2", 0),
                "aqi": aqi,
                "primary_pollutant": self._get_primary_pollutant(components),
                "temperature": weather["main"]["temp"],
                "humidity": weather["main"]["humidity"],
                "pressure": weather["main"]["pressure"],
                "wind_speed": weather.get("wind", {}).get("speed", 0),
                "wind_direction": weather.get("wind", {}).get("deg", 0),
                "source": "openweather",
                "quality_score": 0.9
            }
            
            return normalized
        
        except Exception as e:
            logger.error(f"Error normalizing OpenWeatherMap data: {e}", error=str(e))
            raise
    
    def _calculate_aqi_from_components(self, components: Dict[str, float]) -> float:
        """Calculate AQI based on actual pollutant values using EPA breakpoints."""
        # AQI calculation based on EPA standards
        pollutants = {
            "PM2.5": components.get("pm2_5", 0),
            "PM10": components.get("pm10", 0),
            "O3": components.get("o3", 0),
            "NO2": components.get("no2", 0),
            "SO2": components.get("so2", 0),
            "CO": components.get("co", 0) / 1000  # Convert μg/m³ to ppm
        }
        
        # Calculate AQI for each pollutant
        aqi_values = []
        for pollutant, value in pollutants.items():
            aqi = self._get_aqi_for_pollutant(pollutant, value)
            if aqi > 0:
                aqi_values.append(aqi)
        
        # Return the highest AQI value (worst pollutant)
        return max(aqi_values) if aqi_values else 0
    
    def _get_aqi_for_pollutant(self, pollutant: str, concentration: float) -> float:
        """Calculate AQI for a specific pollutant based on concentration."""
        # EPA AQI breakpoints
        breakpoints = {
            "PM2.5": [
                (0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500)
            ],
            "PM10": [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 504, 301, 400),
                (505, 604, 401, 500)
            ],
            "O3": [
                (0, 54, 0, 50),
                (55, 70, 51, 100),
                (71, 85, 101, 150),
                (86, 105, 151, 200),
                (106, 200, 201, 300)
            ],
            "NO2": [
                (0, 53, 0, 50),
                (54, 100, 51, 100),
                (101, 360, 101, 150),
                (361, 649, 151, 200),
                (650, 1249, 201, 300),
                (1250, 1649, 301, 400),
                (1650, 2049, 401, 500)
            ],
            "SO2": [
                (0, 35, 0, 50),
                (36, 75, 51, 100),
                (76, 185, 101, 150),
                (186, 304, 151, 200),
                (305, 604, 201, 300),
                (605, 804, 301, 400),
                (805, 1004, 401, 500)
            ],
            "CO": [
                (0, 4.4, 0, 50),
                (4.5, 9.4, 51, 100),
                (9.5, 12.4, 101, 150),
                (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300),
                (30.5, 40.4, 301, 400),
                (40.5, 50.4, 401, 500)
            ]
        }
        
        if pollutant not in breakpoints:
            return 0
        
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints[pollutant]:
            if bp_low <= concentration <= bp_high:
                # Linear interpolation
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
                return round(aqi, 1)
        
        # If concentration is above the highest breakpoint, return 500
        return 500 if concentration > 0 else 0
    
    def _get_primary_pollutant(self, components: Dict[str, float]) -> str:
        """Determine primary pollutant based on component values."""
        # Calculate AQI for each pollutant to determine primary pollutant
        pollutant_aqi = {}
        for pollutant, value in components.items():
            if pollutant in ["pm2_5", "pm10", "no2", "co", "o3", "so2"]:
                # Convert key names to standard format
                std_name = pollutant.upper().replace("_", ".")
                if pollutant == "co":
                    std_name = "CO"
                    value = value / 1000  # Convert to ppm
                pollutant_aqi[std_name] = self._get_aqi_for_pollutant(std_name, value)
        
        if not pollutant_aqi:
            return "PM2.5"
        
        return max(pollutant_aqi.items(), key=lambda x: x[1])[0]

"""
Demo data generator for fallback when APIs are unavailable.
"""

from datetime import datetime
from typing import Dict, Any
import random

def generate_demo_air_quality_data(
    latitude: float,
    longitude: float,
    location_id: str = "Demo"
) -> Dict[str, Any]:
    """
    Generate realistic demo air quality data.
    Used as fallback when real APIs are unavailable.
    """
    # Generate realistic values based on location
    base_aqi = 50 + random.randint(-10, 40)
    
    # Urban areas tend to have higher pollution
    if "urban" in location_id.lower() or "delhi" in location_id.lower():
        base_aqi += 20
    
    # Ensure AQI is within valid range
    aqi = max(0, min(500, base_aqi))
    
    # Generate pollutant values based on AQI
    pm25 = aqi * 0.4 + random.uniform(-5, 5)
    pm10 = aqi * 0.5 + random.uniform(-5, 5)
    no2 = aqi * 0.3 + random.uniform(-3, 3)
    co = aqi * 0.08 + random.uniform(-0.5, 0.5)
    o3 = aqi * 0.35 + random.uniform(-5, 5)
    so2 = aqi * 0.12 + random.uniform(-2, 2)
    
    # Determine primary pollutant
    pollutants = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "O3": o3
    }
    primary_pollutant = max(pollutants.items(), key=lambda x: x[1])[0]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "location_id": location_id,
        "latitude": latitude,
        "longitude": longitude,
        "pm25": max(0, pm25),
        "pm10": max(0, pm10),
        "no2": max(0, no2),
        "co": max(0, co),
        "o3": max(0, o3),
        "so2": max(0, so2),
        "aqi": aqi,
        "primary_pollutant": primary_pollutant,
        "temperature": 20 + random.uniform(-5, 10),
        "humidity": 50 + random.uniform(-20, 20),
        "wind_speed": 3 + random.uniform(-2, 5),
        "wind_direction": random.uniform(0, 360),
        "pressure": 1013 + random.uniform(-10, 10),
        "source": "demo",
        "quality_score": 0.7,
        "confidence_score": 0.6,
        "source_api": "demo_fallback"
    }


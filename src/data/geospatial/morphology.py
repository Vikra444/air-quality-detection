"""
Simple geospatial morphology analyzer for urban vs peri-urban classification.
"""

from typing import Dict, Any, Optional
from enum import Enum

from ...utils.logger import get_logger

logger = get_logger("geospatial.morphology")


class AreaType(str, Enum):
    """Area type classification."""
    URBAN = "urban"
    PERI_URBAN = "peri_urban"
    RURAL = "rural"
    UNKNOWN = "unknown"


class GeospatialMorphologyAnalyzer:
    """Simple analyzer for urban vs peri-urban classification."""
    
    def __init__(self):
        """Initialize morphology analyzer."""
        # Simple heuristics based on coordinates
        # In a full implementation, this would use OpenStreetMap or satellite data
        self.urban_centers = {
            # Major Indian cities (latitude, longitude, radius_km)
            "delhi": (28.6139, 77.2090, 30),
            "mumbai": (19.0760, 72.8777, 25),
            "bangalore": (12.9716, 77.5946, 20),
            "chennai": (13.0827, 80.2707, 20),
            "kolkata": (22.5726, 88.3639, 20),
            "hyderabad": (17.3850, 78.4867, 20),
            "pune": (18.5204, 73.8567, 15),
            "ahmedabad": (23.0225, 72.5714, 15),
            "jaipur": (26.9124, 75.7873, 15),
            "lucknow": (26.8467, 80.9462, 15)
        }
    
    def classify_area(
        self,
        latitude: float,
        longitude: float,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify area as urban, peri-urban, or rural.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            additional_data: Optional additional data (population density, etc.)
        
        Returns:
            Dictionary with area classification
        """
        try:
            # Simple distance-based classification
            area_type = AreaType.UNKNOWN
            confidence = 0.5
            nearest_city = None
            distance_km = None
            
            # Check distance from major urban centers
            min_distance = float('inf')
            for city_name, (city_lat, city_lon, radius) in self.urban_centers.items():
                distance = self._haversine_distance(
                    latitude, longitude,
                    city_lat, city_lon
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_city = city_name
                    distance_km = distance
                    
                    # Classify based on distance
                    if distance <= radius * 0.5:  # Within core urban area
                        area_type = AreaType.URBAN
                        confidence = 0.8
                    elif distance <= radius:  # Within urban boundary
                        area_type = AreaType.URBAN
                        confidence = 0.7
                    elif distance <= radius * 1.5:  # Peri-urban
                        area_type = AreaType.PERI_URBAN
                        confidence = 0.7
                    elif distance <= radius * 3:  # Still peri-urban
                        area_type = AreaType.PERI_URBAN
                        confidence = 0.6
                    else:  # Rural
                        area_type = AreaType.RURAL
                        confidence = 0.6
            
            # Use additional data if available
            if additional_data:
                population_density = additional_data.get("population_density")
                if population_density:
                    if population_density > 1000:  # per km²
                        area_type = AreaType.URBAN
                        confidence = 0.9
                    elif population_density > 200:
                        area_type = AreaType.PERI_URBAN
                        confidence = 0.8
                    else:
                        area_type = AreaType.RURAL
                        confidence = 0.7
            
            # Generate explanation
            explanation = self._generate_explanation(area_type, nearest_city, distance_km)
            
            return {
                "area_type": area_type.value,
                "confidence": confidence,
                "nearest_city": nearest_city,
                "distance_km": distance_km,
                "explanation": explanation,
                "characteristics": self._get_characteristics(area_type)
            }
        
        except Exception as e:
            logger.error(f"Error classifying area: {e}", error=str(e))
            return {
                "area_type": AreaType.UNKNOWN.value,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate distance between two points using Haversine formula."""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c
    
    def _generate_explanation(
        self,
        area_type: AreaType,
        nearest_city: Optional[str],
        distance_km: Optional[float]
    ) -> str:
        """Generate explanation for area classification."""
        if area_type == AreaType.URBAN:
            return f"Classified as urban area. "
        elif area_type == AreaType.PERI_URBAN:
            return f"Classified as peri-urban area (transition zone between urban and rural). "
        elif area_type == AreaType.RURAL:
            return f"Classified as rural area. "
        else:
            return "Unable to classify area type with high confidence."
        
        if nearest_city and distance_km:
            return f"{explanation}Nearest major city: {nearest_city.title()} ({distance_km:.1f} km away)."
        
        return explanation
    
    def _get_characteristics(self, area_type: AreaType) -> Dict[str, Any]:
        """Get characteristics of area type."""
        characteristics = {
            AreaType.URBAN: {
                "population_density": "high",
                "building_density": "high",
                "traffic_density": "high",
                "green_cover": "low",
                "typical_sources": ["traffic", "industrial", "construction"]
            },
            AreaType.PERI_URBAN: {
                "population_density": "medium",
                "building_density": "medium",
                "traffic_density": "medium",
                "green_cover": "medium",
                "typical_sources": ["traffic", "crop_burning", "construction"]
            },
            AreaType.RURAL: {
                "population_density": "low",
                "building_density": "low",
                "traffic_density": "low",
                "green_cover": "high",
                "typical_sources": ["crop_burning", "weather"]
            }
        }
        
        return characteristics.get(area_type, {})


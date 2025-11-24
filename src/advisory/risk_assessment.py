"""
Health risk assessment for air quality.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from ..data.models import HealthRiskLevel, AirQualityData
from ..config.settings import HEALTH_RISK_CATEGORIES
from ..utils.logger import get_logger

logger = get_logger("advisory.risk_assessment")


class RiskAssessment:
    """Health risk assessment engine."""
    
    def __init__(self):
        self.risk_categories = HEALTH_RISK_CATEGORIES
    
    def assess_risk(
        self,
        aqi: float,
        pollutant_data: Dict[str, float],
        area_type: Optional[str] = None
    ) -> tuple:
        """
        Assess health risk from air quality data.
        
        Returns:
            Tuple of (risk_level, assessment_details)
        """
        # Determine risk level from AQI
        risk_level = self._determine_risk_level(aqi)
        
        # Analyze pollutants
        primary_concerns = self._identify_primary_concerns(pollutant_data)
        pollutant_analysis = self._analyze_pollutants(pollutant_data)
        
        # Calculate confidence
        confidence = self._calculate_confidence(pollutant_data, aqi)
        
        assessment = {
            "aqi": aqi,
            "risk_level": risk_level,
            "primary_concerns": primary_concerns,
            "pollutant_analysis": pollutant_analysis,
            "confidence": confidence,
            "area_type": area_type,
            "timestamp": datetime.now().isoformat()
        }
        
        return risk_level, assessment
    
    def _determine_risk_level(self, aqi: float) -> HealthRiskLevel:
        """Determine risk level from AQI."""
        if aqi <= 50:
            return HealthRiskLevel.GOOD
        elif aqi <= 100:
            return HealthRiskLevel.MODERATE
        elif aqi <= 150:
            return HealthRiskLevel.UNHEALTHY_FOR_SENSITIVE
        elif aqi <= 200:
            return HealthRiskLevel.UNHEALTHY
        elif aqi <= 300:
            return HealthRiskLevel.VERY_UNHEALTHY
        else:
            return HealthRiskLevel.HAZARDOUS
    
    def _identify_primary_concerns(self, pollutant_data: Dict[str, float]) -> List[str]:
        """Identify primary health concerns based on pollutants."""
        concerns = []
        
        # PM2.5 concerns
        pm25 = pollutant_data.get("PM2.5", 0)
        if pm25 > 35:  # WHO guideline
            concerns.append("High PM2.5 levels - can penetrate deep into lungs")
        if pm25 > 55:
            concerns.append("Very high PM2.5 - increased risk of respiratory and cardiovascular diseases")
        
        # PM10 concerns
        pm10 = pollutant_data.get("PM10", 0)
        if pm10 > 50:  # WHO guideline
            concerns.append("Elevated PM10 - can irritate eyes and respiratory system")
        
        # NO2 concerns
        no2 = pollutant_data.get("NO2", 0)
        if no2 > 200:  # ppb
            concerns.append("High NO2 - increased risk of asthma attacks and respiratory inflammation")
        
        # O3 concerns
        o3 = pollutant_data.get("O3", 0)
        if o3 > 100:  # ppb
            concerns.append("Elevated O3 - can cause lung irritation and reduced lung function")
        
        # CO concerns
        co = pollutant_data.get("CO", 0)
        if co > 9:  # ppm
            concerns.append("High CO - can reduce oxygen delivery to organs")
        
        if not concerns:
            concerns.append("Air quality is within acceptable limits")
        
        return concerns
    
    def _analyze_pollutants(self, pollutant_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze individual pollutants."""
        analysis = {}
        
        # WHO guidelines
        guidelines = {
            "PM2.5": 15,  # μg/m³ annual mean
            "PM10": 45,   # μg/m³ annual mean
            "NO2": 25,    # μg/m³ annual mean (approx 13 ppb)
            "O3": 100,    # μg/m³ 8-hour mean (approx 50 ppb)
            "CO": 10,     # mg/m³ 24-hour mean (approx 9 ppm)
            "SO2": 40     # μg/m³ 24-hour mean
        }
        
        for pollutant, value in pollutant_data.items():
            guideline = guidelines.get(pollutant, 0)
            if guideline > 0:
                ratio = value / guideline
                status = "safe" if ratio < 1 else "moderate" if ratio < 2 else "high"
                
                analysis[pollutant] = {
                    "value": value,
                    "guideline": guideline,
                    "ratio": ratio,
                    "status": status,
                    "exceedance": ratio > 1
                }
        
        return analysis
    
    def _calculate_confidence(
        self,
        pollutant_data: Dict[str, float],
        aqi: float
    ) -> float:
        """Calculate confidence in risk assessment."""
        # Base confidence
        confidence = 0.8
        
        # Increase confidence if we have multiple pollutants
        pollutant_count = len([v for v in pollutant_data.values() if v > 0])
        if pollutant_count >= 4:
            confidence += 0.1
        elif pollutant_count >= 2:
            confidence += 0.05
        
        # Decrease confidence if AQI seems inconsistent with pollutants
        max_pollutant = max(pollutant_data.values()) if pollutant_data else 0
        if max_pollutant > 0 and aqi < max_pollutant * 2:
            confidence -= 0.1
        
        return min(1.0, max(0.5, confidence))


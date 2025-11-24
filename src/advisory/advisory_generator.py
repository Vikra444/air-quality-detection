"""
AI-driven health advisory generator.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from ..data.models import HealthRiskLevel, HealthAdvisory
from ..config.settings import HEALTH_RISK_CATEGORIES
from .risk_assessment import RiskAssessment
from .vulnerable_groups import VulnerableGroupAdvisor
from ..utils.logger import get_logger

logger = get_logger("advisory.generator")


class AdvisoryGenerator:
    """Generate health advisories based on risk assessment."""
    
    def __init__(self):
        self.risk_assessor = RiskAssessment()
        self.vulnerable_advisor = VulnerableGroupAdvisor()
        self.risk_categories = HEALTH_RISK_CATEGORIES
    
    def generate_advisory(
        self,
        location_id: str,
        aqi: float,
        pollutant_data: Dict[str, float],
        vulnerable_groups: Optional[List[str]] = None,
        area_type: Optional[str] = None
    ) -> HealthAdvisory:
        """Generate comprehensive health advisory."""
        # Assess risk
        risk_level, assessment = self.risk_assessor.assess_risk(
            aqi, pollutant_data, area_type
        )
        
        # Get base recommendations
        recommendations = self._get_base_recommendations(risk_level)
        
        # Get preventive measures
        preventive_measures = self._get_preventive_measures(risk_level, pollutant_data)
        
        # Get vulnerable group advisories
        vulnerable_groups = vulnerable_groups or []
        vulnerable_advisories = {}
        for group in vulnerable_groups:
            vulnerable_advisories[group] = self.vulnerable_advisor.get_advisory(
                group, risk_level, pollutant_data
            )
        
        # Generate contextual advisory text
        advisory_text = self._generate_contextual_text(
            risk_level, assessment, pollutant_data, area_type
        )
        
        return HealthAdvisory(
            location_id=location_id,
            timestamp=datetime.now(),
            risk_level=risk_level,
            aqi=aqi,
            primary_concerns=assessment["primary_concerns"],
            recommendations=recommendations,
            vulnerable_group_advisories=vulnerable_advisories,
            preventive_measures=preventive_measures,
            confidence=assessment["confidence"]
        )
    
    def _get_base_recommendations(self, risk_level: HealthRiskLevel) -> List[str]:
        """Get base recommendations for risk level."""
        category = self.risk_categories.get(risk_level.value, {})
        return category.get("recommendations", [])
    
    def _get_preventive_measures(
        self,
        risk_level: HealthRiskLevel,
        pollutant_data: Dict[str, float]
    ) -> List[str]:
        """Get preventive measures based on risk and pollutants."""
        measures = []
        
        if risk_level in [HealthRiskLevel.UNHEALTHY, HealthRiskLevel.VERY_UNHEALTHY, HealthRiskLevel.HAZARDOUS]:
            measures.append("Stay indoors as much as possible")
            measures.append("Use air purifiers with HEPA filters")
            measures.append("Keep windows and doors closed")
            measures.append("Avoid outdoor exercise")
        
        if risk_level == HealthRiskLevel.VERY_UNHEALTHY or risk_level == HealthRiskLevel.HAZARDOUS:
            measures.append("Wear N95 masks if going outside is necessary")
            measures.append("Consider relocating to area with better air quality if possible")
        
        # Pollutant-specific measures
        if pollutant_data.get("PM2.5", 0) > 35:
            measures.append("PM2.5 is high - avoid outdoor activities")
        
        if pollutant_data.get("O3", 0) > 100:
            measures.append("Ozone levels elevated - avoid outdoor activities during afternoon")
        
        if not measures:
            measures.append("Continue normal activities")
            measures.append("Monitor air quality regularly")
        
        return measures
    
    def _generate_contextual_text(
        self,
        risk_level: HealthRiskLevel,
        assessment: Dict[str, Any],
        pollutant_data: Dict[str, float],
        area_type: Optional[str]
    ) -> str:
        """Generate contextual advisory text."""
        primary_pollutant = max(pollutant_data.items(), key=lambda x: x[1])[0] if pollutant_data else "PM2.5"
        aqi = assessment["aqi"]
        
        text = f"Air quality index is {aqi:.1f}, classified as {risk_level.value}. "
        text += f"Primary pollutant is {primary_pollutant}. "
        
        if area_type:
            text += f"Area type: {area_type}. "
        
        # Add time-based context
        hour = datetime.now().hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            text += "Rush hour conditions may worsen air quality. "
        
        # Add weather context if available
        if assessment.get("wind_speed"):
            wind_speed = assessment["wind_speed"]
            if wind_speed < 2:
                text += "Low wind speed may cause pollutant accumulation. "
            elif wind_speed > 5:
                text += "Higher wind speed may help disperse pollutants. "
        
        return text


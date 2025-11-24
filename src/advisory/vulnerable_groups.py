"""
Vulnerable group-specific health advisories.
"""

from typing import Dict, List
from ..data.models import HealthRiskLevel
from ..utils.logger import get_logger

logger = get_logger("advisory.vulnerable_groups")


class VulnerableGroupAdvisor:
    """Generate advisories for vulnerable population groups."""
    
    def __init__(self):
        self.group_advisories = {
            "children": {
                "risk_multiplier": 1.5,
                "concerns": [
                    "Developing respiratory systems are more vulnerable",
                    "Higher breathing rates increase exposure",
                    "More time spent outdoors increases risk"
                ],
                "recommendations": {
                    HealthRiskLevel.GOOD: [
                        "Normal outdoor activities are safe",
                        "Continue regular exercise"
                    ],
                    HealthRiskLevel.MODERATE: [
                        "Limit prolonged outdoor activities",
                        "Monitor for any respiratory symptoms"
                    ],
                    HealthRiskLevel.UNHEALTHY_FOR_SENSITIVE: [
                        "Limit outdoor activities",
                        "Avoid outdoor sports and games",
                        "Ensure proper indoor air quality"
                    ],
                    HealthRiskLevel.UNHEALTHY: [
                        "Avoid all outdoor activities",
                        "Keep children indoors",
                        "Use air purifiers in bedrooms"
                    ],
                    HealthRiskLevel.VERY_UNHEALTHY: [
                        "Stay indoors at all times",
                        "Cancel outdoor school activities",
                        "Use HEPA air purifiers"
                    ],
                    HealthRiskLevel.HAZARDOUS: [
                        "Emergency conditions - keep children indoors",
                        "Consider school closure if possible",
                        "Use N95 masks if going outside is unavoidable"
                    ]
                }
            },
            "elderly": {
                "risk_multiplier": 1.4,
                "concerns": [
                    "Reduced lung capacity",
                    "Weakened immune system",
                    "Pre-existing health conditions"
                ],
                "recommendations": {
                    HealthRiskLevel.GOOD: [
                        "Normal activities are safe",
                        "Continue regular exercise"
                    ],
                    HealthRiskLevel.MODERATE: [
                        "Limit prolonged outdoor exertion",
                        "Monitor existing health conditions"
                    ],
                    HealthRiskLevel.UNHEALTHY_FOR_SENSITIVE: [
                        "Avoid outdoor activities",
                        "Stay indoors with air conditioning",
                        "Have medications readily available"
                    ],
                    HealthRiskLevel.UNHEALTHY: [
                        "Stay indoors",
                        "Use air purifiers",
                        "Monitor health closely"
                    ],
                    HealthRiskLevel.VERY_UNHEALTHY: [
                        "Avoid all outdoor exposure",
                        "Use HEPA air purifiers",
                        "Have emergency contacts ready"
                    ],
                    HealthRiskLevel.HAZARDOUS: [
                        "Emergency conditions - stay indoors",
                        "Use air purifiers continuously",
                        "Seek medical attention if symptoms worsen"
                    ]
                }
            },
            "asthma_patients": {
                "risk_multiplier": 2.0,
                "concerns": [
                    "Airway inflammation",
                    "Increased asthma attacks",
                    "Medication effectiveness reduced"
                ],
                "recommendations": {
                    HealthRiskLevel.GOOD: [
                        "Normal activities with usual precautions",
                        "Carry rescue inhaler"
                    ],
                    HealthRiskLevel.MODERATE: [
                        "Be cautious with outdoor activities",
                        "Carry rescue inhaler at all times",
                        "Monitor peak flow regularly"
                    ],
                    HealthRiskLevel.UNHEALTHY_FOR_SENSITIVE: [
                        "Limit outdoor activities",
                        "Use air purifiers at home",
                        "Increase medication if advised by doctor"
                    ],
                    HealthRiskLevel.UNHEALTHY: [
                        "Avoid outdoor activities",
                        "Stay indoors with air purifiers",
                        "Have rescue inhaler readily available"
                    ],
                    HealthRiskLevel.VERY_UNHEALTHY: [
                        "Stay indoors at all times",
                        "Use HEPA air purifiers",
                        "Consider increasing medication",
                        "Contact doctor if symptoms worsen"
                    ],
                    HealthRiskLevel.HAZARDOUS: [
                        "Emergency conditions - stay indoors",
                        "Use air purifiers continuously",
                        "Seek immediate medical attention if needed",
                        "Have emergency medications ready"
                    ]
                }
            }
        }
    
    def get_advisory(
        self,
        group: str,
        risk_level: HealthRiskLevel,
        pollutant_data: Dict[str, float]
    ) -> List[str]:
        """Get advisory for a specific vulnerable group."""
        if group not in self.group_advisories:
            logger.warning(f"Unknown vulnerable group: {group}")
            return ["Consult healthcare provider for personalized advice"]
        
        group_info = self.group_advisories[group]
        recommendations = group_info["recommendations"].get(
            risk_level,
            ["Consult healthcare provider for personalized advice"]
        )
        
        return recommendations
    
    def get_risk_multiplier(self, group: str) -> float:
        """Get risk multiplier for a vulnerable group."""
        return self.group_advisories.get(group, {}).get("risk_multiplier", 1.0)


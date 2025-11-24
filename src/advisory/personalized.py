"""
Personalized health advisory generation based on user profile.
"""

from typing import Dict, List, Optional
from ..data.models import HealthRiskLevel, User


def get_risk_level_from_aqi(aqi: float) -> HealthRiskLevel:
    """Convert AQI value to health risk level."""
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


def generate_personalized_advisory(user: User, aqi: float) -> Dict[str, any]:
    """
    Generate personalized health advisory based on user profile and AQI.
    
    Args:
        user: User object with profile information
        aqi: Air Quality Index value
        
    Returns:
        Dictionary with risk level, message, and recommendations
    """
    risk_level = get_risk_level_from_aqi(aqi)
    
    # Base message based on AQI
    base_messages = {
        HealthRiskLevel.GOOD: f"AQI {int(aqi)} - Good. Air quality is satisfactory.",
        HealthRiskLevel.MODERATE: f"AQI {int(aqi)} - Moderate. Air quality is acceptable.",
        HealthRiskLevel.UNHEALTHY_FOR_SENSITIVE: f"AQI {int(aqi)} - Unhealthy for sensitive groups.",
        HealthRiskLevel.UNHEALTHY: f"AQI {int(aqi)} - Unhealthy. Everyone may begin to experience health effects.",
        HealthRiskLevel.VERY_UNHEALTHY: f"AQI {int(aqi)} - Very Unhealthy. Health alert.",
        HealthRiskLevel.HAZARDOUS: f"AQI {int(aqi)} - Hazardous. Health warnings of emergency conditions."
    }
    
    # Personalized factors
    factors = []
    recommendations = []
    
    # Age-based considerations
    if user.age < 18:
        factors.append("children")
        recommendations.extend([
            "Children should limit prolonged outdoor exertion",
            "Keep children indoors when possible"
        ])
    elif user.age > 65:
        factors.append("elderly")
        recommendations.extend([
            "Elderly individuals should reduce outdoor activities",
            "Monitor for respiratory symptoms"
        ])
    
    # Chronic conditions
    chronic_conditions = user.chronic_conditions or ""
    if "asthma" in chronic_conditions.lower():
        factors.append("asthma")
        recommendations.extend([
            "Asthma patients should keep inhalers handy",
            "Avoid outdoor activities during high pollution periods"
        ])
    
    if "heart" in chronic_conditions.lower() or "cardiac" in chronic_conditions.lower():
        factors.append("heart condition")
        recommendations.extend([
            "Individuals with heart conditions should limit outdoor exertion",
            "Monitor for chest pain or breathing difficulties"
        ])
    
    if "diabetes" in chronic_conditions.lower():
        factors.append("diabetes")
        recommendations.extend([
            "Diabetic patients should monitor blood sugar more frequently",
            "Stay hydrated and limit outdoor activities"
        ])
    
    if "lung" in chronic_conditions.lower() or "copd" in chronic_conditions.lower():
        factors.append("lung disease")
        recommendations.extend([
            "Individuals with lung disease should avoid outdoor activities",
            "Use prescribed medications as directed"
        ])
    
    # Lifestyle factors
    if user.is_smoker:
        factors.append("smoker")
        recommendations.extend([
            "Avoid smoking and secondhand smoke",
            "Consider indoor air purifiers"
        ])
    
    if user.works_outdoors:
        factors.append("outdoor worker")
        recommendations.extend([
            "Wear N95 or similar mask when working outdoors",
            "Take more frequent breaks in clean air environments",
            "Stay hydrated"
        ])
    
    # Generate personalized message
    base_message = base_messages.get(risk_level, f"AQI {int(aqi)}")
    
    if factors:
        factor_list = ", ".join(factors)
        personalized_message = f"{base_message} Based on your profile ({factor_list}), you are at higher risk."
    else:
        personalized_message = base_message
    
    # Add general recommendations based on risk level
    general_recommendations = {
        HealthRiskLevel.GOOD: [
            "Enjoy outdoor activities",
            "General population can continue normal activities"
        ],
        HealthRiskLevel.MODERATE: [
            "Unusually sensitive people should consider limiting prolonged outdoor exertion",
            "Normal activities are generally safe"
        ],
        HealthRiskLevel.UNHEALTHY_FOR_SENSITIVE: [
            "People with heart or lung disease, older adults, children, and teens should reduce prolonged outdoor exertion",
            "Everyone else should limit prolonged outdoor exertion"
        ],
        HealthRiskLevel.UNHEALTHY: [
            "People with heart or lung disease, older adults, children, and teens should avoid prolonged outdoor exertion",
            "Everyone else should reduce prolonged outdoor exertion"
        ],
        HealthRiskLevel.VERY_UNHEALTHY: [
            "People with heart or lung disease, older adults, children, and teens should avoid all outdoor exertion",
            "Everyone else should reduce outdoor exertion"
        ],
        HealthRiskLevel.HAZARDOUS: [
            "Everyone should avoid all outdoor exertion",
            "Stay indoors and keep windows closed",
            "Use air purifiers if available"
        ]
    }
    
    # Combine personalized and general recommendations
    all_recommendations = recommendations + general_recommendations.get(risk_level, [])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for rec in all_recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    return {
        "risk_level": risk_level.value,
        "message": personalized_message,
        "recommendations": unique_recommendations,
        "factors": factors
    }


def get_aqi_category(aqi: float) -> str:
    """Get AQI category as a string."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"
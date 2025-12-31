"""
Configuration management for AirGuard system with enhanced settings.
"""

import os
from typing import Optional
from pathlib import Path
from ..utils.config_loader import load_env_file, get_env, get_env_bool, get_env_int, get_env_float

# Load environment variables
load_env_file()


class Settings:
    """Application settings with environment variable support."""
    
    # Database Configuration
    database_url: str = get_env("DATABASE_URL", "sqlite:///./airguard.db")
    redis_url: str = get_env("REDIS_URL", "redis://localhost:6379/0")
    
    # API Keys
    openweather_api_key: Optional[str] = get_env("OPENWEATHER_API_KEY")
    
    # API Base URLs
    openweather_base_url: str = get_env("OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
    
    # Model Configuration
    model_update_interval: int = get_env_int("MODEL_UPDATE_INTERVAL", 3600)
    prediction_horizon: int = get_env_int("PREDICTION_HORIZON", 24)
    confidence_threshold: float = get_env_float("CONFIDENCE_THRESHOLD", 0.8)
    
    # Application Configuration
    debug: bool = get_env_bool("DEBUG", False)
    log_level: str = get_env("LOG_LEVEL", "INFO")
    api_host: str = get_env("API_HOST", "0.0.0.0")
    api_port: int = get_env_int("API_PORT", 8001)
    dashboard_port: int = get_env_int("DASHBOARD_PORT", 8050)
    
    # Security
    secret_key: str = get_env("SECRET_KEY", "your-secret-key-change-in-production")
    jwt_secret_key: str = get_env("JWT_SECRET_KEY", "your-jwt-secret-key-change-in-production")
    algorithm: str = get_env("ALGORITHM", "HS256")
    access_token_expire_minutes: int = get_env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    
    # Scheduler Configuration
    data_fetch_interval: int = get_env_int("DATA_FETCH_INTERVAL", 600)  # 10 minutes
    
    # Cache Configuration
    cache_ttl: int = get_env_int("CACHE_TTL", 300)  # 5 minutes
    cache_max_size: int = get_env_int("CACHE_MAX_SIZE", 1000)
    
    # Rate Limiting
    api_rate_limit: int = get_env_int("API_RATE_LIMIT", 100)  # requests per minute
    api_rate_limit_window: int = get_env_int("API_RATE_LIMIT_WINDOW", 60)  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Health Risk Categories
HEALTH_RISK_CATEGORIES = {
    "GOOD": {
        "aqi_range": (0, 50),
        "description": "Air quality is satisfactory",
        "health_impact": "Little or no risk",
        "recommendations": [
            "Ideal air quality for outdoor activities",
            "No restrictions for sensitive groups"
        ],
        "color": "#00E400"
    },
    "MODERATE": {
        "aqi_range": (51, 100),
        "description": "Air quality is acceptable",
        "health_impact": "Moderate concern for unusually sensitive people",
        "recommendations": [
            "Unusually sensitive people should consider limiting prolonged outdoor exertion",
            "Children and elderly can enjoy outdoor activities"
        ],
        "color": "#FFFF00"
    },
    "UNHEALTHY_FOR_SENSITIVE": {
        "aqi_range": (101, 150),
        "description": "Unhealthy for sensitive groups",
        "health_impact": "Members of sensitive groups may experience health effects",
        "recommendations": [
            "Sensitive groups should limit prolonged outdoor exertion",
            "Children, elderly, and people with heart/lung disease should be cautious"
        ],
        "color": "#FF7E00"
    },
    "UNHEALTHY": {
        "aqi_range": (151, 200),
        "description": "Unhealthy",
        "health_impact": "Everyone may begin to experience health effects",
        "recommendations": [
            "Everyone should limit outdoor activities",
            "Sensitive groups should avoid outdoor exertion",
            "Consider wearing masks outdoors"
        ],
        "color": "#FF0000"
    },
    "VERY_UNHEALTHY": {
        "aqi_range": (201, 300),
        "description": "Very unhealthy",
        "health_impact": "Health warnings of emergency conditions",
        "recommendations": [
            "Avoid all outdoor activities",
            "Stay indoors with air purifiers if possible",
            "Wear N95 masks if going outside is necessary"
        ],
        "color": "#8F3F97"
    },
    "HAZARDOUS": {
        "aqi_range": (301, 500),
        "description": "Hazardous",
        "health_impact": "Health alert: everyone may experience more serious health effects",
        "recommendations": [
            "Stay indoors at all times",
            "Use air purifiers with HEPA filters",
            "Emergency conditions - avoid all outdoor exposure"
        ],
        "color": "#7E0023"
    }
}


# Pollutant Information
POLLUTANT_INFO = {
    "PM2.5": {
        "name": "Particulate Matter 2.5",
        "unit": "μg/m³",
        "description": "Fine particles that can penetrate deep into lungs",
        "health_effects": "Respiratory and cardiovascular diseases"
    },
    "PM10": {
        "name": "Particulate Matter 10",
        "unit": "μg/m³",
        "description": "Coarse particles that can irritate eyes and respiratory system",
        "health_effects": "Respiratory irritation, asthma exacerbation"
    },
    "NO2": {
        "name": "Nitrogen Dioxide",
        "unit": "ppb",
        "description": "Gas from vehicle emissions and industrial processes",
        "health_effects": "Respiratory inflammation, increased asthma attacks"
    },
    "CO": {
        "name": "Carbon Monoxide",
        "unit": "ppm",
        "description": "Colorless, odorless gas from incomplete combustion",
        "health_effects": "Reduced oxygen delivery, cardiovascular problems"
    },
    "O3": {
        "name": "Ozone",
        "unit": "ppb",
        "description": "Ground-level ozone formed by chemical reactions",
        "health_effects": "Lung irritation, reduced lung function"
    },
    "SO2": {
        "name": "Sulfur Dioxide",
        "unit": "ppb",
        "description": "Gas from burning fossil fuels",
        "health_effects": "Respiratory irritation, asthma exacerbation"
    }
}


"""
Database storage layer for air quality data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Float, DateTime, Integer, JSON, Index, select, func, Boolean, Text
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
import json

from ..config.settings import settings
from ..utils.logger import get_logger
from ..utils.exceptions import DatabaseError
from .models import AirQualityData, PredictionResult, HealthAdvisory

logger = get_logger("storage")

Base = declarative_base()


class AirQualityRecord(Base):
    """Database model for air quality records."""
    __tablename__ = "air_quality_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    location_id = Column(String(100), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    pm25 = Column(Float, nullable=False)
    pm10 = Column(Float, nullable=False)
    no2 = Column(Float)
    co = Column(Float)
    o3 = Column(Float)
    so2 = Column(Float)
    aqi = Column(Float, nullable=False, index=True)
    primary_pollutant = Column(String(50))
    temperature = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    pressure = Column(Float)
    source = Column(String(50))
    quality_score = Column(Float)
    confidence_score = Column(Float)
    source_api = Column(String(50))
    area_type = Column(String(20))
    raw_data = Column(JSON)  # Store complete raw data
    
    __table_args__ = (
        Index("idx_location_timestamp", "location_id", "timestamp"),
        Index("idx_timestamp", "timestamp"),
    )


class PredictionRecord(Base):
    """Database model for prediction records."""
    __tablename__ = "prediction_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(String(100), nullable=False, index=True)
    prediction_timestamp = Column(DateTime, nullable=False, index=True)
    prediction_horizon = Column(Integer, nullable=False)
    predictions = Column(JSON, nullable=False)
    max_risk_level = Column(String(50))
    risk_timeline = Column(JSON)
    model_version = Column(String(20))
    confidence_score = Column(Float)
    prediction_accuracy = Column(Float)
    source_attribution = Column(JSON)
    ethical_explanation = Column(String(500))


class AdvisoryRecord(Base):
    """Database model for health advisory records."""
    __tablename__ = "advisory_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    risk_level = Column(String(50), nullable=False)
    aqi = Column(Float, nullable=False)
    primary_concerns = Column(JSON)
    recommendations = Column(JSON)
    vulnerable_group_advisories = Column(JSON)
    preventive_measures = Column(JSON)
    confidence = Column(Float)
    cumulative_exposure_score = Column(Float)


class User(Base):
    """Database model for user accounts."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(100), nullable=False)
    mobile = Column(String(20), nullable=False, unique=True)
    email = Column(String(100), unique=True)
    address = Column(Text)
    age = Column(Integer, nullable=False)
    chronic_conditions = Column(Text)  # comma-separated string
    is_smoker = Column(Boolean, default=False)
    works_outdoors = Column(Boolean, default=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_mobile", "mobile"),
        Index("idx_email", "email"),
    )


class DatabaseStorage:
    """Database storage manager."""
    
    def __init__(self):
        self.engine = None
        self.async_session = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                settings.database_url.replace("sqlite://", "sqlite+aiosqlite://"),
                echo=settings.debug,
                future=True
            )
            
            # Create async session factory
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
            logger.info("Database initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", error=str(e))
            raise DatabaseError(f"Database initialization failed: {e}")
    
    async def close(self):
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")
    
    def _check_initialized(self):
        """Check if database is initialized."""
        if not self._initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")
    
    def _serialize_for_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for JSON storage, converting datetime to ISO format."""
        import json
        from datetime import datetime
        
        def convert_value(value):
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif hasattr(value, 'value'):  # Enum
                return value.value
            else:
                return value
        
        return {k: convert_value(v) for k, v in data.items()}
    
    async def store_air_quality_data(self, data: AirQualityData) -> int:
        """Store air quality data."""
        self._check_initialized()
        
        try:
            async with self.async_session() as session:
                record = AirQualityRecord(
                    timestamp=data.timestamp,
                    location_id=data.location_id,
                    latitude=data.latitude,
                    longitude=data.longitude,
                    pm25=data.pm25,
                    pm10=data.pm10,
                    no2=data.no2,
                    co=data.co,
                    o3=data.o3,
                    so2=data.so2,
                    aqi=data.aqi,
                    primary_pollutant=data.primary_pollutant,
                    temperature=data.temperature,
                    humidity=data.humidity,
                    wind_speed=data.wind_speed,
                    wind_direction=data.wind_direction,
                    pressure=data.pressure,
                    source=data.source_api,
                    quality_score=data.quality_score,
                    confidence_score=data.confidence_score,
                    source_api=data.source_api,
                    area_type=data.area_type.value if data.area_type else None,
                    raw_data=self._serialize_for_storage(data.dict())
                )
                
                session.add(record)
                await session.commit()
                await session.refresh(record)
                
                logger.debug(f"Stored air quality data for {data.location_id}", record_id=record.id)
                return record.id
        
        except Exception as e:
            logger.error(f"Error storing air quality data: {e}", error=str(e))
            raise DatabaseError(f"Failed to store data: {e}")
    
    async def get_current_air_quality(self, location_id: str) -> Optional[AirQualityData]:
        """Get most recent air quality data for a location."""
        self._check_initialized()
        
        try:
            async with self.async_session() as session:
                stmt = select(AirQualityRecord).where(
                    AirQualityRecord.location_id == location_id
                ).order_by(
                    AirQualityRecord.timestamp.desc()
                ).limit(1)
                
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                
                if record:
                    return self._record_to_model(record)
                return None
        
        except Exception as e:
            logger.error(f"Error getting current air quality: {e}", error=str(e))
            raise DatabaseError(f"Failed to get current data: {e}")
    
    async def get_historical_data(
        self,
        location_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[AirQualityData]:
        """Get historical air quality data."""
        self._check_initialized()
        
        try:
            async with self.async_session() as session:
                stmt = select(AirQualityRecord).where(
                    AirQualityRecord.location_id == location_id,
                    AirQualityRecord.timestamp >= start_time,
                    AirQualityRecord.timestamp <= end_time
                ).order_by(
                    AirQualityRecord.timestamp.desc()
                ).limit(limit)
                
                result = await session.execute(stmt)
                records = result.scalars().all()
                
                return [self._record_to_model(record) for record in records]
        
        except Exception as e:
            logger.error(f"Error getting historical data: {e}", error=str(e))
            raise DatabaseError(f"Failed to get historical data: {e}")
    
    async def store_prediction(self, prediction: PredictionResult) -> int:
        """Store prediction result."""
        self._check_initialized()
        
        try:
            async with self.async_session() as session:
                record = PredictionRecord(
                    location_id=prediction.location_id,
                    prediction_timestamp=prediction.prediction_timestamp,
                    prediction_horizon=prediction.prediction_horizon,
                    predictions=self._serialize_for_storage(prediction.predictions),
                    max_risk_level=prediction.max_risk_level.value,
                    risk_timeline={k: v.value for k, v in prediction.risk_timeline.items()},
                    model_version=prediction.model_version,
                    confidence_score=prediction.confidence_score,
                    prediction_accuracy=prediction.prediction_accuracy,
                    source_attribution=self._serialize_for_storage(prediction.source_attribution) if prediction.source_attribution else None,
                    ethical_explanation=prediction.ethical_explanation
                )
                
                session.add(record)
                await session.commit()
                await session.refresh(record)
                
                return record.id
        
        except Exception as e:
            logger.error(f"Error storing prediction: {e}", error=str(e))
            raise DatabaseError(f"Failed to store prediction: {e}")
    
    async def store_advisory(self, advisory: HealthAdvisory) -> int:
        """Store health advisory."""
        self._check_initialized()
        
        try:
            async with self.async_session() as session:
                record = AdvisoryRecord(
                    location_id=advisory.location_id,
                    timestamp=advisory.timestamp,
                    risk_level=advisory.risk_level.value,
                    aqi=advisory.aqi,
                    primary_concerns=advisory.primary_concerns,
                    recommendations=advisory.recommendations,
                    vulnerable_group_advisories=advisory.vulnerable_group_advisories,
                    preventive_measures=advisory.preventive_measures,
                    confidence=advisory.confidence,
                    cumulative_exposure_score=advisory.cumulative_exposure_score
                )
                
                session.add(record)
                await session.commit()
                await session.refresh(record)
                
                return record.id
        
        except Exception as e:
            logger.error(f"Error storing advisory: {e}", error=str(e))
            raise DatabaseError(f"Failed to store advisory: {e}")
    
    def _record_to_model(self, record: AirQualityRecord) -> AirQualityData:
        """Convert database record to Pydantic model."""
        from .models import AreaType
        
        return AirQualityData(
            timestamp=record.timestamp,
            location_id=record.location_id,
            latitude=record.latitude,
            longitude=record.longitude,
            pm25=record.pm25,
            pm10=record.pm10,
            no2=record.no2,
            co=record.co,
            o3=record.o3,
            so2=record.so2,
            aqi=record.aqi,
            primary_pollutant=record.primary_pollutant or "PM2.5",
            temperature=record.temperature,
            humidity=record.humidity,
            wind_speed=record.wind_speed,
            wind_direction=record.wind_direction,
            pressure=record.pressure,
            source_api=record.source_api or "database",
            quality_score=record.quality_score or 0.5,
            confidence_score=record.confidence_score,
            area_type=AreaType(record.area_type) if record.area_type else None
        )


# Global storage instance
storage = DatabaseStorage()


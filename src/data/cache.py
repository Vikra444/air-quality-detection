"""
Redis caching layer for air quality data.
"""

import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import redis.asyncio as redis
from ..config.settings import settings
from ..utils.logger import get_logger
from .models import AirQualityData

logger = get_logger("cache")


class CacheManager:
    """Redis cache manager."""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.ttl = settings.cache_ttl
        self._initialized = False
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,  # Add connection timeout
                socket_timeout=5,           # Add socket timeout
                retry_on_timeout=True       # Retry on timeout
            )
            await self.client.ping()
            self._initialized = True
            logger.info("Redis cache initialized successfully")
        
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}. Continuing without cache.")
            self.client = None
            self._initialized = False
    
    async def close(self):
        """Close Redis connection."""
        if self.client:
            try:
                await self.client.close()
                logger.info("Redis cache connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self.client = None
                self._initialized = False
    
    def _get_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key."""
        return f"airguard:{prefix}:{identifier}"
    
    async def get_air_quality(
        self,
        location_id: str,
        latitude: float,
        longitude: float
    ) -> Optional[AirQualityData]:
        """Get cached air quality data."""
        if not self.client or not self._initialized:
            return None
        
        try:
            # Try location-based key first
            key = self._get_key("aq", location_id)
            data = await self.client.get(key)
            
            if data:
                data_dict = json.loads(data)
                # Check if data is still fresh
                timestamp = datetime.fromisoformat(data_dict["timestamp"])
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    logger.debug(f"Cache hit for {location_id}")
                    return AirQualityData(**data_dict)
                else:
                    # Expired, remove it
                    await self.client.delete(key)
            
            # Try coordinate-based key
            coord_key = self._get_key("aq_coord", f"{latitude:.4f},{longitude:.4f}")
            data = await self.client.get(coord_key)
            
            if data:
                data_dict = json.loads(data)
                timestamp = datetime.fromisoformat(data_dict["timestamp"])
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    logger.debug(f"Cache hit for coordinates {latitude},{longitude}")
                    return AirQualityData(**data_dict)
                else:
                    await self.client.delete(coord_key)
            
            return None
        
        except redis.ConnectionError:
            logger.warning("Redis connection error, cache unavailable")
            return None
        except redis.TimeoutError:
            logger.warning("Redis timeout error, cache unavailable")
            return None
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}", error=str(e))
            return None
    
    async def set_air_quality(
        self,
        data: AirQualityData,
        ttl: Optional[int] = None
    ):
        """Cache air quality data."""
        if not self.client or not self._initialized:
            return
        
        try:
            ttl = ttl or self.ttl
            data_dict = data.dict()
            data_json = json.dumps(data_dict, default=str)
            
            # Cache by location ID
            location_key = self._get_key("aq", data.location_id)
            await self.client.setex(location_key, ttl, data_json)
            
            # Cache by coordinates
            coord_key = self._get_key("aq_coord", f"{data.latitude:.4f},{data.longitude:.4f}")
            await self.client.setex(coord_key, ttl, data_json)
            
            logger.debug(f"Cached air quality data for {data.location_id}")
        
        except redis.ConnectionError:
            logger.warning("Redis connection error, unable to cache data")
        except redis.TimeoutError:
            logger.warning("Redis timeout error, unable to cache data")
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}", error=str(e))
    
    async def get_prediction(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction."""
        if not self.client or not self._initialized:
            return None
        
        try:
            key = self._get_key("prediction", location_id)
            data = await self.client.get(key)
            
            if data:
                return json.loads(data)
            return None
        
        except redis.ConnectionError:
            logger.warning("Redis connection error, cache unavailable")
            return None
        except redis.TimeoutError:
            logger.warning("Redis timeout error, cache unavailable")
            return None
        except Exception as e:
            logger.warning(f"Error reading prediction from cache: {e}", error=str(e))
            return None
    
    async def set_prediction(
        self,
        location_id: str,
        prediction: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Cache prediction."""
        if not self.client or not self._initialized:
            return
        
        try:
            ttl = ttl or self.ttl
            key = self._get_key("prediction", location_id)
            data_json = json.dumps(prediction, default=str)
            await self.client.setex(key, ttl, data_json)
        
        except redis.ConnectionError:
            logger.warning("Redis connection error, unable to cache prediction")
        except redis.TimeoutError:
            logger.warning("Redis timeout error, unable to cache prediction")
        except Exception as e:
            logger.warning(f"Error writing prediction to cache: {e}", error=str(e))
    
    async def clear_location_cache(self, location_id: str):
        """Clear all cache for a location."""
        if not self.client or not self._initialized:
            return
        
        try:
            keys = [
                self._get_key("aq", location_id),
                self._get_key("prediction", location_id)
            ]
            await self.client.delete(*keys)
            logger.debug(f"Cleared cache for {location_id}")
        
        except redis.ConnectionError:
            logger.warning("Redis connection error, unable to clear cache")
        except redis.TimeoutError:
            logger.warning("Redis timeout error, unable to clear cache")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}", error=str(e))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.client or not self._initialized:
            return {"status": "disabled"}
        
        try:
            info = await self.client.info("stats")
            return {
                "status": "active",
                "keys": await self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0)
            }
        except redis.ConnectionError:
            logger.warning("Redis connection error, unable to get stats")
            return {"status": "error", "error": "connection_error"}
        except redis.TimeoutError:
            logger.warning("Redis timeout error, unable to get stats")
            return {"status": "error", "error": "timeout_error"}
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}", error=str(e))
            return {"status": "error", "error": str(e)}


# Global cache instance
cache = CacheManager()
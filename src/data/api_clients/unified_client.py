"""
Unified API client with fallback mechanism.
"""

import asyncio
import time
from typing import Dict, Optional, Any, List
from datetime import datetime
from .base_client import BaseAPIClient
from .openweather_client import OpenWeatherClient
from .openmeteo_client import OpenMeteoClient
from .confidence_scorer import ConfidenceScorer
from ...utils.logger import get_logger
from ...utils.exceptions import APIError

logger = get_logger("api_client.unified")


class UnifiedAirQualityClient:
    """
    Unified client that orchestrates multiple API clients with fallback mechanism.
    """
    
    def __init__(self):
        self.clients: List[BaseAPIClient] = [
            OpenWeatherClient(),
            OpenMeteoClient()  # Free, no key needed - used as fallback
        ]
        self.confidence_scorer = ConfidenceScorer()
        self.preferred_order = [0, 1]  # Order of preference
    
    async def fetch_air_quality(
        self,
        latitude: float,
        longitude: float,
        prefer_source: Optional[str] = None,
        use_fallback: bool = True,
        timeout: int = 15  # Reduced timeout for faster response
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch air quality data with automatic fallback.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            prefer_source: Preferred API source name
            use_fallback: Whether to use fallback if primary fails
        
        Returns:
            Normalized air quality data with confidence score
        """
        # Reorder clients if preference specified
        clients_to_try = self._get_client_order(prefer_source)
        
        best_result = None
        best_confidence = 0.0
        errors = []
        
        for client in clients_to_try:
            try:
                start_time = time.time()
                # Use asyncio.wait_for for timeout control
                data = await asyncio.wait_for(
                    client.fetch_air_quality(latitude, longitude),
                    timeout=timeout
                )
                latency = time.time() - start_time
                
                if data:
                    # Calculate confidence score
                    consistency = self.confidence_scorer.calculate_consistency(data)
                    confidence = self.confidence_scorer.calculate_confidence(
                        data, latency, client.__class__.__name__, consistency
                    )
                    
                    data["confidence_score"] = confidence
                    data["source_api"] = client.__class__.__name__
                    data["fetch_latency"] = latency
                    
                    logger.info(
                        f"Successfully fetched data from {client.__class__.__name__}",
                        source=client.__class__.__name__,
                        confidence=confidence,
                        latency=latency
                    )
                    
                    # If this is the first successful result or has higher confidence, use it
                    if confidence > best_confidence:
                        best_result = data
                        best_confidence = confidence
                    
                    # If confidence is high enough, return immediately
                    if confidence >= 0.8 and not use_fallback:
                        return best_result
            
            except asyncio.TimeoutError:
                error_msg = f"{client.__class__.__name__}: Request timeout"
                errors.append(error_msg)
                logger.warning(f"Timeout fetching from {client.__class__.__name__}")
                if not use_fallback:
                    break
            except Exception as e:
                error_msg = f"{client.__class__.__name__}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Failed to fetch from {client.__class__.__name__}: {e}")
                
                # Continue to next client if fallback enabled
                if not use_fallback:
                    break
        
        if best_result:
            logger.info(
                f"Returning best result with confidence {best_confidence}",
                confidence=best_confidence,
                source=best_result.get("source_api")
            )
            return best_result
        
        # All APIs failed - return None so caller can use fallback
        logger.warning(
            "All API clients failed, returning None for fallback",
            errors=errors,
            latitude=latitude,
            longitude=longitude
        )
        return None  # Return None instead of raising, so demo data can be used
    
    def _get_client_order(self, prefer_source: Optional[str] = None) -> List[BaseAPIClient]:
        """Get ordered list of clients based on preference."""
        if not prefer_source:
            return [self.clients[i] for i in self.preferred_order]
        
        # Find preferred client and move to front
        preferred_index = None
        for i, client in enumerate(self.clients):
            if prefer_source.lower() in client.__class__.__name__.lower():
                preferred_index = i
                break
        
        if preferred_index is not None:
            order = [preferred_index] + [i for i in self.preferred_order if i != preferred_index]
            return [self.clients[i] for i in order]
        
        return [self.clients[i] for i in self.preferred_order]
    
    async def fetch_multiple_locations(
        self,
        locations: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch air quality for multiple locations in parallel."""
        tasks = [
            self.fetch_air_quality(loc["latitude"], loc["longitude"])
            for loc in locations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        location_data = {}
        for i, (loc, result) in enumerate(zip(locations, results)):
            location_id = loc.get("location_id", f"location_{i}")
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {location_id}: {result}")
                location_data[location_id] = None
            else:
                location_data[location_id] = result
        
        return location_data
    
    async def close(self):
        """Close all client sessions."""
        for client in self.clients:
            await client.close()
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for all clients."""
        return {
            client.__class__.__name__: client.get_health_metrics()
            for client in self.clients
        }


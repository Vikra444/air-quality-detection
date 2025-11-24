"""
Base class for air quality API clients.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from ...utils.logger import get_logger
from ...utils.exceptions import APIError

logger = get_logger("api_client")


class BaseAPIClient(ABC):
    """Base class for all API clients."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "", timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        self.request_count = 0
        self.error_count = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _rate_limit(self):
        """Rate limiting to avoid overwhelming APIs."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        await self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()
        
        try:
            self.request_count += 1
            logger.debug(f"Making {method} request to {url}", url=url, params=params)
            
            async with session.request(
                method=method,
                url=url,
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Request successful: {url}")
                    return data
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    raise APIError(
                        f"API request failed with status {response.status}: {error_text}",
                        api_name=self.__class__.__name__,
                        status_code=response.status
                    )
        
        except aiohttp.ClientError as e:
            self.error_count += 1
            raise APIError(
                f"Network error: {str(e)}",
                api_name=self.__class__.__name__
            )
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get API client health metrics."""
        return {
            "api_name": self.__class__.__name__,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "last_request_time": self.last_request_time
        }
    
    @abstractmethod
    async def fetch_air_quality(
        self,
        latitude: float,
        longitude: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Fetch air quality data for given coordinates."""
        pass
    
    @abstractmethod
    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize API response to standard format."""
        pass


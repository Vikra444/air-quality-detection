"""
Rate limiting utilities for AirGuard API security.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
from fastapi import HTTPException, status
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger("security.rate_limiting")


class RateLimiter:
    """Rate limiter to prevent API abuse."""
    
    def __init__(self):
        self.requests = defaultdict(deque)  # Store timestamps of requests
        self.blocked_ips = {}  # Store blocked IPs with expiration time
    
    def is_allowed(self, identifier: str, max_requests: int = None, window_seconds: int = None) -> bool:
        """
        Check if a request is allowed based on rate limiting.
        
        Args:
            identifier: Unique identifier (IP address, API key, etc.)
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
            
        Returns:
            bool: True if request is allowed, False if rate limited
        """
        # Use settings defaults if not provided
        if max_requests is None:
            max_requests = settings.api_rate_limit
        if window_seconds is None:
            window_seconds = settings.api_rate_limit_window
        
        current_time = time.time()
        
        # Check if IP is blocked
        if identifier in self.blocked_ips:
            block_expiration = self.blocked_ips[identifier]
            if current_time < block_expiration:
                return False
            else:
                # Unblock if expiration time has passed
                del self.blocked_ips[identifier]
        
        # Clean old requests outside the window
        request_times = self.requests[identifier]
        while request_times and request_times[0] < current_time - window_seconds:
            request_times.popleft()
        
        # Check if within rate limit
        if len(request_times) < max_requests:
            # Add current request
            request_times.append(current_time)
            return True
        else:
            # Rate limit exceeded, block for a period
            block_duration = 60  # Block for 1 minute
            self.blocked_ips[identifier] = current_time + block_duration
            logger.warning(f"Rate limit exceeded for {identifier}. Blocking for {block_duration} seconds.")
            return False
    
    def get_current_usage(self, identifier: str) -> Dict[str, int]:
        """Get current usage statistics for an identifier."""
        request_times = self.requests.get(identifier, deque())
        current_time = time.time()
        
        # Clean old requests
        while request_times and request_times[0] < current_time - settings.api_rate_limit_window:
            request_times.popleft()
        
        return {
            "current_requests": len(request_times),
            "max_requests": settings.api_rate_limit,
            "window_seconds": settings.api_rate_limit_window,
            "blocked": identifier in self.blocked_ips
        }
    
    def reset_for_identifier(self, identifier: str):
        """Reset rate limiting for a specific identifier."""
        if identifier in self.requests:
            del self.requests[identifier]
        if identifier in self.blocked_ips:
            del self.blocked_ips[identifier]
        logger.info(f"Rate limiting reset for {identifier}")


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit_dependency(
    identifier: str,
    max_requests: Optional[int] = None,
    window_seconds: Optional[int] = None
):
    """
    Dependency function for FastAPI rate limiting.
    
    Args:
        identifier: Unique identifier for rate limiting
        max_requests: Maximum requests allowed (overrides settings)
        window_seconds: Time window in seconds (overrides settings)
    """
    if not rate_limiter.is_allowed(identifier, max_requests, window_seconds):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )


def get_client_ip(request) -> str:
    """Extract client IP address from request."""
    # Check for forwarded headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to client host
    if request.client:
        return request.client.host
    
    return "unknown"
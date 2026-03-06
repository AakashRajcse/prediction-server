"""
Rate Limiting Middleware
Simple in-memory rate limiter for API protection
"""
import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse


class RateLimiter:
    """Simple token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests = defaultdict(lambda: {"count": 0, "reset_time": 0})
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        current_time = time.time()
        
        # Reset if minute has passed
        if current_time > self.requests[client_id]["reset_time"]:
            self.requests[client_id] = {
                "count": 1,
                "reset_time": current_time + 60
            }
            return True
        
        # Check if under limit
        if self.requests[client_id]["count"] < self.requests_per_minute:
            self.requests[client_id]["count"] += 1
            return True
        
        return False
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client"""
        if client_id not in self.requests:
            return self.requests_per_minute
        
        current_time = time.time()
        if current_time > self.requests[client_id]["reset_time"]:
            return self.requests_per_minute
        
        return max(0, self.requests_per_minute - self.requests[client_id]["count"])
    
    def get_reset_time(self, client_id: str) -> int:
        """Get seconds until rate limit resets"""
        if client_id not in self.requests:
            return 60
        
        current_time = time.time()
        return max(0, int(self.requests[client_id]["reset_time"] - current_time))


# Global rate limiter instance
_rate_limiter = RateLimiter(
    requests_per_minute=60,  # 60 requests per minute
    burst_size=10            # Allow bursts up to 10
)


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Skip rate limiting for health check and docs
    if request.url.path in ["/", "/api/v1/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Get client identifier (IP address)
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not _rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please try again later.",
                "retry_after": _rate_limiter.get_reset_time(client_ip),
                "limit": _rate_limiter.requests_per_minute
            },
            headers={
                "X-RateLimit-Limit": str(_rate_limiter.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(_rate_limiter.get_reset_time(client_ip)),
                "Retry-After": str(_rate_limiter.get_reset_time(client_ip))
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(_rate_limiter.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(_rate_limiter.get_remaining(client_ip))
    response.headers["X-RateLimit-Reset"] = str(_rate_limiter.get_reset_time(client_ip))
    
    return response

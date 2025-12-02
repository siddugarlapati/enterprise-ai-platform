from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import Dict, Optional
import asyncio
from collections import defaultdict

from app.config.settings import settings

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_window: int = 100, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.clients: Dict[str, Dict] = defaultdict(lambda: {"tokens": requests_per_window, "last_update": time.time()})
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> tuple[bool, Dict]:
        async with self._lock:
            now = time.time()
            client = self.clients[client_id]
            
            # Refill tokens based on time passed
            time_passed = now - client["last_update"]
            tokens_to_add = (time_passed / self.window_seconds) * self.requests_per_window
            client["tokens"] = min(self.requests_per_window, client["tokens"] + tokens_to_add)
            client["last_update"] = now
            
            if client["tokens"] >= 1:
                client["tokens"] -= 1
                return True, {
                    "remaining": int(client["tokens"]),
                    "limit": self.requests_per_window,
                    "reset": int(now + self.window_seconds)
                }
            
            return False, {
                "remaining": 0,
                "limit": self.requests_per_window,
                "reset": int(now + self.window_seconds),
                "retry_after": int(self.window_seconds - time_passed)
            }

    def get_client_id(self, request: Request) -> str:
        # Use API key if present, otherwise use IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key[:16]}"
        
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        return f"ip:{request.client.host if request.client else 'unknown'}"

rate_limiter = RateLimiter(
    requests_per_window=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW
)

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        client_id = rate_limiter.get_client_id(request)
        allowed, info = await rate_limiter.is_allowed(client_id)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(info.get("retry_after", 60))
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])
        
        return response

"""
Redis caching utilities for predictions and responses.
"""
import logging
import hashlib
import json
import pickle
from typing import Any, Optional, Callable
from functools import wraps
import asyncio

from app.db.connection import get_redis
from app.config.settings import settings

logger = logging.getLogger(__name__)


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
    return f"cache:{hashlib.md5(key_data.encode()).hexdigest()}"


def cache_response(
    ttl: int = 3600,
    key_prefix: Optional[str] = None,
    serialize: str = "json"
):
    """
    Decorator to cache function responses in Redis.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Custom key prefix
        serialize: Serialization method ('json' or 'pickle')
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            try:
                redis = await get_redis()
                
                # Try to get from cache
                cached = await redis.get(cache_key)
                if cached:
                    logger.debug(f"Cache hit: {cache_key}")
                    if serialize == "json":
                        return json.loads(cached)
                    else:
                        return pickle.loads(cached)
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                if serialize == "json":
                    await redis.setex(cache_key, ttl, json.dumps(result))
                else:
                    await redis.setex(cache_key, ttl, pickle.dumps(result))
                
                logger.debug(f"Cache set: {cache_key}")
                return result
                
            except Exception as e:
                logger.warning(f"Cache error: {e}")
                # Fallback to executing function without cache
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class CacheManager:
    """Manage cache operations."""
    
    def __init__(self):
        self._redis = None
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            self._redis = await get_redis()
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            redis = await self._get_redis()
            value = await redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> bool:
        """Set value in cache."""
        try:
            redis = await self._get_redis()
            await redis.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            redis = await self._get_redis()
            await redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            redis = await self._get_redis()
            keys = []
            async for key in redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await redis.delete(*keys)
            
            return len(keys)
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
            return 0
    
    async def clear_all(self) -> bool:
        """Clear all cache."""
        try:
            redis = await self._get_redis()
            await redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            redis = await self._get_redis()
            info = await redis.info()
            
            return {
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)


# Singleton instance
cache_manager = CacheManager()

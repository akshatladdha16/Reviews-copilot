import redis
import json
import os
from functools import wraps
from typing import Any, Callable
import asyncio

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.local_cache = {}
        
    def init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(
                os.getenv('REDIS_URL', 'redis://localhost:6379'),
                decode_responses=True
            )
            self.redis_client.ping()  # Test connection
            print("Redis cache initialized")
        except Exception as e:
            print(f"Redis not available, using local cache: {e}")
            self.redis_client = None
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                return self.local_cache.get(key)
        except Exception:
            return self.local_cache.get(key)
    
    def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache with expiration"""
        try:
            if self.redis_client:
                self.redis_client.setex(key, expire, json.dumps(value))
            else:
                self.local_cache[key] = value
        except Exception:
            self.local_cache[key] = value
    
    def delete(self, key: str):
        """Delete key from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self.local_cache.pop(key, None)
        except Exception:
            self.local_cache.pop(key, None)

# Global cache instance
cache = CacheManager()

def cached(expire: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, expire)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, expire)
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
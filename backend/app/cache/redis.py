from __future__ import annotations

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.app.core.config import settings


def create_redis_client() -> Redis:
    return redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )
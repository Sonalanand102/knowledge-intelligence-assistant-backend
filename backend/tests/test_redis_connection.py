from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from backend.app.cache.redis import create_redis_client


@pytest_asyncio.fixture
async def redis_client():
    client = create_redis_client()

    try:
        await client.ping()
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_redis_connection(redis_client):
    assert await redis_client.ping() is True


@pytest.mark.asyncio
async def test_redis_set_and_get(redis_client):
    key = f"test:kia:{uuid.uuid4().hex}"
    value = "hello-kia"

    try:
        await redis_client.set(
            key,
            value,
            ex=30,
        )

        result = await redis_client.get(key)

        assert result == value

    finally:
        await redis_client.delete(key)
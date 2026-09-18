from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app


@pytest.mark.asyncio
async def test_app_health_and_redis_lifespan():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/v1/health"
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }
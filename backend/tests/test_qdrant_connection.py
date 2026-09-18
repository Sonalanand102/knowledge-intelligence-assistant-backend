from __future__ import annotations

import pytest

from backend.app.retrieval.qdrant import (
    create_qdrant_client,
)


@pytest.mark.asyncio
async def test_qdrant_connection():
    client = create_qdrant_client()

    try:
        collections = await client.get_collections()

        assert collections is not None

    finally:
        await client.close()
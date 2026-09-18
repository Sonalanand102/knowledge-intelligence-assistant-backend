from __future__ import annotations

import qdrant_client
from qdrant_client import AsyncQdrantClient

from backend.app.core.config import settings


def create_qdrant_client() -> AsyncQdrantClient:
    return qdrant_client.AsyncQdrantClient(
        url=settings.qdrant_url,
    )
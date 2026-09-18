from __future__ import annotations

import asyncio

from backend.app.core.config import settings

from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore




async def main() -> None:
    client = create_qdrant_client()

    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
    )

    try:
        collection = await client.get_collection(
            collection_name=settings.qdrant_collection_name,
        )

        before_count = collection.points_count

        print(
            f"Collection: {settings.qdrant_collection_name}"
        )
        print(
            f"Points before backfill: {before_count}"
        )

        updated_count = await store.backfill_bm25(
            batch_size=100,
        )

        collection = await client.get_collection(
            collection_name=settings.qdrant_collection_name,
        )

        after_count = collection.points_count

        print(
            f"BM25 points updated: {updated_count}"
        )
        print(
            f"Points after backfill: {after_count}"
        )

        if before_count != after_count:
            raise RuntimeError(
                "Point count changed during BM25 backfill"
            )

        if updated_count != before_count:
            raise RuntimeError(
                "Not all existing points were backfilled"
            )

        print("BM25 backfill completed successfully.")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
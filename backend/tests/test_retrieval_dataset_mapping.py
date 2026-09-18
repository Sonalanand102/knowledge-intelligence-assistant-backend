from __future__ import annotations

import pytest

from backend.app.core.config import settings
from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_cases,
    get_embedding_evaluation_corpus,
)
from backend.app.retrieval.qdrant import create_qdrant_client


@pytest.mark.asyncio
async def test_golden_corpus_is_present_in_qdrant():
    client = create_qdrant_client()

    try:
        records, _ = await client.scroll(
            collection_name=settings.qdrant_collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        qdrant_by_content = {}

        for record in records:
            payload = record.payload or {}
            content = payload.get("content")

            if content:
                qdrant_by_content[str(content)] = str(
                    payload["chunk_id"]
                )

        corpus = get_embedding_evaluation_corpus()

        assert corpus

        missing = []

        for golden_id, content in corpus.items():
            if content not in qdrant_by_content:
                missing.append(golden_id)

        assert not missing, (
            "Golden corpus chunks missing from Qdrant: "
            f"{missing}"
        )

        # Every golden chunk must map to one actual Qdrant chunk_id.
        actual_chunk_ids = {
            qdrant_by_content[content]
            for content in corpus.values()
        }

        assert len(actual_chunk_ids) == len(corpus)

        # Every evaluation case must reference a known golden chunk.
        known_ids = set(corpus)

        for case in get_embedding_evaluation_cases():
            assert case.relevant_chunk_ids <= known_ids

    finally:
        await client.close()
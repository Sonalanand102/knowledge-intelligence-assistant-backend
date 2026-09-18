from __future__ import annotations

import uuid

import pytest

from backend.app.embeddings.local import LocalEmbeddingProvider
from backend.app.embeddings.service import EmbeddingService
from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_corpus,
)
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.dense_retriever import DenseRetriever
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore
from backend.app.retrieval.vector_indexer import VectorIndexer


@pytest.mark.asyncio
async def test_vector_indexer_indexes_corpus_into_qdrant():
    client = create_qdrant_client()

    collection_name = (
        f"vector_indexer_test_{uuid.uuid4().hex}"
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    embedding_service = EmbeddingService(
        provider=LocalEmbeddingProvider(),
    )

    indexer = VectorIndexer(
        embedding_service=embedding_service,
        vector_store=store,
    )

    corpus = get_embedding_evaluation_corpus()

    chunks = [
        ChunkDocument(
            content=content,
            document_id=chunk_id,
            chunk_index=0,
        )
        for chunk_id, content in corpus.items()
    ]

    try:
        await indexer.index(chunks)

        results = await store.search(
            query_embedding=(
                embedding_service.embed_query(
                    "What is RAG?"
                )
            ),
            top_k=5,
        )

        assert len(results) == 5

        result_chunk_ids = {
            result.chunk_id
            for result in results
        }

        rag_chunk_ids = {
            chunk.chunk_id
            for chunk in chunks
            if chunk.document_id.startswith(
                "chunk_rag_"
            )
        }

        assert result_chunk_ids.intersection(
            rag_chunk_ids
        )

    finally:
        exists = await client.collection_exists(
            collection_name=collection_name,
        )

        if exists:
            await client.delete_collection(
                collection_name=collection_name,
            )

        await client.close()
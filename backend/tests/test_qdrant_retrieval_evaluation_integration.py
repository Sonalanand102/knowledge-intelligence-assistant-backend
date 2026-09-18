from __future__ import annotations

import uuid

import pytest

from backend.app.embeddings.local import LocalEmbeddingProvider
from backend.app.embeddings.service import EmbeddingService
from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_cases,
    get_embedding_evaluation_corpus,
)
from backend.app.evaluation.retrieval_runner import (
    DenseRetrievalEvaluationRunner,
)
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.dense_retriever import DenseRetriever
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore


@pytest.mark.asyncio
async def test_qdrant_dense_retrieval_evaluation():
    client = create_qdrant_client()

    collection_name = (
        f"dense_eval_{uuid.uuid4().hex}"
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    embedding_service = EmbeddingService(
        provider=LocalEmbeddingProvider(),
    )

    retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=store,
    )

    runner = DenseRetrievalEvaluationRunner(
        retriever=retriever,
    )

    corpus = get_embedding_evaluation_corpus()
    cases = get_embedding_evaluation_cases()

    chunks = [
        ChunkDocument(
            content=content,
            document_id=chunk_id,
            chunk_index=0,
        )
        for chunk_id, content in corpus.items()
    ]

    chunk_id_mapping = {
        chunk.document_id: chunk.chunk_id
        for chunk in chunks
    }

    resolved_cases = [
        type(case)(
            query=case.query,
            relevant_chunk_ids={
                chunk_id_mapping[chunk_id]
                for chunk_id in case.relevant_chunk_ids
            },
        )
        for case in cases
    ]

    try:
        embedded_chunks = (
            embedding_service.embed_chunks(chunks)
        )

        await store.upsert(
            embedded_chunks
        )

        results = await runner.evaluate_cases(
            cases=resolved_cases,
            k=5,
        )

        aggregate = runner.aggregate_results(
            results
        )

        assert len(results) == len(cases)

        assert (
            aggregate.evaluated_queries
            == len(cases)
        )

        assert 0.0 <= aggregate.recall_at_k <= 1.0
        assert 0.0 <= aggregate.precision_at_k <= 1.0
        assert 0.0 <= aggregate.mrr <= 1.0

        print(
            "\nDense Retrieval Evaluation"
            f"\nRecall@5: {aggregate.recall_at_k:.4f}"
            f"\nPrecision@5: {aggregate.precision_at_k:.4f}"
            f"\nMRR: {aggregate.mrr:.4f}"
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
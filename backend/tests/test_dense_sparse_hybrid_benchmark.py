from __future__ import annotations

import pytest
from google import genai

from backend.app.core.config import settings
from backend.app.embeddings.evaluation import (
    EmbeddingEvaluationCase,
)
from backend.app.embeddings.gemini import (
    GeminiEmbeddingProvider,
)
from backend.app.embeddings.service import (
    EmbeddingService,
)
from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_cases,
    get_embedding_evaluation_corpus,
)
from backend.app.evaluation.retrieval_runner import (
    DenseRetrievalEvaluationRunner,
)
from backend.app.retrieval.dense_retriever import (
    DenseRetriever,
)
from backend.app.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from backend.app.retrieval.qdrant import (
    create_qdrant_client,
)
from backend.app.retrieval.qdrant_store import (
    QdrantVectorStore,
)
from backend.app.retrieval.sparse_retriever import (
    SparseRetriever,
)


async def _build_golden_id_mapping(
    client,
) -> dict[str, str]:
    records, _ = await client.scroll(
        collection_name=settings.qdrant_collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    qdrant_by_content: dict[str, str] = {}

    for record in records:
        payload = record.payload or {}

        content = payload.get("content")
        chunk_id = payload.get("chunk_id")

        if content is not None and chunk_id is not None:
            qdrant_by_content[str(content)] = str(
                chunk_id
            )

    corpus = get_embedding_evaluation_corpus()

    mapping: dict[str, str] = {}

    for golden_id, content in corpus.items():
        actual_chunk_id = qdrant_by_content.get(content)

        if actual_chunk_id is None:
            raise AssertionError(
                f"Golden chunk not found in Qdrant: {golden_id}"
            )

        mapping[golden_id] = actual_chunk_id

    return mapping


def _map_evaluation_cases(
    cases: list[EmbeddingEvaluationCase],
    mapping: dict[str, str],
) -> list[EmbeddingEvaluationCase]:
    mapped_cases: list[EmbeddingEvaluationCase] = []

    for case in cases:
        mapped_cases.append(
            EmbeddingEvaluationCase(
                query=case.query,
                relevant_chunk_ids={
                    mapping[golden_id]
                    for golden_id in case.relevant_chunk_ids
                },
            )
        )

    return mapped_cases



@pytest.mark.asyncio
async def test_dense_sparse_hybrid_benchmark():
    client = create_qdrant_client()

    try:
        mapping = await _build_golden_id_mapping(client)

        golden_cases = get_embedding_evaluation_cases()

        cases = _map_evaluation_cases(
            golden_cases,
            mapping,
        )

        # --------------------------------------------------
        # Infrastructure
        # --------------------------------------------------

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection_name,
        )

        gemini_client = genai.Client(
            api_key=settings.gemini_api_key,
        )

        embedding_provider = GeminiEmbeddingProvider(
            client=gemini_client,
        )

        embedding_service = EmbeddingService(
            provider=embedding_provider,
        )

        dense_retriever = DenseRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        sparse_retriever = SparseRetriever(
            vector_store=vector_store,
        )

        # --------------------------------------------------
        # Evaluation runners
        # --------------------------------------------------

        dense_runner = DenseRetrievalEvaluationRunner(
            retriever=dense_retriever,
        )

        sparse_runner = DenseRetrievalEvaluationRunner(
            retriever=sparse_retriever,
        )

        # --------------------------------------------------
        # Evaluation configuration
        # --------------------------------------------------

        k = 5
        rrf_k_values = [5, 10, 20, 60]

        # --------------------------------------------------
        # Dense
        # --------------------------------------------------

        dense_results = await dense_runner.evaluate_cases(
            cases,
            k=k,
        )

        dense_metrics = dense_runner.aggregate_results(
            dense_results,
        )

        # --------------------------------------------------
        # Sparse
        # --------------------------------------------------

        sparse_results = await sparse_runner.evaluate_cases(
            cases,
            k=k,
        )

        sparse_metrics = sparse_runner.aggregate_results(
            sparse_results,
        )

        # --------------------------------------------------
        # Hybrid RRF tuning
        # --------------------------------------------------

        hybrid_results_by_rrf_k = {}
        hybrid_metrics_by_rrf_k = {}

        for rrf_k in rrf_k_values:
            hybrid_retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                rrf_k=rrf_k,
            )

            hybrid_runner = DenseRetrievalEvaluationRunner(
                retriever=hybrid_retriever,
            )

            hybrid_results = await hybrid_runner.evaluate_cases(
                cases,
                k=k,
            )

            hybrid_metrics = hybrid_runner.aggregate_results(
                hybrid_results,
            )

            hybrid_results_by_rrf_k[rrf_k] = hybrid_results
            hybrid_metrics_by_rrf_k[rrf_k] = hybrid_metrics

        # --------------------------------------------------
        # Print baseline benchmark
        # --------------------------------------------------

        print("\n" + "=" * 72)
        print("DENSE vs SPARSE vs HYBRID RETRIEVAL")
        print("=" * 72)

        print(
            f"\nEvaluated queries: "
            f"{dense_metrics.evaluated_queries}"
        )

        print(
            "\nDense:"
            f"\n  Recall@5:    {dense_metrics.recall_at_k:.4f}"
            f"\n  Precision@5: {dense_metrics.precision_at_k:.4f}"
            f"\n  MRR:         {dense_metrics.mrr:.4f}"
        )

        print(
            "\nSparse / BM25:"
            f"\n  Recall@5:    {sparse_metrics.recall_at_k:.4f}"
            f"\n  Precision@5: {sparse_metrics.precision_at_k:.4f}"
            f"\n  MRR:         {sparse_metrics.mrr:.4f}"
        )

        # --------------------------------------------------
        # Print RRF tuning results
        # --------------------------------------------------

        print("\n" + "=" * 72)
        print("RRF K TUNING")
        print("=" * 72)

        for rrf_k in rrf_k_values:
            metrics = hybrid_metrics_by_rrf_k[rrf_k]

            print(
                f"\nHybrid / RRF k={rrf_k}:"
                f"\n  Recall@5:    {metrics.recall_at_k:.4f}"
                f"\n  Precision@5: {metrics.precision_at_k:.4f}"
                f"\n  MRR:         {metrics.mrr:.4f}"
            )

        # --------------------------------------------------
        # Query-level analysis
        # --------------------------------------------------

        print("\n" + "=" * 72)
        print("QUERY-LEVEL ANALYSIS")
        print("=" * 72)

        for rrf_k in rrf_k_values:
            hybrid_results = hybrid_results_by_rrf_k[rrf_k]

            print("\n" + "-" * 72)
            print(f"RRF k={rrf_k}")
            print("-" * 72)

            for case, dense, sparse, hybrid in zip(
                cases,
                dense_results,
                sparse_results,
                hybrid_results,
            ):
                print(f"\nQuery: {case.query}")

                print(
                    f"Dense   → "
                    f"R={dense.recall_at_k:.2f} "
                    f"P={dense.precision_at_k:.2f} "
                    f"MRR={dense.mrr:.2f}"
                )

                print(
                    f"Sparse  → "
                    f"R={sparse.recall_at_k:.2f} "
                    f"P={sparse.precision_at_k:.2f} "
                    f"MRR={sparse.mrr:.2f}"
                )

                print(
                    f"Hybrid  → "
                    f"R={hybrid.recall_at_k:.2f} "
                    f"P={hybrid.precision_at_k:.2f} "
                    f"MRR={hybrid.mrr:.2f}"
                )

                print(
                    f"Relevant: "
                    f"{sorted(case.relevant_chunk_ids)}"
                )

                print(
                    f"Hybrid ranking: "
                    f"{hybrid.ranked_chunk_ids[:5]}"
                )

        print("\n" + "=" * 72)

        # --------------------------------------------------
        # Sanity checks
        # --------------------------------------------------

        assert len(dense_results) == len(cases)
        assert len(sparse_results) == len(cases)

        for rrf_k in rrf_k_values:
            assert len(
                hybrid_results_by_rrf_k[rrf_k]
            ) == len(cases)

    finally:
        await client.close()
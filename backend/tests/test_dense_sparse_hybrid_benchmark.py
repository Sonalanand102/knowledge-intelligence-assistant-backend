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
from backend.app.retrieval.cross_encoder_reranker import (
    CrossEncoderReranker,
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
from backend.app.retrieval.reranking_retriever import (
    RerankingRetriever,
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
        # --------------------------------------------------
        # Golden dataset
        # --------------------------------------------------

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
        candidate_k_values = [10, 20, 30]

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
        # Hybrid baseline for reranking
        # --------------------------------------------------

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            rrf_k=60,
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

        # --------------------------------------------------
        # Cross-Encoder candidate_k benchmark
        # --------------------------------------------------

        reranker = CrossEncoderReranker()

        reranked_results_by_candidate_k = {}
        reranked_metrics_by_candidate_k = {}

        for candidate_k in candidate_k_values:
            reranking_retriever = RerankingRetriever(
                retriever=hybrid_retriever,
                reranker=reranker,
                candidate_k=candidate_k,
            )

            reranking_runner = DenseRetrievalEvaluationRunner(
                retriever=reranking_retriever,
            )

            reranked_results = await reranking_runner.evaluate_cases(
                cases,
                k=k,
            )

            reranked_metrics = reranking_runner.aggregate_results(
                reranked_results,
            )

            reranked_results_by_candidate_k[candidate_k] = (
                reranked_results
            )

            reranked_metrics_by_candidate_k[candidate_k] = (
                reranked_metrics
            )

        # --------------------------------------------------
        # Baseline benchmark
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
        # RRF summary
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
        # Candidate K benchmark
        # --------------------------------------------------

        print("\n" + "=" * 72)
        print("CROSS-ENCODER CANDIDATE_K BENCHMARK")
        print("=" * 72)

        print(
            "\nHybrid baseline / RRF k=60:"
            f"\n  Recall@5:    {hybrid_metrics.recall_at_k:.4f}"
            f"\n  Precision@5: {hybrid_metrics.precision_at_k:.4f}"
            f"\n  MRR:         {hybrid_metrics.mrr:.4f}"
        )

        for candidate_k in candidate_k_values:
            metrics = reranked_metrics_by_candidate_k[
                candidate_k
            ]

            print(
                f"\nCandidate k={candidate_k}:"
                f"\n  Recall@5:    {metrics.recall_at_k:.4f}"
                f"\n  Precision@5: {metrics.precision_at_k:.4f}"
                f"\n  MRR:         {metrics.mrr:.4f}"
            )

            print(
                f"  Recall Δ:    "
                f"{metrics.recall_at_k - hybrid_metrics.recall_at_k:+.4f}"
            )

            print(
                f"  Precision Δ: "
                f"{metrics.precision_at_k - hybrid_metrics.precision_at_k:+.4f}"
            )

            print(
                f"  MRR Δ:        "
                f"{metrics.mrr - hybrid_metrics.mrr:+.4f}"
            )

        # --------------------------------------------------
        # Query-level changes only
        # --------------------------------------------------

        print("\n" + "=" * 72)
        print("QUERIES AFFECTED BY CANDIDATE_K")
        print("=" * 72)

        affected_queries = 0

        for index, case in enumerate(cases):
            baseline = hybrid_results[index]

            candidate_metrics = []

            for candidate_k in candidate_k_values:
                result = reranked_results_by_candidate_k[
                    candidate_k
                ][index]

                candidate_metrics.append(
                    (
                        candidate_k,
                        result.recall_at_k,
                        result.precision_at_k,
                        result.mrr,
                    )
                )

            baseline_metrics = (
                baseline.recall_at_k,
                baseline.precision_at_k,
                baseline.mrr,
            )

            changed = any(
                (
                    recall,
                    precision,
                    mrr,
                )
                != baseline_metrics
                for _, recall, precision, mrr in candidate_metrics
            )

            if not changed:
                continue

            affected_queries += 1

            print(f"\nQuery: {case.query}")

            print(
                f"  Hybrid k=60 → "
                f"R={baseline.recall_at_k:.2f} "
                f"P={baseline.precision_at_k:.2f} "
                f"MRR={baseline.mrr:.2f}"
            )

            for (
                candidate_k,
                recall,
                precision,
                mrr,
            ) in candidate_metrics:
                print(
                    f"  candidate_k={candidate_k} → "
                    f"R={recall:.2f} "
                    f"P={precision:.2f} "
                    f"MRR={mrr:.2f}"
                )

        if affected_queries == 0:
            print("\nNo queries changed across candidate_k values.")

        print(
            f"\nAffected queries: "
            f"{affected_queries}/{len(cases)}"
        )

        # --------------------------------------------------
        # Sanity checks
        # --------------------------------------------------

        assert len(dense_results) == len(cases)
        assert len(sparse_results) == len(cases)
        assert len(hybrid_results) == len(cases)

        for rrf_k in rrf_k_values:
            assert len(
                hybrid_results_by_rrf_k[rrf_k]
            ) == len(cases)

        for candidate_k in candidate_k_values:
            assert len(
                reranked_results_by_candidate_k[candidate_k]
            ) == len(cases)

    finally:
        await client.close()
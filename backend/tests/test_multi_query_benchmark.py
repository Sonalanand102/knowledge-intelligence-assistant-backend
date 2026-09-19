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
from backend.app.retrieval.multi_query_retriever import (
    BaseMultiQueryGenerator,
    GeminiMultiQueryGenerator,
    MultiQueryRetriever,
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
            qdrant_by_content[str(content)] = str(chunk_id)

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


class RecordingMultiQueryGenerator(
    BaseMultiQueryGenerator
):
    def __init__(
        self,
        generator: BaseMultiQueryGenerator,
    ) -> None:
        self.generator = generator
        self.generated_queries: dict[
            str,
            list[str],
        ] = {}

    def _generate(
        self,
        query: str,
    ) -> list[str]:
        queries = self.generator.generate(query)

        self.generated_queries[query] = queries

        return queries


@pytest.mark.asyncio
async def test_multi_query_benchmark():
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
        # Hybrid baseline
        # --------------------------------------------------

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            rrf_k=60,
        )

        hybrid_runner = DenseRetrievalEvaluationRunner(
            retriever=hybrid_retriever,
        )

        k = 5

        baseline_results = (
            await hybrid_runner.evaluate_cases(
                cases,
                k=k,
            )
        )

        baseline_metrics = (
            hybrid_runner.aggregate_results(
                baseline_results,
            )
        )

        # --------------------------------------------------
        # Multi-query retriever
        # --------------------------------------------------

        gemini_generator = GeminiMultiQueryGenerator(
            client=gemini_client,
        )

        recording_generator = RecordingMultiQueryGenerator(
            generator=gemini_generator,
        )

        multi_query_retriever = MultiQueryRetriever(
            retriever=hybrid_retriever,
            query_generator=recording_generator,
            include_original=True,
        )

        multi_query_runner = (
            DenseRetrievalEvaluationRunner(
                retriever=multi_query_retriever,
            )
        )

        multi_query_results = (
            await multi_query_runner.evaluate_cases(
                cases,
                k=k,
            )
        )

        multi_query_metrics = (
            multi_query_runner.aggregate_results(
                multi_query_results,
            )
        )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        print("\n" + "=" * 72)
        print("MULTI-QUERY RETRIEVAL BENCHMARK")
        print("=" * 72)

        print(
            f"\nEvaluated queries: "
            f"{baseline_metrics.evaluated_queries}"
        )

        print(
            "\nBaseline Hybrid / RRF k=60:"
            f"\n  Recall@5:    "
            f"{baseline_metrics.recall_at_k:.4f}"
            f"\n  Precision@5: "
            f"{baseline_metrics.precision_at_k:.4f}"
            f"\n  MRR:         "
            f"{baseline_metrics.mrr:.4f}"
        )

        print(
            "\nMulti-Query + Hybrid:"
            f"\n  Recall@5:    "
            f"{multi_query_metrics.recall_at_k:.4f}"
            f"\n  Precision@5: "
            f"{multi_query_metrics.precision_at_k:.4f}"
            f"\n  MRR:         "
            f"{multi_query_metrics.mrr:.4f}"
        )

        # --------------------------------------------------
        # Metric deltas
        # --------------------------------------------------

        recall_delta = (
            multi_query_metrics.recall_at_k
            - baseline_metrics.recall_at_k
        )

        precision_delta = (
            multi_query_metrics.precision_at_k
            - baseline_metrics.precision_at_k
        )

        mrr_delta = (
            multi_query_metrics.mrr
            - baseline_metrics.mrr
        )

        print("\nMetric delta:")

        print(
            f"  Recall@5:    {recall_delta:+.4f}"
        )

        print(
            f"  Precision@5: {precision_delta:+.4f}"
        )

        print(
            f"  MRR:         {mrr_delta:+.4f}"
        )

        # --------------------------------------------------
        # Affected queries only
        # --------------------------------------------------

        print("\n" + "=" * 72)
        print("QUERIES AFFECTED BY MULTI-QUERY RETRIEVAL")
        print("=" * 72)

        affected_queries = 0

        for index, case in enumerate(cases):
            baseline = baseline_results[index]
            multi_query = multi_query_results[index]

            baseline_metrics_tuple = (
                baseline.recall_at_k,
                baseline.precision_at_k,
                baseline.mrr,
            )

            multi_query_metrics_tuple = (
                multi_query.recall_at_k,
                multi_query.precision_at_k,
                multi_query.mrr,
            )

            if (
                baseline_metrics_tuple
                == multi_query_metrics_tuple
            ):
                continue

            affected_queries += 1

            generated_queries = (
                recording_generator.generated_queries.get(
                    case.query,
                    [],
                )
            )

            print(f"\nOriginal: {case.query}")

            print("Generated queries:")

            for generated_query in generated_queries:
                print(f"  - {generated_query}")

            print(
                f"  Hybrid → "
                f"R={baseline.recall_at_k:.2f} "
                f"P={baseline.precision_at_k:.2f} "
                f"MRR={baseline.mrr:.2f}"
            )

            print(
                f"  Multi   → "
                f"R={multi_query.recall_at_k:.2f} "
                f"P={multi_query.precision_at_k:.2f} "
                f"MRR={multi_query.mrr:.2f}"
            )

        print(
            f"\nAffected queries: "
            f"{affected_queries}/{len(cases)}"
        )

        print("\n" + "=" * 72)

        # --------------------------------------------------
        # Sanity checks
        # --------------------------------------------------

        assert len(baseline_results) == len(cases)
        assert len(multi_query_results) == len(cases)

        assert (
            len(recording_generator.generated_queries)
            == len(cases)
        )

    finally:
        await client.close()

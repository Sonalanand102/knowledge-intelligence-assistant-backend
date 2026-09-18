from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

from backend.app.embeddings.gemini import GeminiEmbeddingProvider
from backend.app.embeddings.local import LocalEmbeddingProvider
from backend.app.embeddings.service import EmbeddingService
from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_cases,
    get_embedding_evaluation_corpus,
)
from backend.app.evaluation.runner import EmbeddingEvaluationRunner
from backend.app.ingestion.models.chunk_document import ChunkDocument


K = 5


def build_chunks() -> tuple[
    list[ChunkDocument],
    dict[str, str],
]:
    """
    Build ChunkDocuments from the evaluation corpus.

    Returns:
        chunks:
            ChunkDocuments used for embedding evaluation.

        chunk_id_mapping:
            Maps the human-readable golden dataset ID
            to the deterministic ChunkDocument.chunk_id.
    """
    corpus = get_embedding_evaluation_corpus()

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

    return chunks, chunk_id_mapping


def resolve_evaluation_cases(
    cases,
    chunk_id_mapping: dict[str, str],
):
    """
    Convert human-readable golden dataset chunk IDs
    into the actual ChunkDocument.chunk_id values.
    """
    return [
        type(case)(
            query=case.query,
            relevant_chunk_ids={
                chunk_id_mapping[chunk_id]
                for chunk_id in case.relevant_chunk_ids
            },
        )
        for case in cases
    ]


def evaluate_provider(
    provider_name: str,
    embedding_service: EmbeddingService,
    cases,
    chunks: list[ChunkDocument],
):
    """
    Evaluate one embedding provider against the golden dataset.
    """
    print(f"\n{'=' * 60}")
    print(provider_name)
    print(f"{'=' * 60}")

    runner = EmbeddingEvaluationRunner(
        embedding_service=embedding_service,
    )

    results = runner.evaluate_cases(
        cases=cases,
        chunks=chunks,
        k=K,
    )

    aggregate = runner.aggregate_results(results)

    print(f"Recall@{K}:    {aggregate.recall_at_k:.4f}")
    print(f"Precision@{K}: {aggregate.precision_at_k:.4f}")
    print(f"MRR:           {aggregate.mrr:.4f}")
    print(f"Queries:       {aggregate.evaluated_queries}")

    return aggregate


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set"
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    cases = get_embedding_evaluation_cases()

    chunks, chunk_id_mapping = build_chunks()

    resolved_cases = resolve_evaluation_cases(
        cases=cases,
        chunk_id_mapping=chunk_id_mapping,
    )

    print("\nEmbedding Evaluation")
    print("-" * 60)
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Evaluation queries: {len(resolved_cases)}")
    print(f"K: {K}")

    # ---------------------------------------------------------
    # Gemini Embeddings
    # ---------------------------------------------------------

    gemini_client = genai.Client(
        api_key=api_key,
    )

    gemini_provider = GeminiEmbeddingProvider(
        client=gemini_client,
    )

    gemini_service = EmbeddingService(
        provider=gemini_provider,
    )

    gemini_result = evaluate_provider(
        provider_name="Gemini Embedding",
        embedding_service=gemini_service,
        cases=resolved_cases,
        chunks=chunks,
    )

    # ---------------------------------------------------------
    # Local MiniLM Embeddings
    # ---------------------------------------------------------

    local_provider = LocalEmbeddingProvider()

    local_service = EmbeddingService(
        provider=local_provider,
    )

    local_result = evaluate_provider(
        provider_name="MiniLM Embedding",
        embedding_service=local_service,
        cases=resolved_cases,
        chunks=chunks,
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    print(
        f"{'Metric':<15}"
        f"{'Gemini':>15}"
        f"{'MiniLM':>15}"
    )

    print("-" * 45)

    print(
        f"{'Recall@5':<15}"
        f"{gemini_result.recall_at_k:>15.4f}"
        f"{local_result.recall_at_k:>15.4f}"
    )

    print(
        f"{'Precision@5':<15}"
        f"{gemini_result.precision_at_k:>15.4f}"
        f"{local_result.precision_at_k:>15.4f}"
    )

    print(
        f"{'MRR':<15}"
        f"{gemini_result.mrr:>15.4f}"
        f"{local_result.mrr:>15.4f}"
    )


if __name__ == "__main__":
    main()
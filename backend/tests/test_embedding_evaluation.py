import pytest

from backend.app.embeddings.evaluation import (
    EmbeddingEvaluationCase,
    EmbeddingEvaluator,
)
from backend.app.embeddings.service import EmbeddedChunk
from backend.app.ingestion.models.chunk_document import ChunkDocument

def make_chunk(
    chunk_id: str,
    content: str,
    chunk_index: int,
) -> ChunkDocument:
    return ChunkDocument(
        content=content,
        document_id=chunk_id,
        chunk_index=chunk_index,
    )

def test_cosine_similarity():
    evaluator = EmbeddingEvaluator()

    similarity = evaluator.cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    evaluator = EmbeddingEvaluator()

    similarity = evaluator.cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert similarity == pytest.approx(0.0)


def test_rank_chunks_by_similarity():
    evaluator = EmbeddingEvaluator()

    query_embedding = [1.0, 0.0]

    chunk_embeddings = {
        "chunk_1": [0.2, 0.8],
        "chunk_2": [0.9, 0.1],
        "chunk_3": [1.0, 0.0],
    }

    ranked = evaluator.rank_chunks(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
    )

    assert ranked == [
        "chunk_3",
        "chunk_2",
        "chunk_1",
    ]


def test_recall_at_k_when_relevant_chunk_is_retrieved():
    evaluator = EmbeddingEvaluator()

    ranked_chunk_ids = [
        "chunk_3",
        "chunk_2",
        "chunk_1",
    ]

    relevant_chunk_ids = {
        "chunk_2",
    }

    recall = evaluator.recall_at_k(
        ranked_chunk_ids=ranked_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
        k=2,
    )

    assert recall == pytest.approx(1.0)


def test_recall_at_k_when_relevant_chunk_is_not_retrieved():
    evaluator = EmbeddingEvaluator()

    ranked_chunk_ids = [
        "chunk_3",
        "chunk_2",
        "chunk_1",
    ]

    relevant_chunk_ids = {
        "chunk_1",
    }

    recall = evaluator.recall_at_k(
        ranked_chunk_ids=ranked_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
        k=2,
    )

    assert recall == pytest.approx(0.0)


def test_precision_at_k():
    evaluator = EmbeddingEvaluator()

    ranked_chunk_ids = [
        "chunk_2",
        "chunk_3",
        "chunk_1",
    ]

    relevant_chunk_ids = {
        "chunk_2",
        "chunk_1",
    }

    precision = evaluator.precision_at_k(
        ranked_chunk_ids=ranked_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
        k=2,
    )

    assert precision == pytest.approx(0.5)


def test_mrr_when_first_relevant_result_is_second():
    evaluator = EmbeddingEvaluator()

    ranked_chunk_ids = [
        "chunk_3",
        "chunk_2",
        "chunk_1",
    ]

    relevant_chunk_ids = {
        "chunk_2",
    }

    mrr = evaluator.mrr(
        ranked_chunk_ids=ranked_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
    )

    assert mrr == pytest.approx(0.5)


def test_mrr_when_no_relevant_result_exists():
    evaluator = EmbeddingEvaluator()

    ranked_chunk_ids = [
        "chunk_3",
        "chunk_2",
        "chunk_1",
    ]

    relevant_chunk_ids = {
        "chunk_99",
    }

    mrr = evaluator.mrr(
        ranked_chunk_ids=ranked_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
    )

    assert mrr == pytest.approx(0.0)


def test_evaluation_case_requires_query_and_relevant_chunks():
    case = EmbeddingEvaluationCase(
        query="What is FastAPI?",
        relevant_chunk_ids={"chunk_1", "chunk_2"},
    )

    assert case.query == "What is FastAPI?"
    assert case.relevant_chunk_ids == {"chunk_1", "chunk_2"}

def test_rank_embedded_chunks_by_similarity():
    evaluator = EmbeddingEvaluator()

    chunk_1 = ChunkDocument(
        content="Some unrelated content",
        document_id="doc_1",
        chunk_index=0,
    )

    chunk_2 = ChunkDocument(
        content="FastAPI is a Python web framework",
        document_id="doc_1",
        chunk_index=1,
    )

    chunk_3 = ChunkDocument(
        content="FastAPI provides API functionality",
        document_id="doc_1",
        chunk_index=2,
    )

    chunks = [
        EmbeddedChunk(
            chunk=chunk_1,
            embedding=[0.2, 0.8],
        ),
        EmbeddedChunk(
            chunk=chunk_2,
            embedding=[0.9, 0.1],
        ),
        EmbeddedChunk(
            chunk=chunk_3,
            embedding=[1.0, 0.0],
        ),
    ]

    ranked = evaluator.rank_embedded_chunks(
        query_embedding=[1.0, 0.0],
        embedded_chunks=chunks,
    )

    assert ranked == [
        chunk_3.chunk_id,
        chunk_2.chunk_id,
        chunk_1.chunk_id,
    ]

def test_evaluate_case_using_embedded_chunks():
    evaluator = EmbeddingEvaluator()

    chunk_1 = ChunkDocument(
        content="Unrelated content",
        document_id="doc_1",
        chunk_index=0,
    )

    chunk_2 = ChunkDocument(
        content="FastAPI is a Python web framework",
        document_id="doc_1",
        chunk_index=1,
    )

    chunks = [
        EmbeddedChunk(
            chunk=chunk_1,
            embedding=[0.0, 1.0],
        ),
        EmbeddedChunk(
            chunk=chunk_2,
            embedding=[1.0, 0.0],
        ),
    ]

    case = EmbeddingEvaluationCase(
        query="What is FastAPI?",
        relevant_chunk_ids={chunk_2.chunk_id},
    )

    result = evaluator.evaluate_case(
        case=case,
        query_embedding=[1.0, 0.0],
        embedded_chunks=chunks,
        k=1,
    )

    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 1.0
    assert result.mrr == 1.0
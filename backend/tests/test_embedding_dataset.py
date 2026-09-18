from backend.app.evaluation.datasets.embedding_dataset import (
    EmbeddingEvaluationCase,
    get_embedding_evaluation_cases,
)


def test_embedding_evaluation_dataset_contains_cases():
    cases = get_embedding_evaluation_cases()

    assert cases
    assert all(
        isinstance(case, EmbeddingEvaluationCase)
        for case in cases
    )


def test_embedding_evaluation_case_has_query_and_relevant_chunks():
    cases = get_embedding_evaluation_cases()

    for case in cases:
        assert case.query.strip()
        assert case.relevant_chunk_ids


def test_embedding_evaluation_queries_are_unique():
    cases = get_embedding_evaluation_cases()

    queries = [case.query for case in cases]

    assert len(queries) == len(set(queries))


def test_embedding_evaluation_has_multiple_topics():
    cases = get_embedding_evaluation_cases()

    assert len(cases) >= 10
from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_cases,
    get_embedding_evaluation_corpus,
)


def test_evaluation_dataset_is_ready_for_real_evaluation():
    cases = get_embedding_evaluation_cases()
    corpus = get_embedding_evaluation_corpus()

    assert len(cases) >= 10
    assert len(corpus) >= 30

    corpus_ids = set(corpus)

    for case in cases:
        assert case.relevant_chunk_ids <= corpus_ids
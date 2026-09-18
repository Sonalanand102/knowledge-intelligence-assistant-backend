from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_cases,
    get_embedding_evaluation_corpus,
)


def test_embedding_corpus_contains_chunks():
    corpus = get_embedding_evaluation_corpus()

    assert corpus
    assert all(
        isinstance(chunk_id, str)
        and isinstance(content, str)
        for chunk_id, content in corpus.items()
    )


def test_embedding_corpus_chunk_ids_are_unique():
    corpus = get_embedding_evaluation_corpus()

    assert len(corpus) == len(set(corpus.keys()))


def test_every_relevant_chunk_exists_in_corpus():
    corpus = get_embedding_evaluation_corpus()
    cases = get_embedding_evaluation_cases()

    corpus_ids = set(corpus.keys())

    for case in cases:
        assert case.relevant_chunk_ids <= corpus_ids


def test_corpus_contains_enough_distractor_chunks():
    corpus = get_embedding_evaluation_corpus()

    assert len(corpus) >= 30


def test_corpus_chunks_have_meaningful_content():
    corpus = get_embedding_evaluation_corpus()

    for content in corpus.values():
        assert len(content.strip()) >= 50
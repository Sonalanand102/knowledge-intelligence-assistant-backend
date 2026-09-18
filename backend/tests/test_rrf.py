from __future__ import annotations

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.rrf import (
    reciprocal_rank_fusion,
)


def make_result(
    chunk_id: str,
    score: float = 1.0,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        content=f"Content for {chunk_id}",
        score=score,
        metadata={},
    )


def test_rrf_fuses_ranked_lists():
    dense_results = [
        make_result("A"),
        make_result("B"),
        make_result("C"),
    ]

    sparse_results = [
        make_result("B"),
        make_result("A"),
        make_result("D"),
    ]

    results = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        top_k=4,
        rrf_k=60,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        "A",
        "B",
        "C",
        "D",
    ]

    assert results[0].score == pytest.approx(
        (1 / 61) + (1 / 62)
    )

    assert results[1].score == pytest.approx(
        (1 / 62) + (1 / 61)
    )


def test_rrf_deduplicates_same_chunk():
    dense_results = [
        make_result("A"),
        make_result("B"),
    ]

    sparse_results = [
        make_result("A"),
        make_result("C"),
    ]

    results = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        top_k=10,
    )

    chunk_ids = [
        result.chunk_id
        for result in results
    ]

    assert chunk_ids.count("A") == 1
    assert set(chunk_ids) == {
        "A",
        "B",
        "C",
    }


def test_rrf_respects_top_k():
    results = reciprocal_rank_fusion(
        [
            [
                make_result("A"),
                make_result("B"),
                make_result("C"),
                make_result("D"),
            ]
        ],
        top_k=2,
    )

    assert len(results) == 2

    assert [
        result.chunk_id
        for result in results
    ] == ["A", "B"]


def test_rrf_empty_input_returns_empty():
    results = reciprocal_rank_fusion(
        [],
        top_k=5,
    )

    assert results == []


def test_rrf_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        reciprocal_rank_fusion(
            [[]],
            top_k=0,
        )


def test_rrf_rejects_invalid_rrf_k():
    with pytest.raises(
        ValueError,
        match="rrf_k must be greater than zero",
    ):
        reciprocal_rank_fusion(
            [[]],
            rrf_k=0,
        )
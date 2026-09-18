from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

from backend.app.retrieval.base import VectorSearchResult


def reciprocal_rank_fusion(
    result_lists: Sequence[
        Sequence[VectorSearchResult]
    ],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[VectorSearchResult]:
    """
    Fuse multiple ranked result lists using
    Reciprocal Rank Fusion (RRF).

    RRF score:

        score(d) = sum(1 / (rrf_k + rank))

    where rank starts at 1.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero"
        )

    if rrf_k <= 0:
        raise ValueError(
            "rrf_k must be greater than zero"
        )

    if not result_lists:
        return []

    fused_scores: dict[str, float] = {}
    representatives: dict[
        str,
        VectorSearchResult,
    ] = {}

    for results in result_lists:
        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = result.chunk_id

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

            if chunk_id not in representatives:
                representatives[chunk_id] = result

    fused_results = [
        replace(
            representatives[chunk_id],
            score=score,
        )
        for chunk_id, score in fused_scores.items()
    ]

    fused_results.sort(
        key=lambda result: (
            -result.score,
            result.chunk_id,
        )
    )

    return fused_results[:top_k]
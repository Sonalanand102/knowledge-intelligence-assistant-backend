from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.dependencies import get_search_service
from backend.app.retrieval.search_service import SearchService
from backend.app.schemas.search import (
    SearchResponse,
    SearchResultResponse,
)


router = APIRouter(
    prefix="/search",
    tags=["search"],
)


@router.get(
    "",
    response_model=SearchResponse,
)
async def search(
    q: str = Query(
        min_length=1,
        description="Natural-language search query",
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return",
    ),
    search_service: SearchService = Depends(
        get_search_service,
    ),
) -> SearchResponse:
    query = q.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty",
        )

    result = await search_service.search(
        query=query,
        top_k=top_k,
    )

    response_results = [
        SearchResultResponse(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            chunk_index=item.chunk_index,
            content=item.content,
            score=item.score,
            metadata=item.metadata,
        )
        for item in result.results
    ]

    return SearchResponse(
        query=result.query,
        results=response_results,
    )
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.dependencies import (
    get_answer_service,
)
from backend.app.generation.answer_service import (
    AnswerService,
)
from backend.app.schemas.ask import (
    AskRequest,
    AskResponse,
    CitationResponse,
)


router = APIRouter(
    prefix="/ask",
    tags=["ask"],
)


@router.post(
    "",
    response_model=AskResponse,
)
async def ask(
    request: AskRequest,
    answer_service: AnswerService = Depends(
        get_answer_service,
    ),
) -> AskResponse:
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty",
        )

    result = await answer_service.answer(
        query=query,
        top_k=request.top_k,
    )

    citations = [
        CitationResponse(
            citation_id=citation.citation_id,
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            metadata=citation.metadata,
        )
        for citation in result.citations
    ]

    return AskResponse(
        query=query,
        answer=result.answer,
        citations=citations,
    )

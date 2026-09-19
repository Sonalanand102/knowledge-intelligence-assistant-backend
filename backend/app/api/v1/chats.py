from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from backend.app.api.dependencies import (
    get_chat_answer_service,
    get_chat_search_service,
    get_chat_service,
)

from backend.app.schemas.ask import (
    AskRequest,
    ChatAskResponse,
    CitationResponse,
)

from backend.app.schemas.chat import (
    ChatListResponse,
    ChatResponse,
    CreateChatRequest,
)

from backend.app.schemas.search import (
    ChatSearchResponse,
    SearchResultResponse,
)

from backend.app.services.chat_answer_service import (
    ChatAnswerService,
    ChatNotFoundError as ChatAnswerNotFoundError,
)

from backend.app.services.chat_search_service import (
    ChatSearchService,
    ChatNotFoundError as ChatSearchNotFoundError,
)

from backend.app.services.chat_service import (
    ChatNotFoundError as ChatServiceNotFoundError,
    ChatService,
)


router = APIRouter(
    prefix="/chats",
    tags=["chats"],
)


# ============================================================
# CREATE CHAT
# ============================================================

@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chat(
    request: CreateChatRequest,
    chat_service: ChatService = Depends(
        get_chat_service
    ),
) -> ChatResponse:
    return await chat_service.create_chat(
        request
    )


# ============================================================
# LIST CHATS
# ============================================================

@router.get(
    "",
    response_model=ChatListResponse,
)
async def list_chats(
    chat_service: ChatService = Depends(
        get_chat_service
    ),
) -> ChatListResponse:
    return await chat_service.list_chats()


# ============================================================
# CHAT-SCOPED SEARCH
# ============================================================

@router.get(
    "/{chat_id}/search",
    response_model=ChatSearchResponse,
)
async def search_chat(
    chat_id: str,
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
    chat_search_service: ChatSearchService = Depends(
        get_chat_search_service,
    ),
) -> ChatSearchResponse:
    query = q.strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty",
        )

    try:
        result = await chat_search_service.search(
            chat_id=chat_id,
            query=query,
            top_k=top_k,
        )

    except ChatSearchNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return ChatSearchResponse(
        chat_id=chat_id,
        query=result.query,
        results=[
            SearchResultResponse(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                chunk_index=item.chunk_index,
                content=item.content,
                score=item.score,
                metadata=item.metadata,
            )
            for item in result.results
        ],
    )


# ============================================================
# CHAT-SCOPED ASK
# ============================================================

@router.post(
    "/{chat_id}/ask",
    response_model=ChatAskResponse,
)
async def ask_chat(
    chat_id: str,
    request: AskRequest,
    chat_answer_service: ChatAnswerService = Depends(
        get_chat_answer_service,
    ),
) -> ChatAskResponse:
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty",
        )

    try:
        result = await chat_answer_service.answer(
            chat_id=chat_id,
            query=query,
            top_k=request.top_k,
        )

    except ChatAnswerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return ChatAskResponse(
        chat_id=chat_id,
        query=query,
        answer=result.answer,
        citations=[
            CitationResponse(
                citation_id=citation.citation_id,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                metadata=citation.metadata,
            )
            for citation in result.citations
        ],
    )


# ============================================================
# GET CHAT
# ============================================================

@router.get(
    "/{chat_id}",
    response_model=ChatResponse,
)
async def get_chat(
    chat_id: str,
    chat_service: ChatService = Depends(
        get_chat_service
    ),
) -> ChatResponse:
    try:
        return await chat_service.get_chat(
            chat_id
        )

    except ChatServiceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
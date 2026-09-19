from __future__ import annotations

from fastapi import Request

from backend.app.generation.answer_service import (
    AnswerService,
)
from backend.app.retrieval.search_service import (
    SearchService,
)

from backend.app.services.document_upload_service import (
    DocumentUploadService,
)

from backend.app.services.chat_service import ChatService

from backend.app.services.chat_search_service import (
    ChatSearchService,
)

from backend.app.services.chat_answer_service import (
    ChatAnswerService,
)

def get_search_service(
    request: Request,
) -> SearchService:
    return request.app.state.search_service


def get_answer_service(
    request: Request,
) -> AnswerService:
    return request.app.state.answer_service

def get_document_upload_service(
    request: Request,
) -> DocumentUploadService:
    return request.app.state.document_upload_service

def get_chat_service() -> ChatService:
    return ChatService()

def get_chat_search_service(
    request: Request,
) -> ChatSearchService:
    return ChatSearchService(
        search_service=request.app.state.search_service,
    )

def get_chat_answer_service(
    request: Request,
) -> ChatAnswerService:
    return ChatAnswerService(
        answer_service=request.app.state.answer_service,
    )
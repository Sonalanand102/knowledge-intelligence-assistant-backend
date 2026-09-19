from __future__ import annotations

import uuid

from sqlalchemy import select

from backend.app.db.models.chat import Chat
from backend.app.db.models.chat_document import ChatDocument
from backend.app.db.session import AsyncSessionLocal
from backend.app.retrieval.search_service import (
    SearchResponse,
    SearchService,
)


class ChatNotFoundError(ValueError):
    pass


class ChatSearchService:
    """
    Resolves the documents attached to a chat and delegates
    retrieval to the existing SearchService.
    """

    def __init__(
        self,
        search_service: SearchService,
    ) -> None:
        self.search_service = search_service

    async def search(
        self,
        chat_id: str,
        query: str,
        top_k: int = 5,
    ) -> SearchResponse:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        try:
            parsed_chat_id = uuid.UUID(chat_id)
        except ValueError as exc:
            raise ChatNotFoundError(
                f"Chat not found: {chat_id}"
            ) from exc

        async with AsyncSessionLocal() as session:
            chat = await session.scalar(
                select(Chat).where(
                    Chat.id == parsed_chat_id
                )
            )

            if chat is None:
                raise ChatNotFoundError(
                    f"Chat not found: {chat_id}"
                )

            document_ids = list(
                (
                    await session.scalars(
                        select(
                            ChatDocument.document_id
                        ).where(
                            ChatDocument.chat_id
                            == parsed_chat_id
                        )
                    )
                ).all()
            )

        # A valid chat with no documents simply has
        # no searchable context yet.
        if not document_ids:
            return SearchResponse(
                query=query,
                results=[],
            )

        return await self.search_service.search(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )
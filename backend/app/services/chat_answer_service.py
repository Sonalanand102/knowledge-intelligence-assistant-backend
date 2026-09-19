from __future__ import annotations

import uuid

from sqlalchemy import select

from backend.app.db.models.chat import Chat
from backend.app.db.models.chat_document import ChatDocument
from backend.app.db.session import AsyncSessionLocal
from backend.app.generation.answer_generation import (
    GeneratedAnswer,
)
from backend.app.generation.answer_service import (
    AnswerService,
)


class ChatNotFoundError(ValueError):
    pass


class ChatAnswerService:
    """
    Generates answers using only documents attached to a chat.
    """

    def __init__(
        self,
        answer_service: AnswerService,
    ) -> None:
        self.answer_service = answer_service

    async def answer(
        self,
        chat_id: str,
        query: str,
        top_k: int = 5,
    ) -> GeneratedAnswer:
        if not query or not query.strip():
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

        if not document_ids:
            return GeneratedAnswer(
                answer=(
                    "I could not find enough relevant information "
                    "to answer the question."
                ),
                citations=[],
            )

        return await self.answer_service.answer(
            query=query.strip(),
            top_k=top_k,
            document_ids=document_ids,
        )
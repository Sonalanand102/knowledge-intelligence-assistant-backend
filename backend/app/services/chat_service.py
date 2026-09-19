from __future__ import annotations

import uuid

from sqlalchemy import select

from backend.app.db.models.chat import Chat
from backend.app.db.session import AsyncSessionLocal
from backend.app.schemas.chat import (
    ChatListResponse,
    ChatResponse,
    CreateChatRequest,
)


class ChatNotFoundError(ValueError):
    """Raised when a chat does not exist."""


class ChatService:
    async def create_chat(
        self,
        request: CreateChatRequest,
    ) -> ChatResponse:
        async with AsyncSessionLocal() as session:
            chat = Chat(
                title=request.title.strip(),
            )

            session.add(chat)

            await session.commit()
            await session.refresh(chat)

            return self._to_response(chat)

    async def list_chats(self) -> ChatListResponse:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Chat).order_by(
                    Chat.updated_at.desc()
                )
            )

            chats = result.scalars().all()

            return ChatListResponse(
                chats=[
                    self._to_response(chat)
                    for chat in chats
                ]
            )

    async def get_chat(
        self,
        chat_id: str,
    ) -> ChatResponse:
        try:
            parsed_chat_id = uuid.UUID(chat_id)
        except ValueError as exc:
            raise ChatNotFoundError(
                f"Chat not found: {chat_id}"
            ) from exc

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Chat).where(
                    Chat.id == parsed_chat_id
                )
            )

            chat = result.scalar_one_or_none()

            if chat is None:
                raise ChatNotFoundError(
                    f"Chat not found: {chat_id}"
                )

            return self._to_response(chat)

    @staticmethod
    def _to_response(chat: Chat) -> ChatResponse:
        return ChatResponse(
            chat_id=chat.id,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
        )
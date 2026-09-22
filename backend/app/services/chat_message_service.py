from __future__ import annotations

import uuid

from sqlalchemy import func, select

from backend.app.db.models.chat import Chat
from backend.app.db.models.chat_message import ChatMessage
from backend.app.db.session import AsyncSessionLocal
from backend.app.schemas.chat_message import (
    ChatMessageListResponse,
    ChatMessageResponse,
)


class ChatMessageService:
    """
    Handles chat message persistence and history retrieval.
    """

    ALLOWED_ROLES = {
        "user",
        "assistant",
    }

    async def create_message(
        self,
        *,
        chat_id: str,
        role: str,
        content: str,
        citations: list[dict[str, object]] | None = None,
    ) -> ChatMessageResponse:
        if not content or not content.strip():
            raise ValueError(
                "Message content cannot be empty"
            )

        normalized_role = role.strip().lower()

        if normalized_role not in self.ALLOWED_ROLES:
            raise ValueError(
                "Invalid message role: "
                f"{role}"
            )

        try:
            parsed_chat_id = uuid.UUID(chat_id)
        except ValueError as exc:
            raise ValueError(
                f"Chat not found: {chat_id}"
            ) from exc

        async with AsyncSessionLocal() as session:
            chat = await session.scalar(
                select(Chat).where(
                    Chat.id == parsed_chat_id
                )
            )

            if chat is None:
                raise ValueError(
                    f"Chat not found: {chat_id}"
                )

            message = ChatMessage(
                chat_id=parsed_chat_id,
                role=normalized_role,
                content=content.strip(),
                citations=citations,
            )

            session.add(message)

            # A new message means the chat was updated.
            chat.updated_at = func.now()

            await session.commit()
            await session.refresh(message)

            return self._to_response(message)

    async def list_messages(
        self,
        *,
        chat_id: str,
    ) -> ChatMessageListResponse:
        try:
            parsed_chat_id = uuid.UUID(chat_id)
        except ValueError as exc:
            raise ValueError(
                f"Chat not found: {chat_id}"
            ) from exc

        async with AsyncSessionLocal() as session:
            chat = await session.scalar(
                select(Chat).where(
                    Chat.id == parsed_chat_id
                )
            )

            if chat is None:
                raise ValueError(
                    f"Chat not found: {chat_id}"
                )

            result = await session.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.chat_id
                    == parsed_chat_id
                )
                .order_by(
                    ChatMessage.created_at.asc(),
                    ChatMessage.id.asc(),
                )
            )

            messages = result.scalars().all()

            return ChatMessageListResponse(
                messages=[
                    self._to_response(message)
                    for message in messages
                ]
            )

    @staticmethod
    def _to_response(
        message: ChatMessage,
    ) -> ChatMessageResponse:
        return ChatMessageResponse(
            message_id=message.id,
            chat_id=message.chat_id,
            role=message.role,
            content=message.content,
            citations=(
                list(message.citations)
                if message.citations
                else []
            ),
            created_at=message.created_at,
        )
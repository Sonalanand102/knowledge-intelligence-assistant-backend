from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CreateChatRequest(BaseModel):
    title: str = Field(
        default="New Chat",
        min_length=1,
        max_length=255,
    )


class ChatResponse(BaseModel):
    chat_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ChatListResponse(BaseModel):
    chats: list[ChatResponse] = Field(
        default_factory=list
    )
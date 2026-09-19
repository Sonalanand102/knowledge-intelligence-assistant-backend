from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class ChatDocument(Base):
    """
    Associates a chat with a source document.

    The association is kept in a separate table so that the same document
    can potentially be attached to multiple chats in the future.
    """

    __tablename__ = "chat_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "chats.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "source_documents.document_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "document_id",
            name="uq_chat_documents_chat_document",
        ),
        Index(
            "ix_chat_documents_document_id",
            "document_id",
        ),
    )
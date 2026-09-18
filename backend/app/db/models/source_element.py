from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class SourceElement(Base):
    __tablename__ = "source_elements"

    element_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey(
            "source_documents.document_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    element_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    content_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    text_content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    asset_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
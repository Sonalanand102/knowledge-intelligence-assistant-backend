from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class ElementRelationship(Base):
    __tablename__ = "element_relationships"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    source_element_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey(
            "source_elements.element_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    relationship_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    target_element_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey(
            "source_elements.element_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
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

    __table_args__ = (
        UniqueConstraint(
            "source_element_id",
            "relationship_type",
            "target_element_id",
            name="uq_element_relationship",
        ),
    )
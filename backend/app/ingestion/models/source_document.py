from dataclasses import dataclass, field
from typing import Any

from backend.app.ingestion.models.content import (
    AudioContent,
    ImageContent,
    TableContent,
    TextContent,
    VideoContent,
)


Content = (
    TextContent
    | ImageContent
    | AudioContent
    | VideoContent
    | TableContent
)


@dataclass
class SourceDocument:
    document_id: str
    source_type: str
    content: Content
    metadata: dict[str, Any] = field(default_factory=dict)
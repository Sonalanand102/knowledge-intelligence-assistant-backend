from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib


@dataclass
class ChunkDocument:
    content: str
    document_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        raw_id = f"{self.document_id}:{self.chunk_index}"

        return hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()
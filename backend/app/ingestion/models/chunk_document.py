from dataclasses import dataclass, field
from typing import Any

@dataclass
class ChunkDocument:
    content: str
    document_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
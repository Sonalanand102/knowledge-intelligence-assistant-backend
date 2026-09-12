from dataclasses import dataclass, field
from typing import Any


@dataclass
class ElementRelationship:
    source_element_id: str
    relationship_type: str
    target_element_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
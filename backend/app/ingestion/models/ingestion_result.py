from dataclasses import dataclass, field

from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.source_element import SourceElement


@dataclass
class IngestionResult:
    document_id: str
    elements: list[SourceElement] = field(default_factory=list)
    relationships: list[ElementRelationship] = field(
        default_factory=list
    )
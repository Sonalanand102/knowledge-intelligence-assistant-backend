from backend.app.ingestion.models.content import (
    AudioContent,
    ImageContent,
    TableContent,
    TextContent,
    VideoContent,
)
from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.models.source_document import (
    SourceDocument,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)


def source_document_to_element(
    document: SourceDocument,
    element_id: str,
    element_type: str,
) -> SourceElement:
    return SourceElement(
        element_id=element_id,
        document_id=document.document_id,
        element_type=element_type,
        content=document.content,
        metadata=document.metadata.copy(),
    )


def wrap_source_document(
    document: SourceDocument,
    element_type: str,
) -> IngestionResult:
    element = source_document_to_element(
        document=document,
        element_id=f"{element_type}-1",
        element_type=element_type,
    )

    root = SourceElement(
        element_id="document",
        document_id=document.document_id,
        element_type="document",
        content=TextContent(text=""),
        metadata=document.metadata.copy(),
    )

    relationship = ElementRelationship(
        source_element_id="document",
        relationship_type="contains",
        target_element_id=element.element_id,
    )

    return IngestionResult(
        document_id=document.document_id,
        elements=[
            root,
            element,
        ],
        relationships=[
            relationship,
        ],
    )
from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)
from backend.app.ingestion.relationships.temporal import (
    resolve_temporal_relationships,
)


def test_resolve_temporal_relationships():
    transcript = SourceElement(
        element_id="transcript-1",
        document_id="video-1",
        element_type="transcript",
        content=TextContent(
            text="RAG combines retrieval and generation."
        ),
        metadata={
            "content_type": "transcript",
            "start_seconds": 10.0,
            "end_seconds": 15.0,
        },
    )

    frame = SourceElement(
        element_id="frame-1",
        document_id="video-1",
        element_type="video_frame",
        content=ImageContent(
            path="frame-1.jpg"
        ),
        metadata={
            "content_type": "video_frame",
            "timestamp_seconds": 12.0,
        },
    )

    relationships = (
        resolve_temporal_relationships(
            [
                transcript,
                frame,
            ]
        )
    )

    assert len(relationships) == 2

    assert any(
        relationship.source_element_id
        == "transcript-1"
        and relationship.target_element_id
        == "frame-1"
        and relationship.relationship_type
        == "temporally_adjacent"
        for relationship in relationships
    )

    assert any(
        relationship.source_element_id
        == "frame-1"
        and relationship.target_element_id
        == "transcript-1"
        and relationship.relationship_type
        == "temporally_adjacent"
        for relationship in relationships
    )


def test_distant_frame_is_not_related():
    transcript = SourceElement(
        element_id="transcript-1",
        document_id="video-1",
        element_type="transcript",
        content=TextContent(
            text="RAG combines retrieval and generation."
        ),
        metadata={
            "content_type": "transcript",
            "start_seconds": 10.0,
            "end_seconds": 15.0,
        },
    )

    frame = SourceElement(
        element_id="frame-1",
        document_id="video-1",
        element_type="video_frame",
        content=ImageContent(
            path="frame-1.jpg"
        ),
        metadata={
            "content_type": "video_frame",
            "timestamp_seconds": 30.0,
        },
    )

    relationships = (
        resolve_temporal_relationships(
            [
                transcript,
                frame,
            ]
        )
    )

    assert relationships == []


def test_frame_at_segment_boundary_is_related():
    transcript = SourceElement(
        element_id="transcript-1",
        document_id="video-1",
        element_type="transcript",
        content=TextContent(
            text="Hello world."
        ),
        metadata={
            "content_type": "transcript",
            "start_seconds": 10.0,
            "end_seconds": 15.0,
        },
    )

    frame = SourceElement(
        element_id="frame-1",
        document_id="video-1",
        element_type="video_frame",
        content=ImageContent(
            path="frame-1.jpg"
        ),
        metadata={
            "content_type": "video_frame",
            "timestamp_seconds": 15.0,
        },
    )

    relationships = (
        resolve_temporal_relationships(
            [
                transcript,
                frame,
            ]
        )
    )

    assert relationships
from __future__ import annotations

from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)


def _get_timestamp(
    element: SourceElement,
) -> float | None:
    timestamp = element.metadata.get(
        "timestamp_seconds"
    )

    if timestamp is None:
        return None

    return float(timestamp)


def _get_start_time(
    element: SourceElement,
) -> float | None:
    start = element.metadata.get(
        "start_seconds"
    )

    if start is None:
        return None

    return float(start)


def _get_end_time(
    element: SourceElement,
) -> float | None:
    end = element.metadata.get(
        "end_seconds"
    )

    if end is None:
        return None

    return float(end)


def _timestamp_overlaps_segment(
    timestamp: float,
    start: float,
    end: float,
    tolerance_seconds: float = 1.0,
) -> bool:
    return (
        start - tolerance_seconds
        <= timestamp
        <= end + tolerance_seconds
    )


def resolve_temporal_relationships(
    elements: list[SourceElement],
    tolerance_seconds: float = 1.0,
) -> list[ElementRelationship]:
    relationships: list[
        ElementRelationship
    ] = []

    transcript_elements = [
        element
        for element in elements
        if element.element_type == "transcript"
        or element.metadata.get(
            "content_type"
        ) == "transcript"
    ]

    frame_elements = [
        element
        for element in elements
        if element.element_type
        in {
            "video_frame",
            "frame",
        }
    ]

    for transcript in transcript_elements:
        start = _get_start_time(
            transcript
        )
        end = _get_end_time(
            transcript
        )

        if start is None or end is None:
            continue

        for frame in frame_elements:
            timestamp = _get_timestamp(
                frame
            )

            if timestamp is None:
                continue

            if not _timestamp_overlaps_segment(
                timestamp,
                start,
                end,
                tolerance_seconds,
            ):
                continue

            relationships.append(
                ElementRelationship(
                    source_element_id=(
                        transcript.element_id
                    ),
                    relationship_type=(
                        "temporally_adjacent"
                    ),
                    target_element_id=(
                        frame.element_id
                    ),
                    metadata={
                        "timestamp_seconds": timestamp,
                        "transcript_start_seconds": start,
                        "transcript_end_seconds": end,
                    },
                )
            )

            relationships.append(
                ElementRelationship(
                    source_element_id=(
                        frame.element_id
                    ),
                    relationship_type=(
                        "temporally_adjacent"
                    ),
                    target_element_id=(
                        transcript.element_id
                    ),
                    metadata={
                        "timestamp_seconds": timestamp,
                        "transcript_start_seconds": start,
                        "transcript_end_seconds": end,
                    },
                )
            )

    return relationships
from __future__ import annotations

from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)
from backend.app.ingestion.relationships.spatial import (
    get_bbox,
    is_spatially_near,
    spatial_metadata,
)


def resolve_spatial_relationships(
    elements: list[SourceElement],
) -> list[ElementRelationship]:
    relationships: list[ElementRelationship] = []

    for index, source in enumerate(elements):
        source_bbox = get_bbox(source)

        if source_bbox is None:
            continue

        for target in elements[index + 1 :]:
            target_bbox = get_bbox(target)

            if target_bbox is None:
                continue

            # Elements from different pages/slides should
            # not be considered spatially related.
            source_page = source.metadata.get(
                "page_number"
            )
            target_page = target.metadata.get(
                "page_number"
            )

            source_slide = source.metadata.get(
                "slide_number"
            )
            target_slide = target.metadata.get(
                "slide_number"
            )

            if (
                source_page is not None
                and target_page is not None
                and source_page != target_page
            ):
                continue

            if (
                source_slide is not None
                and target_slide is not None
                and source_slide != target_slide
            ):
                continue

            if not is_spatially_near(
                source_bbox,
                target_bbox,
            ):
                continue

            metadata = spatial_metadata(
                source_bbox,
                target_bbox,
            )

            relationships.append(
                ElementRelationship(
                    source_element_id=source.element_id,
                    relationship_type="spatially_adjacent",
                    target_element_id=target.element_id,
                    metadata=metadata,
                )
            )

            relationships.append(
                ElementRelationship(
                    source_element_id=target.element_id,
                    relationship_type="spatially_adjacent",
                    target_element_id=source.element_id,
                    metadata=metadata,
                )
            )

    return relationships
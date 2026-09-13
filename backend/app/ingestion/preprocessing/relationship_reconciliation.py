from __future__ import annotations

from backend.app.ingestion.models.element_relationship import ElementRelationship
from backend.app.ingestion.models.source_element import SourceElement


def reconcile_relationships(
    elements: list[SourceElement],
    relationships: list[ElementRelationship],
) -> list[ElementRelationship]:
    """
    Keep only relationships whose source and target elements still exist.

    The input lists are not modified.
    """
    valid_element_ids = {element.element_id for element in elements}

    return [
        relationship
        for relationship in relationships
        if (
            relationship.source_element_id in valid_element_ids
            and relationship.target_element_id in valid_element_ids
        )
    ]
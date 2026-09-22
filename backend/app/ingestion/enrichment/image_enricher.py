from __future__ import annotations

from pathlib import Path

from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)
from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)

from backend.app.ingestion.enrichment.base import (
    ImageSemanticProvider,
)

import logging
import time

logger = logging.getLogger(__name__)

class ImageEnricher:
    """
    Converts image SourceElements into searchable semantic
    SourceElements while preserving the original asset.
    """

    def __init__(
        self,
        provider: ImageSemanticProvider,
    ) -> None:
        self.provider = provider

    def supports(
        self,
        element: SourceElement,
    ) -> bool:
        return isinstance(
            element.content,
            ImageContent,
        )

    def enrich(
        self,
        element: SourceElement,
    ) -> tuple[
        list[SourceElement],
        list[ElementRelationship],
    ]:
        if not self.supports(element):
            return [], []

        assert isinstance(
            element.content,
            ImageContent,
        )

        image_path = Path(
            element.content.path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image asset not found: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Image asset is not a file: {image_path}"
            )

        started_at = time.perf_counter()

        logger.info(
            "[ENRICH] image started element=%s path=%s",
            element.element_id,
            image_path.name,
        )

        semantic_result = self.provider.describe_image(
            str(image_path)
        )

        logger.info(
            "[ENRICH] image completed element=%s duration=%.2fs",
            element.element_id,
            time.perf_counter() - started_at,
        )
        
        text = semantic_result.text.strip()

        if not text:
            raise ValueError(
                "Image semantic provider returned empty text"
            )

        derived_element_id = (
            f"{element.element_id}:semantic"
        )

        metadata = {
            **element.metadata,
            **semantic_result.metadata,
            "modality": "image",
            "representation_type": (
                "semantic_description"
            ),
            "derived_from": element.element_id,
            "asset_path": str(image_path),
        }

        semantic_element = SourceElement(
            element_id=derived_element_id,
            document_id=element.document_id,
            element_type="image_semantic",
            content=TextContent(
                text=text,
            ),
            metadata=metadata,
        )

        relationship = ElementRelationship(
            source_element_id=derived_element_id,
            relationship_type="derived_from",
            target_element_id=element.element_id,
            metadata={
                "derivation_type": (
                    "visual_semantic_representation"
                ),
            },
        )

        return [semantic_element], [relationship]
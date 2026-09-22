from __future__ import annotations

from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.enrichment.image_enricher import (
    ImageEnricher,
)


class EnrichmentService:
    """
    Runs modality-specific semantic enrichment.
    """

    def __init__(
        self,
        image_enricher: ImageEnricher,
    ) -> None:
        self.image_enricher = image_enricher

    def enrich(
        self,
        ingestion_result: IngestionResult,
    ) -> IngestionResult:
        enriched_elements = list(
            ingestion_result.elements
        )

        enriched_relationships = list(
            ingestion_result.relationships
        )

        for element in ingestion_result.elements:
            if self.image_enricher.supports(element):
                elements, relationships = (
                    self.image_enricher.enrich(
                        element
                    )
                )

                enriched_elements.extend(
                    elements
                )
                enriched_relationships.extend(
                    relationships
                )

        return IngestionResult(
            document_id=ingestion_result.document_id,
            elements=enriched_elements,
            relationships=enriched_relationships,
        )
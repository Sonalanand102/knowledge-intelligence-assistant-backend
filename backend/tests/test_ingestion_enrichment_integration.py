from __future__ import annotations

import uuid
from pathlib import Path
from dataclasses import dataclass

import pytest

from backend.app.ingestion.enrichment.base import (
    ImageSemanticResult,
)
from backend.app.ingestion.enrichment.image_enricher import (
    ImageEnricher,
)
from backend.app.ingestion.enrichment.service import (
    EnrichmentService,
)
from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)
from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)
from backend.app.ingestion.models.chunk_document import (
    ChunkDocument,
)
from backend.app.ingestion.pipeline import (
    IngestionPipeline,
)
from backend.app.ingestion.service import (
    IngestionPersistenceResult,
)


class FakeImageProvider:
    def describe_image(
        self,
        image_path: str,
    ) -> ImageSemanticResult:
        return ImageSemanticResult(
            text="A system architecture diagram.",
            metadata={
                "provider": "fake",
            },
        )


class FakeIngestionService:
    def __init__(self) -> None:
        self.persisted_result: IngestionResult | None = None

    async def start_run(self, *, stats=None):
        return uuid.uuid4()

    async def persist(
        self,
        result,
        *,
        source_type,
        title=None,
        document_metadata=None,
    ):
        self.persisted_result = result

        return IngestionPersistenceResult(
            document_id=result.document_id,
            elements_persisted=len(
                result.elements
            ),
            relationships_persisted=len(
                result.relationships
            ),
        )

    async def attach_document(
        self,
        run_id,
        document_id,
    ):
        pass

    async def mark_completed(
        self,
        run_id,
        *,
        stats=None,
    ):
        pass

    async def mark_failed(
        self,
        run_id,
        *,
        error_message,
        stats=None,
    ):
        raise error_message


class FakeVectorIndexer:
    def __init__(self) -> None:
        self.indexed_chunks = []

    async def index(self, chunks):
        self.indexed_chunks = chunks


def make_image_result(
    image_path: str,
) -> IngestionResult:
    image = SourceElement(
        element_id="image-1",
        document_id="document-1",
        element_type="image",
        content=ImageContent(
            path=image_path,
        ),
        metadata={
            "page_number": 2,
            "content_type": "image",
        },
    )

    return IngestionResult(
        document_id="document-1",
        elements=[image],
        relationships=[],
    )


def test_enrichment_service_preserves_original_and_adds_semantic_element(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake")

    enricher = EnrichmentService(
        image_enricher=ImageEnricher(
            provider=FakeImageProvider()
        )
    )

    result = enricher.enrich(
        make_image_result(
            str(image_path)
        )
    )

    assert len(result.elements) == 2
    assert len(result.relationships) == 1

    original = result.elements[0]
    semantic = result.elements[1]

    assert original.element_id == "image-1"
    assert isinstance(
        original.content,
        ImageContent,
    )

    assert semantic.element_id == "image-1:semantic"
    assert isinstance(
        semantic.content,
        TextContent,
    )

    relationship = result.relationships[0]

    assert (
        relationship.source_element_id
        == "image-1:semantic"
    )
    assert (
        relationship.target_element_id
        == "image-1"
    )
    assert (
        relationship.relationship_type
        == "derived_from"
    )


@pytest.mark.asyncio
async def test_pipeline_runs_enrichment_before_preprocessing(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake")

    ingestion_service = FakeIngestionService()
    vector_indexer = FakeVectorIndexer()

    enrichment_service = EnrichmentService(
        image_enricher=ImageEnricher(
            provider=FakeImageProvider()
        )
    )

    observed_results = []

    def fake_preprocessor(
        result: IngestionResult,
    ) -> IngestionResult:
        observed_results.append(result)

        # Simulate real preprocessing boundary.
        return result

    def fake_chunker(
        result: IngestionResult,
    ) -> list[ChunkDocument]:
        assert any(
            element.element_id
            == "image-1:semantic"
            for element in result.elements
        )

        assert any(
            relationship.relationship_type
            == "derived_from"
            for relationship in result.relationships
        )

        semantic_element = next(
            element
            for element in result.elements
            if element.element_id
            == "image-1:semantic"
        )

        assert isinstance(
            semantic_element.content,
            TextContent,
        )

        return [
            ChunkDocument(
                content=semantic_element.content.text,
                document_id=result.document_id,
                chunk_index=0,
                metadata=semantic_element.metadata,
            )
        ]

    pipeline = IngestionPipeline(
        ingestion_service=ingestion_service,
        vector_indexer=vector_indexer,
        preprocessor=fake_preprocessor,
        chunker=fake_chunker,
        enricher=enrichment_service,
    )

    result = await pipeline.ingest(
        lambda: make_image_result(
            str(image_path)
        ),
        source_type="image",
        title="diagram.png",
    )

    assert len(observed_results) == 1

    preprocessed_input = observed_results[0]

    assert any(
        element.element_id
        == "image-1:semantic"
        for element in preprocessed_input.elements
    )

    assert result.elements_persisted == 2
    assert result.relationships_persisted == 1
    assert result.chunks_indexed == 1

    assert len(
        ingestion_service.persisted_result.elements
    ) == 2
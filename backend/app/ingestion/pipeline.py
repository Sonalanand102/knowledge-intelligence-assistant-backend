from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from backend.app.ingestion.models.chunk_document import (
    ChunkDocument,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.service import (
    IngestionService,
)
from backend.app.retrieval.vector_indexer import (
    VectorIndexer,
)


@dataclass(frozen=True)
class IngestionPipelineResult:
    run_id: uuid.UUID
    document_id: str
    elements_persisted: int
    relationships_persisted: int
    chunks_indexed: int


class IngestionPipeline:
    """
    Orchestrates source ingestion from loader output
    through relational persistence and vector indexing.

    Flow:

        Loader
          ↓
        Preprocessing
          ↓
        Chunking
          ↓
        PostgreSQL
          ↓
        Qdrant
    """

    def __init__(
        self,
        ingestion_service: IngestionService,
        vector_indexer: VectorIndexer,
        preprocessor: Callable[
            [IngestionResult],
            IngestionResult,
        ],
        chunker: Callable[
            [IngestionResult],
            list[ChunkDocument],
        ],
    ) -> None:
        self.ingestion_service = ingestion_service
        self.vector_indexer = vector_indexer
        self.preprocessor = preprocessor
        self.chunker = chunker

    async def ingest(
        self,
        loader: Callable[[], IngestionResult],
        *,
        source_type: str,
        title: str | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> IngestionPipelineResult:

        # -----------------------------------------------------
        # 1. Start ingestion run
        # -----------------------------------------------------

        run_id = await self.ingestion_service.start_run(
            stats={
                "source_type": source_type,
            }
        )

        try:

            # -------------------------------------------------
            # 2. Load
            # -------------------------------------------------

            ingestion_result = await asyncio.to_thread(
                loader
            )

            if not isinstance(
                ingestion_result,
                IngestionResult,
            ):
                raise TypeError(
                    "Loader must return an IngestionResult"
                )

            # -------------------------------------------------
            # 3. Preprocess
            # -------------------------------------------------

            processed_result = await asyncio.to_thread(
                self.preprocessor,
                ingestion_result,
            )

            if not isinstance(
                processed_result,
                IngestionResult,
            ):
                raise TypeError(
                    "Preprocessor must return an IngestionResult"
                )

            # -------------------------------------------------
            # 4. Chunk
            # -------------------------------------------------

            chunks = await asyncio.to_thread(
                self.chunker,
                processed_result,
            )

            if not isinstance(chunks, list):
                raise TypeError(
                    "Chunker must return a list of ChunkDocument"
                )

            if not all(
                isinstance(chunk, ChunkDocument)
                for chunk in chunks
            ):
                raise TypeError(
                    "Chunker returned an invalid chunk"
                )

            # -------------------------------------------------
            # 5. Persist source data in PostgreSQL
            # -------------------------------------------------

            persistence_result = (
                await self.ingestion_service.persist(
                    processed_result,
                    source_type=source_type,
                    title=title,
                    document_metadata=document_metadata,
                )
            )

            # -------------------------------------------------
            # 6. Attach document to ingestion run
            # -------------------------------------------------

            await self.ingestion_service.attach_document(
                run_id,
                persistence_result.document_id,
            )

            # -------------------------------------------------
            # 7. Index chunks in Qdrant
            # -------------------------------------------------

            await self.vector_indexer.index(
                chunks
            )

            # -------------------------------------------------
            # 8. Mark ingestion as completed
            # -------------------------------------------------

            stats = {
                "source_type": source_type,
                "elements_persisted": (
                    persistence_result.elements_persisted
                ),
                "relationships_persisted": (
                    persistence_result.relationships_persisted
                ),
                "chunks_indexed": len(chunks),
            }

            await self.ingestion_service.mark_completed(
                run_id,
                stats=stats,
            )

            # -------------------------------------------------
            # 9. Return pipeline result
            # -------------------------------------------------

            return IngestionPipelineResult(
                run_id=run_id,
                document_id=(
                    persistence_result.document_id
                ),
                elements_persisted=(
                    persistence_result.elements_persisted
                ),
                relationships_persisted=(
                    persistence_result.relationships_persisted
                ),
                chunks_indexed=len(chunks),
            )

        except Exception as exc:

            # -------------------------------------------------
            # 10. Mark ingestion as failed
            # -------------------------------------------------

            await self.ingestion_service.mark_failed(
                run_id,
                error_message=str(exc),
            )

            raise
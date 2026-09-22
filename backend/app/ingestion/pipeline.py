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

from backend.app.ingestion.enrichment.service import (
    EnrichmentService,
)

import logging
import time

logger = logging.getLogger(__name__)

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
        enricher: EnrichmentService | None = None,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.vector_indexer = vector_indexer
        self.preprocessor = preprocessor
        self.chunker = chunker
        self.enricher = enricher

    async def ingest(
        self,
        loader: Callable[[], IngestionResult],
        *,
        source_type: str,
        title: str | None = None,
        document_metadata: dict[str, Any] | None = None,
        allow_existing_document: bool = False,
    ) -> IngestionPipelineResult:

        pipeline_started_at = time.perf_counter()

        logger.info(
            "[PIPELINE] started source_type=%s",
            source_type,
        )
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
            load_started_at = time.perf_counter()

            logger.info("[PIPELINE] loader started")

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

            logger.info(
                "[PIPELINE] loader completed duration=%.2fs elements=%d relationships=%d",
                time.perf_counter() - load_started_at,
                len(ingestion_result.elements),
                len(ingestion_result.relationships),
            )

            # -------------------------------------------------
            # 3. Semantic enrichment
            # -------------------------------------------------

            if self.enricher is not None:

                enrichment_started_at = time.perf_counter()

                logger.info(
                    "[PIPELINE] enrichment started elements=%d",
                    len(ingestion_result.elements),
                )

                ingestion_result = await asyncio.to_thread(
                    self.enricher.enrich,
                    ingestion_result,
                )

                logger.info(
                    "[PIPELINE] enrichment completed duration=%.2fs elements=%d relationships=%d",
                    time.perf_counter() - enrichment_started_at,
                    len(ingestion_result.elements),
                    len(ingestion_result.relationships),
                )

                if not isinstance(
                    ingestion_result,
                    IngestionResult,
                ):
                    raise TypeError(
                        "Enricher must return an IngestionResult"
                    )

            # -------------------------------------------------
            # 3. Preprocess
            # -------------------------------------------------

            preprocess_started_at = time.perf_counter()

            logger.info("[PIPELINE] preprocessing started")

            processed_result = await asyncio.to_thread(
                self.preprocessor,
                ingestion_result,
            )

            logger.info(
                "[PIPELINE] preprocessing completed duration=%.2fs",
                time.perf_counter() - preprocess_started_at,
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

            chunk_started_at = time.perf_counter()

            logger.info("[PIPELINE] chunking started")

            chunks = await asyncio.to_thread(
                self.chunker,
                processed_result,
            )

            logger.info(
                "[PIPELINE] chunking completed duration=%.2fs chunks=%d",
                time.perf_counter() - chunk_started_at,
                len(chunks),
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

            if allow_existing_document:

                persist_started_at = time.perf_counter()

                logger.info("[PIPELINE] postgres persistence started")

                persistence_result = (
                    await self.ingestion_service.persist(
                        processed_result,
                        source_type=source_type,
                        title=title,
                        document_metadata=document_metadata,
                        allow_existing=True,
                    )
                )

                logger.info(
                    "[PIPELINE] postgres persistence completed duration=%.2fs elements=%d relationships=%d",
                    time.perf_counter() - persist_started_at,
                    persistence_result.elements_persisted,
                    persistence_result.relationships_persisted,
                )
            else:

                persist_started_at = time.perf_counter()

                logger.info("[PIPELINE] postgres persistence started")

                persistence_result = (
                    await self.ingestion_service.persist(
                        processed_result,
                        source_type=source_type,
                        title=title,
                        document_metadata=document_metadata,
                    )
                )

                logger.info(
                    "[PIPELINE] postgres persistence completed duration=%.2fs elements=%d relationships=%d",
                    time.perf_counter() - persist_started_at,
                    persistence_result.elements_persisted,
                    persistence_result.relationships_persisted,
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

            index_started_at = time.perf_counter()

            logger.info(
                "[PIPELINE] qdrant indexing started chunks=%d",
                len(chunks),
            )

            await self.vector_indexer.index(chunks)

            logger.info(
                "[PIPELINE] qdrant indexing completed duration=%.2fs",
                time.perf_counter() - index_started_at,
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

            logger.info(
                "[PIPELINE] completed total_duration=%.2fs",
                time.perf_counter() - pipeline_started_at,
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
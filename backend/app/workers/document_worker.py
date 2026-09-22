from __future__ import annotations

import asyncio
import logging

from backend.app.services.document_processing_service import (
    DocumentProcessingService,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def process_document_job(
    document_id: str,
) -> dict:
    """
    RQ entry point.

    RQ executes synchronous Python callables,
    while our application pipeline is async.
    """

    result = asyncio.run(
        DocumentProcessingService().process_document(
            document_id,
        )
    )

    return {
        "document_id": result.document_id,
        "run_id": str(result.run_id),
        "elements_persisted": (
            result.elements_persisted
        ),
        "relationships_persisted": (
            result.relationships_persisted
        ),
        "chunks_indexed": (
            result.chunks_indexed
        ),
    }
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from backend.app.ingestion.models.chunk_document import (
    ChunkDocument,
)
from backend.app.ingestion.models.content import (
    TableContent,
    TextContent,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.models.source_document import (
    SourceDocument,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)


def chunk_documents(
    documents: IngestionResult | list[SourceDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[ChunkDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[ChunkDocument] = []

    # ---------------------------------------------------------
    # New relationship-aware ingestion path
    # ---------------------------------------------------------
    if isinstance(documents, IngestionResult):
        for element in documents.elements:
            if not isinstance(
                element,
                SourceElement,
            ):
                continue

            if isinstance(
                element.content,
                TextContent,
            ):
                text = element.content.text

            elif isinstance(
                element.content,
                TableContent,
            ):
                text = element.content.text

            else:
                continue

            if not text.strip():
                continue

            split_texts = splitter.split_text(
                text
            )

            for chunk_index, chunk_text in enumerate(
                split_texts
            ):
                metadata = (
                    element.metadata.copy()
                )

                metadata.update(
                    {
                        "element_id": (
                            element.element_id
                        ),
                        "element_type": (
                            element.element_type
                        ),
                    }
                )

                chunks.append(
                    ChunkDocument(
                        content=chunk_text,
                        document_id=(
                            documents.document_id
                        ),
                        chunk_index=chunk_index,
                        metadata=metadata,
                    )
                )

        return chunks

    # ---------------------------------------------------------
    # Legacy SourceDocument path
    # Keep this temporarily while other loaders
    # are being migrated.
    # ---------------------------------------------------------
    for document in documents:
        if isinstance(
            document.content,
            TextContent,
        ):
            text = document.content.text

        elif isinstance(
            document.content,
            TableContent,
        ):
            text = document.content.text

        else:
            continue

        if not text.strip():
            continue

        split_texts = splitter.split_text(
            text
        )

        for chunk_index, chunk_text in enumerate(
            split_texts
        ):
            chunks.append(
                ChunkDocument(
                    content=chunk_text,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    metadata=document.metadata.copy(),
                )
            )

    return chunks
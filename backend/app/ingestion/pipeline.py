from backend.app.ingestion.chunking.text_chunker import (
    chunk_documents,
)
from backend.app.ingestion.loaders.pdf_loader import (
    load_pdf,
)


def process_pdf(
    file_path: str,
    document_id: str,
    file_name: str,
):
    ingestion_result = load_pdf(
        file_path=file_path,
        document_id=document_id,
        output_dir="tmp/pdf",
    )

    chunks = chunk_documents(
        ingestion_result
    )

    return chunks
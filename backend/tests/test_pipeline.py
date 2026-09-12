from backend.app.ingestion.pipeline import process_pdf


def test_process_pdf():

    chunks = process_pdf(
        file_path="backend/tests/data/Sonal Anand - Resume7.pdf",
        document_id="test-document",
        file_name="Sonal Anand - Resume7.pdf",
    )

    assert len(chunks) > 0

    first_chunk = chunks[0]

    assert first_chunk.document_id == "test-document"
    assert first_chunk.metadata["file_name"] == "Sonal Anand - Resume7.pdf"
    assert first_chunk.metadata["page_number"] >= 1
    assert first_chunk.chunk_index >= 0
    assert first_chunk.content
from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.dependencies import (
    get_document_upload_service,
)
from backend.app.api.v1.documents import (
    router,
)
from backend.app.schemas.document_upload import (
    MultiDocumentUploadResponse,
    UploadedDocumentResponse,
)


class FakeDocumentUploadService:
    async def upload_documents(
        self,
        chat_id: str,
        files,
    ) -> MultiDocumentUploadResponse:
        return MultiDocumentUploadResponse(
            chat_id=chat_id,
            documents=[
                UploadedDocumentResponse(
                    document_id="doc-1",
                    filename="research.pdf",
                    source_type="pdf",
                    status="pending",
                ),
                UploadedDocumentResponse(
                    document_id="doc-2",
                    filename="notes.docx",
                    source_type="docx",
                    status="pending",
                ),
            ],
        )


def create_test_app() -> FastAPI:
    app = FastAPI()

    app.include_router(
        router,
        prefix="/api/v1",
    )

    app.dependency_overrides[
        get_document_upload_service
    ] = lambda: FakeDocumentUploadService()

    return app


def test_upload_multiple_documents():
    app = create_test_app()

    client = TestClient(app)

    response = client.post(
        "/api/v1/chats/chat-123/documents",
        files=[
            (
                "files",
                (
                    "research.pdf",
                    b"%PDF-test",
                    "application/pdf",
                ),
            ),
            (
                "files",
                (
                    "notes.docx",
                    b"fake-docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            ),
        ],
    )

    assert response.status_code == 201

    body = response.json()

    assert body["chat_id"] == "chat-123"

    assert len(
        body["documents"]
    ) == 2

    assert (
        body["documents"][0]["source_type"]
        == "pdf"
    )

    assert (
        body["documents"][1]["source_type"]
        == "docx"
    )

    assert (
        body["documents"][0]["status"]
        == "pending"
    )
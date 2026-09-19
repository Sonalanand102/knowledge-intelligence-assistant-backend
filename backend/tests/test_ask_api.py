from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.dependencies import (
    get_answer_service,
)
from backend.app.api.v1.ask import router
from backend.app.generation.answer_generation import (
    Citation,
    GeneratedAnswer,
)


class FakeAnswerService:
    async def answer(
        self,
        query: str,
        top_k: int,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=(
                "RAG combines retrieval with "
                "language model generation."
            ),
            citations=[
                Citation(
                    citation_id=1,
                    chunk_id="chunk_rag_001",
                    document_id="doc_rag",
                    metadata={
                        "file_name": "rag.pdf",
                        "page_number": 1,
                    },
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
        get_answer_service
    ] = lambda: FakeAnswerService()

    return app


def test_ask_endpoint():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "query": "What is RAG?",
                "top_k": 5,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is RAG?"

    assert (
        data["answer"]
        == "RAG combines retrieval with "
        "language model generation."
    )

    assert len(data["citations"]) == 1

    citation = data["citations"][0]

    assert citation["citation_id"] == 1

    assert citation["chunk_id"] == (
        "chunk_rag_001"
    )

    assert citation["document_id"] == (
        "doc_rag"
    )

    assert citation["metadata"] == {
        "file_name": "rag.pdf",
        "page_number": 1,
    }


def test_ask_endpoint_rejects_empty_query():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "query": "   ",
                "top_k": 5,
            },
        )

    assert response.status_code == 422


def test_ask_endpoint_rejects_missing_query():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "top_k": 5,
            },
        )

    assert response.status_code == 422


def test_ask_endpoint_rejects_invalid_top_k():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "query": "What is RAG?",
                "top_k": 0,
            },
        )

    assert response.status_code == 422


def test_ask_endpoint_rejects_top_k_above_limit():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "query": "What is RAG?",
                "top_k": 51,
            },
        )

    assert response.status_code == 422

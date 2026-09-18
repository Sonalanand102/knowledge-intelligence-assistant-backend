from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.dependencies import (
    get_search_service,
)
from backend.app.api.v1.search import router
from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.search_service import (
    SearchResponse,
)


@dataclass
class FakeSearchService:
    async def search(
        self,
        query: str,
        top_k: int,
    ) -> SearchResponse:
        return SearchResponse(
            query=query,
            results=[
                VectorSearchResult(
                    chunk_id="chunk_rag_001",
                    document_id="doc_rag",
                    chunk_index=0,
                    content="RAG retrieves external context.",
                    score=0.95,
                    metadata={
                        "source": "evaluation",
                    },
                )
            ],
        )


def create_test_app() -> FastAPI:
    app = FastAPI()

    app.include_router(
        router,
        prefix="/api/v1",
    )

    app.dependency_overrides[
        get_search_service
    ] = lambda: FakeSearchService()

    return app


def test_search_endpoint():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/search",
            params={
                "q": "What is RAG?",
                "top_k": 5,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is RAG?"

    assert len(data["results"]) == 1

    result = data["results"][0]

    assert result["chunk_id"] == (
        "chunk_rag_001"
    )

    assert result["document_id"] == (
        "doc_rag"
    )

    assert result["chunk_index"] == 0

    assert result["score"] == 0.95


def test_search_endpoint_rejects_empty_query():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/search",
            params={
                "q": "   ",
            },
        )

    assert response.status_code == 422


def test_search_endpoint_rejects_invalid_top_k():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/search",
            params={
                "q": "What is RAG?",
                "top_k": 0,
            },
        )

    assert response.status_code == 422


def test_search_endpoint_rejects_top_k_above_limit():
    app = create_test_app()

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/search",
            params={
                "q": "What is RAG?",
                "top_k": 51,
            },
        )

    assert response.status_code == 422
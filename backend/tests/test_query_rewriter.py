from types import SimpleNamespace

import pytest

from backend.app.retrieval.query_rewriter import (
    BaseQueryRewriter,
    GeminiQueryRewriter,
)


class FakeQueryRewriter(BaseQueryRewriter):
    def _rewrite(self, query: str) -> str:
        return f"rewritten: {query}"


def test_query_rewriter_rewrites_query():
    rewriter = FakeQueryRewriter()

    result = rewriter.rewrite("What is RAG?")

    assert result == "rewritten: What is RAG?"


def test_query_rewriter_strips_query():
    rewriter = FakeQueryRewriter()

    result = rewriter.rewrite("  What is RAG?  ")

    assert result == "rewritten: What is RAG?"


def test_query_rewriter_rejects_empty_query():
    rewriter = FakeQueryRewriter()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        rewriter.rewrite("   ")


class EmptyQueryRewriter(BaseQueryRewriter):
    def _rewrite(self, query: str) -> str:
        return "   "


def test_query_rewriter_rejects_empty_rewrite():
    rewriter = EmptyQueryRewriter()

    with pytest.raises(
        ValueError,
        match="Rewriter returned an empty query",
    ):
        rewriter.rewrite("What is RAG?")


class FakeGeminiModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            text=self.text
        )


class FakeGeminiClient:
    def __init__(self, text: str) -> None:
        self.models = FakeGeminiModels(text)


def test_gemini_query_rewriter_calls_generate_content():
    client = FakeGeminiClient(
        "multimodal RAG combining images and text"
    )

    rewriter = GeminiQueryRewriter(
        client=client,
        model_name="fake-model",
    )

    result = rewriter.rewrite(
        "How can images and text be used together in RAG?"
    )

    assert result == (
        "multimodal RAG combining images and text"
    )

    assert len(client.models.calls) == 1

    call = client.models.calls[0]

    assert call["model"] == "fake-model"
    assert call["contents"] == (
        "How can images and text be used together in RAG?"
    )


def test_gemini_query_rewriter_rejects_empty_response():
    client = FakeGeminiClient("")

    rewriter = GeminiQueryRewriter(
        client=client,
        model_name="fake-model",
    )

    with pytest.raises(
        ValueError,
        match="Gemini returned an empty response",
    ):
        rewriter.rewrite("What is RAG?")
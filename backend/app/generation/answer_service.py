from __future__ import annotations

from backend.app.generation.answer_generation import (
    BaseAnswerGenerator,
    GeneratedAnswer,
)
from backend.app.retrieval.base import Retriever


class AnswerService:
    """
    Orchestrates retrieval and answer generation.

    Flow:
        user query
            ↓
        retriever
            ↓
        retrieved chunks
            ↓
        answer generator
            ↓
        generated answer + citations

    document_ids can optionally scope retrieval to a set of
    documents, which is used by chat-scoped RAG.
    """

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: BaseAnswerGenerator,
    ) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator

    async def answer(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> GeneratedAnswer:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        normalized_query = query.strip()

        if document_ids:
            results = await self.retriever.retrieve(
                query=normalized_query,
                top_k=top_k,
                document_ids=document_ids,
            )
        else:
            results = await self.retriever.retrieve(
                query=normalized_query,
                top_k=top_k,
            )

        return self.answer_generator.generate(
            query=normalized_query,
            results=results,
        )
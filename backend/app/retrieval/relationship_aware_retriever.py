from __future__ import annotations

from backend.app.retrieval.base import (
    Retriever,
    VectorSearchResult,
)
from backend.app.retrieval.relationship_context_expander import (
    RelationshipContextExpander,
)


class RelationshipAwareRetriever:
    """
    Wraps a base retriever and enriches retrieved results
    with related source-element context.

    Flow:

        query
          ↓
        base retriever
          ↓
        top-k results
          ↓
        relationship context expansion
          ↓
        enriched results
    """

    def __init__(
        self,
        retriever: Retriever,
        context_expander: RelationshipContextExpander,
    ) -> None:
        self.retriever = retriever
        self.context_expander = context_expander

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
    ) -> list[VectorSearchResult]:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        results = await self.retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
            document_ids=document_ids,
        )

        return await self.context_expander.expand(
            results
        )
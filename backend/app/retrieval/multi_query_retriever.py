from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from backend.app.retrieval.base import (
    Retriever,
    VectorSearchResult,
)
from backend.app.retrieval.rrf import (
    reciprocal_rank_fusion,
)
from pydantic import BaseModel, Field
from google.genai import types

class MultiQueryResponse(BaseModel):
    queries: list[str] = Field(min_length=3, max_length=3)

class BaseMultiQueryGenerator(ABC):
    def generate(self, query: str) -> list[str]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        queries = self._generate(query.strip())

        cleaned_queries: list[str] = []
        seen: set[str] = set()

        for generated_query in queries:
            if not generated_query:
                continue

            cleaned_query = generated_query.strip()

            if not cleaned_query:
                continue

            normalized_query = cleaned_query.lower()

            if normalized_query in seen:
                continue

            seen.add(normalized_query)
            cleaned_queries.append(cleaned_query)

        if not cleaned_queries:
            raise ValueError(
                "Multi-query generator returned no valid queries"
            )

        return cleaned_queries

    @abstractmethod
    def _generate(self, query: str) -> list[str]:
        raise NotImplementedError


class GeminiMultiQueryGenerator(BaseMultiQueryGenerator):
    DEFAULT_MODEL = "gemini-3.8-flash"

    SYSTEM_INSTRUCTION = """
You are a search query generator for a knowledge retrieval system.

Generate exactly 3 different search queries for the user's question.

Rules:
- Preserve the original intent.
- Do not answer the question.
- Do not invent facts.
- Preserve important technical terms.
- Each query should provide a different retrieval perspective.
- Keep each query concise.
- Do not add unrelated concepts.
""".strip()

    def __init__(
        self,
        client: Any,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.client = client
        self.model_name = model_name

    def _generate(
        self,
        query: str,
    ) -> list[str]:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=200,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low",
                ),
                response_mime_type="application/json",
                response_schema=MultiQueryResponse,
            ),
        )

        parsed = response.parsed

        if parsed is None:
            raise ValueError(
                "Gemini returned no structured multi-query response"
            )

        queries = parsed.queries

        if len(queries) != 3:
            raise ValueError(
                "Gemini must return exactly 3 queries"
            )

        return queries 

class MultiQueryRetriever:
    """
    Multi-query retrieval pipeline:

        original query
              ↓
        query generator
              ↓
       multiple queries
              ↓
      parallel retrieval
              ↓
          RRF fusion
              ↓
          top_k results
    """

    def __init__(
        self,
        retriever: Retriever,
        query_generator: BaseMultiQueryGenerator,
        include_original: bool = True,
    ) -> None:
        self.retriever = retriever
        self.query_generator = query_generator
        self.include_original = include_original

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        original_query = query.strip()

        generated_queries = await asyncio.to_thread(
            self.query_generator.generate,
            original_query,
        )

        queries = list(generated_queries)

        if self.include_original:
            normalized_queries = {
                item.lower()
                for item in queries
            }

            if original_query.lower() not in normalized_queries:
                queries.insert(0, original_query)

        if not queries:
            return []

        result_lists = await asyncio.gather(
            *(
                self.retriever.retrieve(
                    query=retrieval_query,
                    top_k=top_k,
                )
                for retrieval_query in queries
            )
        )

        return reciprocal_rank_fusion(
            result_lists,
            top_k=top_k,
        )

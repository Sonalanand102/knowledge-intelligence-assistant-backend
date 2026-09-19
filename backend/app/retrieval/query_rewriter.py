
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseQueryRewriter(ABC):
    def rewrite(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        rewritten_query = self._rewrite(query.strip())

        if not rewritten_query or not rewritten_query.strip():
            raise ValueError("Rewriter returned an empty query")

        return rewritten_query.strip()

    @abstractmethod
    def _rewrite(self, query: str) -> str:
        raise NotImplementedError


class GeminiQueryRewriter(BaseQueryRewriter):
    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    SYSTEM_INSTRUCTION = """
You are a search query optimizer for a knowledge retrieval system.

Your task is to rewrite a user's query only when doing so can improve
retrieval quality.

Rules:
- Preserve the user's original intent exactly.
- Preserve important technical terms and entities.
- Do not invent facts or assumptions.
- Do not answer the question.
- Do not add unrelated concepts.
- Keep the rewritten query concise.
- Prefer terminology that is likely to appear in knowledge sources.
- If the original query is already clear and retrieval-friendly,
  return it unchanged.
- Return ONLY the rewritten search query.
- Do not add explanations.
- Do not use markdown.
""".strip()

    def __init__(
        self,
        client: Any,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.client = client
        self.model_name = model_name

    def _rewrite(self, query: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=query,
            config={
                "system_instruction": self.SYSTEM_INSTRUCTION,
                "temperature": 0.0,
                "max_output_tokens": 100,
            },
        )

        rewritten_query = response.text

        if not rewritten_query:
            raise ValueError(
                "Gemini returned an empty response"
            )

        rewritten_query = rewritten_query.strip()

        # Remove accidental markdown code fences.
        if rewritten_query.startswith("```"):
            rewritten_query = (
                rewritten_query
                .replace("```text", "")
                .replace("```", "")
                .strip()
            )

        # Remove accidental surrounding quotes.
        if (
            rewritten_query.startswith('"')
            and rewritten_query.endswith('"')
        ):
            rewritten_query = rewritten_query[1:-1].strip()

        if not rewritten_query:
            raise ValueError(
                "Gemini returned an empty response"
            )

        return rewritten_query

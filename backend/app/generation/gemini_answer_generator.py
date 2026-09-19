from __future__ import annotations

from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from backend.app.generation.answer_generation import (
    BaseAnswerGenerator,
    Citation,
    GeneratedAnswer,
)
from backend.app.retrieval.base import VectorSearchResult


class AnswerGenerationResponse(BaseModel):
    answer: str = Field(
        description="Grounded answer based only on the provided context."
    )
    citation_ids: list[int] = Field(
        description=(
            "1-based citation numbers corresponding to the provided "
            "context chunks."
        )
    )


class GeminiAnswerGenerator(BaseAnswerGenerator):
    DEFAULT_MODEL = "gemini-3.8-flash"

    SYSTEM_INSTRUCTION = """
You are a grounded knowledge assistant.

Answer the user's question using ONLY the provided retrieved context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the context is insufficient, say that the available context is insufficient.
- Keep the answer concise and directly answer the user's question.
- Every factual claim must be supported by one or more provided context chunks.
- Select citation_ids only from the provided context numbers.
- Do not invent citation numbers.
- Return only the structured response.
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
        results: list[VectorSearchResult],
    ) -> GeneratedAnswer:
        context_parts: list[str] = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            context_parts.append(
                (
                    f"[{index}]\n"
                    f"Chunk ID: {result.chunk_id}\n"
                    f"Document ID: {result.document_id}\n"
                    f"Content:\n{result.content}"
                )
            )

        context = "\n\n".join(context_parts)

        prompt = (
            f"User question:\n{query}\n\n"
            f"Retrieved context:\n{context}"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                max_output_tokens=1000,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low",
                ),
                response_mime_type="application/json",
                response_schema=AnswerGenerationResponse,
            ),
        )

        parsed = response.parsed

        if parsed is None:
            raise ValueError(
                "Gemini returned no structured answer response"
            )

        answer = parsed.answer.strip()

        if not answer:
            raise ValueError(
                "Gemini returned an empty answer"
            )

        citation_ids = parsed.citation_ids

        if not citation_ids:
            raise ValueError(
                "Gemini returned no citation IDs"
            )

        # Remove duplicate citation IDs while preserving order.
        unique_citation_ids = list(
            dict.fromkeys(citation_ids)
        )

        max_citation_id = len(results)

        invalid_ids = [
            citation_id
            for citation_id in unique_citation_ids
            if citation_id < 1
            or citation_id > max_citation_id
        ]

        if invalid_ids:
            raise ValueError(
                f"Gemini returned invalid citation IDs: {invalid_ids}"
            )

        citations: list[Citation] = []

        for citation_id in unique_citation_ids:
            result = results[citation_id - 1]

            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    metadata=dict(result.metadata),
                )
            )

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
        )

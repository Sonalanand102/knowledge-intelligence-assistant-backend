from __future__ import annotations

import mimetypes
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from backend.app.ingestion.enrichment.base import (
    ImageSemanticProvider,
    ImageSemanticResult,
)


class GeminiImageAnalysis(BaseModel):
    description: str = Field(
        description="Detailed semantic description of the image."
    )
    ocr_text: str = Field(
        default="",
        description="All meaningful readable text visible in the image."
    )
    visual_elements: list[str] = Field(
        default_factory=list,
        description="Important objects, components, labels, or visual structures."
    )
    document_context: str = Field(
        default="",
        description="What role this image appears to play in a document."
    )


class GeminiImageSemanticProvider(ImageSemanticProvider):
    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-3.8-flash",
    ) -> None:
        self.client = client
        self.model = model

    def describe_image(
        self,
        image_path: str,
    ) -> ImageSemanticResult:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image asset not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Image asset is not a file: {path}"
            )

        mime_type, _ = mimetypes.guess_type(path.name)

        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(
                f"Unsupported or unknown image MIME type: {path}"
            )

        prompt = """
Analyze this image for a knowledge retrieval system.

Extract information that would help answer future questions about this image.

Focus on:
1. A detailed semantic description.
2. Any readable/OCR text.
3. Important visual entities, labels, components, or structures.
4. The role/context of the image in a document.

For diagrams, explain the relationships and flow between components.
For charts, explain titles, axes, labels, legends, and the main visible trends.
For screenshots, identify the application/UI context and meaningful visible text.
Do not invent information that cannot be reasonably observed.
"""

        image_part = types.Part.from_bytes(
            data=path.read_bytes(),
            mime_type=mime_type,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                image_part,
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiImageAnalysis,
            ),
        )

        raw_text = (response.text or "").strip()

        if not raw_text:
            raise ValueError(
                "Gemini image semantic provider returned empty response"
            )

        analysis = GeminiImageAnalysis.model_validate_json(raw_text)

        description = analysis.description.strip()

        if not description:
            raise ValueError(
                "Gemini image semantic provider returned empty description"
            )

        searchable_text_parts = [
            description,
        ]

        if analysis.ocr_text.strip():
            searchable_text_parts.append(
                f"Readable text: {analysis.ocr_text.strip()}"
            )

        if analysis.visual_elements:
            searchable_text_parts.append(
                "Visual elements: "
                + ", ".join(
                    item.strip()
                    for item in analysis.visual_elements
                    if item.strip()
                )
            )

        if analysis.document_context.strip():
            searchable_text_parts.append(
                f"Context: {analysis.document_context.strip()}"
            )

        return ImageSemanticResult(
            text="\n".join(searchable_text_parts),
            metadata={
                "representation_type": "semantic_description",
                "provider": "gemini",
                "model": self.model,
                "ocr_text": analysis.ocr_text.strip(),
                "visual_elements": analysis.visual_elements,
                "document_context": analysis.document_context.strip(),
            },
        )
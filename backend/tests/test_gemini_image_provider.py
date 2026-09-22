from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.app.ingestion.enrichment.providers.gemini_image import (
    GeminiImageSemanticProvider,
)


def test_describe_image_returns_semantic_result(tmp_path: Path):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")

    client = MagicMock()

    response = MagicMock()
    response.text = """
    {
        "description": "A system architecture diagram showing an API layer connected to a PostgreSQL database.",
        "ocr_text": "FastAPI\\nPostgreSQL",
        "visual_elements": ["API layer", "PostgreSQL database"],
        "document_context": "Technical architecture diagram"
    }
    """

    client.models.generate_content.return_value = response

    provider = GeminiImageSemanticProvider(
        client=client,
        model="gemini-3.8-flash",
    )

    result = provider.describe_image(str(image_path))

    assert result.text
    assert "FastAPI" in result.text
    assert result.metadata["representation_type"] == "semantic_description"
    assert result.metadata["ocr_text"] == "FastAPI\nPostgreSQL"

    client.models.generate_content.assert_called_once()


def test_describe_image_rejects_missing_file():
    client = MagicMock()

    provider = GeminiImageSemanticProvider(
        client=client,
        model="gemini-3.8-flash",
    )

    with pytest.raises(FileNotFoundError):
        provider.describe_image("/does/not/exist.png")


def test_describe_image_rejects_empty_response(tmp_path: Path):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")

    client = MagicMock()

    response = MagicMock()
    response.text = ""

    client.models.generate_content.return_value = response

    provider = GeminiImageSemanticProvider(
        client=client,
        model="gemini-3.8-flash",
    )

    with pytest.raises(ValueError, match="empty"):
        provider.describe_image(str(image_path))
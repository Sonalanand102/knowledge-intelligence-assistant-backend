from __future__ import annotations

from dataclasses import dataclass

from backend.app.ingestion.models.content import (
    AudioContent,
    ImageContent,
    TableContent,
    TextContent,
    VideoContent,
)
from backend.app.ingestion.models.source_element import SourceElement


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reason: str | None = None


def validate_element(element: SourceElement) -> ValidationResult:
    content = element.content

    if isinstance(content, TextContent):
        if not content.text.strip():
            return ValidationResult(
                is_valid=False,
                reason="empty_text",
            )

        return ValidationResult(is_valid=True)

    if isinstance(content, TableContent):
        if not content.text.strip():
            return ValidationResult(
                is_valid=False,
                reason="empty_table",
            )

        return ValidationResult(is_valid=True)

    if isinstance(content, ImageContent):
        if not content.path.strip():
            return ValidationResult(
                is_valid=False,
                reason="missing_image_path",
            )

        return ValidationResult(is_valid=True)

    if isinstance(content, AudioContent):
        if not content.path.strip():
            return ValidationResult(
                is_valid=False,
                reason="missing_audio_path",
            )

        return ValidationResult(is_valid=True)

    if isinstance(content, VideoContent):
        if not content.path.strip():
            return ValidationResult(
                is_valid=False,
                reason="missing_video_path",
            )

        return ValidationResult(is_valid=True)

    # Unknown content types should not silently pass validation.
    return ValidationResult(
        is_valid=False,
        reason="unsupported_content_type",
    )
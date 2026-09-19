from __future__ import annotations

import pytest

from backend.app.ingestion.loader_registry import (
    LoaderRegistry,
    UnsupportedFileTypeError,
)


def test_pdf_is_detected():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "research.pdf"
        )
        == "pdf"
    )


def test_docx_is_detected():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "architecture.docx"
        )
        == "docx"
    )


def test_pptx_is_detected():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "presentation.pptx"
        )
        == "pptx"
    )


def test_excel_is_detected():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "data.xlsx"
        )
        == "excel"
    )


def test_markdown_is_detected():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "notes.md"
        )
        == "markdown"
    )


def test_image_is_detected():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "diagram.png"
        )
        == "image"
    )


def test_audio_is_detected():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "meeting.mp3"
        )
        == "audio"
    )


def test_video_is_detected():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "lecture.mp4"
        )
        == "video"
    )


def test_extension_matching_is_case_insensitive():
    registry = LoaderRegistry()

    assert (
        registry.detect_source_type(
            "Research.PDF"
        )
        == "pdf"
    )


def test_unsupported_extension_is_rejected():
    registry = LoaderRegistry()

    with pytest.raises(
        UnsupportedFileTypeError
    ):
        registry.detect_source_type(
            "archive.zip"
        )


def test_missing_extension_is_rejected():
    registry = LoaderRegistry()

    with pytest.raises(
        UnsupportedFileTypeError
    ):
        registry.detect_source_type(
            "research"
        )
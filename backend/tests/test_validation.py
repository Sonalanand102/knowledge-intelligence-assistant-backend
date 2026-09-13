from backend.app.ingestion.models.content import (
    AudioContent,
    ImageContent,
    TableContent,
    TextContent,
    VideoContent,
)
from backend.app.ingestion.models.source_element import SourceElement
from backend.app.ingestion.preprocessing.validation import validate_element


def make_element(content, element_type="paragraph"):
    return SourceElement(
        element_id="element-1",
        document_id="document-1",
        element_type=element_type,
        content=content,
        metadata={},
    )


def test_valid_text_element():
    element = make_element(TextContent("Hello world"))

    result = validate_element(element)

    assert result.is_valid is True
    assert result.reason is None


def test_empty_text_element_is_invalid():
    element = make_element(TextContent(""))

    result = validate_element(element)

    assert result.is_valid is False
    assert result.reason == "empty_text"


def test_whitespace_only_text_element_is_invalid():
    element = make_element(TextContent("   \n\t  "))

    result = validate_element(element)

    assert result.is_valid is False
    assert result.reason == "empty_text"


def test_valid_table_element():
    element = make_element(
        TableContent("Name | Age\nAlice | 25"),
        element_type="table",
    )

    result = validate_element(element)

    assert result.is_valid is True
    assert result.reason is None


def test_empty_table_element_is_invalid():
    element = make_element(
        TableContent("   "),
        element_type="table",
    )

    result = validate_element(element)

    assert result.is_valid is False
    assert result.reason == "empty_table"


def test_valid_image_element():
    element = make_element(
        ImageContent("/tmp/image.png"),
        element_type="image",
    )

    result = validate_element(element)

    assert result.is_valid is True
    assert result.reason is None


def test_image_without_path_is_invalid():
    element = make_element(
        ImageContent(""),
        element_type="image",
    )

    result = validate_element(element)

    assert result.is_valid is False
    assert result.reason == "missing_image_path"


def test_valid_audio_element():
    element = make_element(
        AudioContent("/tmp/audio.wav"),
        element_type="audio",
    )

    result = validate_element(element)

    assert result.is_valid is True
    assert result.reason is None


def test_audio_without_path_is_invalid():
    element = make_element(
        AudioContent(""),
        element_type="audio",
    )

    result = validate_element(element)

    assert result.is_valid is False
    assert result.reason == "missing_audio_path"


def test_valid_video_element():
    element = make_element(
        VideoContent("/tmp/video.mp4"),
        element_type="video",
    )

    result = validate_element(element)

    assert result.is_valid is True
    assert result.reason is None


def test_video_without_path_is_invalid():
    element = make_element(
        VideoContent(""),
        element_type="video",
    )

    result = validate_element(element)

    assert result.is_valid is False
    assert result.reason == "missing_video_path"


def test_validation_does_not_modify_element():
    element = make_element(TextContent("  Hello  "))

    original_text = element.content.text

    validate_element(element)

    assert element.content.text == original_text
import pytest

from backend.app.ingestion.preprocessing.text import normalize_text


def test_normalize_text_strips_leading_and_trailing_whitespace():
    text = "   Hello world   "

    result = normalize_text(text)

    assert result == "Hello world"


def test_normalize_text_normalizes_line_endings():
    text = "Hello\r\nWorld\rTest"

    result = normalize_text(text)

    assert result == "Hello\nWorld\nTest"


def test_normalize_text_removes_trailing_spaces():
    text = "Hello   \nWorld   "

    result = normalize_text(text)

    assert result == "Hello\nWorld"


def test_normalize_text_reduces_excessive_blank_lines():
    text = "Hello\n\n\n\nWorld"

    result = normalize_text(text)

    assert result == "Hello\n\nWorld"


def test_normalize_text_normalizes_unicode():
    text = "Cafe\u0301"

    result = normalize_text(text)

    assert result == "Café"


def test_normalize_text_preserves_meaningful_newlines():
    text = "Step 1:\nInstall package.\n\nStep 2:\nRun server."

    result = normalize_text(text)

    assert result == text


def test_normalize_text_empty_string():
    assert normalize_text("") == ""


def test_normalize_text_whitespace_only():
    assert normalize_text("   \n\n   ") == ""
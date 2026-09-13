from backend.app.ingestion.preprocessing.metadata import normalize_metadata


def test_normalize_page_number_to_int():
    metadata = {
        "page_number": "3",
    }

    result = normalize_metadata(metadata)

    assert result["page_number"] == 3
    assert isinstance(result["page_number"], int)


def test_normalize_slide_number_to_int():
    metadata = {
        "slide_number": "5",
    }

    result = normalize_metadata(metadata)

    assert result["slide_number"] == 5
    assert isinstance(result["slide_number"], int)


def test_normalize_timestamp_fields_to_float():
    metadata = {
        "timestamp_seconds": "12.5",
        "start_seconds": "10",
        "end_seconds": "15",
    }

    result = normalize_metadata(metadata)

    assert result["timestamp_seconds"] == 12.5
    assert result["start_seconds"] == 10.0
    assert result["end_seconds"] == 15.0


def test_normalize_numeric_metadata_preserves_zero():
    metadata = {
        "page_number": 0,
        "timestamp_seconds": 0,
    }

    result = normalize_metadata(metadata)

    assert result["page_number"] == 0
    assert result["timestamp_seconds"] == 0.0


def test_missing_known_fields_are_not_added():
    metadata = {
        "source_type": "pdf",
    }

    result = normalize_metadata(metadata)

    assert result == {
        "source_type": "pdf",
    }


def test_unknown_metadata_fields_are_preserved():
    metadata = {
        "page_number": "3",
        "bbox": {
            "x0": 10,
            "y0": 20,
            "x1": 100,
            "y1": 80,
        },
        "font_size": 14,
        "heading_level": 2,
        "custom_loader_field": "keep-me",
    }

    result = normalize_metadata(metadata)

    assert result["page_number"] == 3
    assert result["bbox"] == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }
    assert result["font_size"] == 14
    assert result["heading_level"] == 2
    assert result["custom_loader_field"] == "keep-me"


def test_original_metadata_is_not_modified():
    metadata = {
        "page_number": "3",
        "timestamp_seconds": "12.5",
    }

    original = metadata.copy()

    result = normalize_metadata(metadata)

    assert metadata == original
    assert result is not metadata


def test_none_metadata_values_are_preserved():
    metadata = {
        "page_number": None,
        "timestamp_seconds": None,
        "custom_field": None,
    }

    result = normalize_metadata(metadata)

    assert result["page_number"] is None
    assert result["timestamp_seconds"] is None
    assert result["custom_field"] is None


def test_string_metadata_is_stripped():
    metadata = {
        "source_type": "  pdf  ",
        "element_type": " paragraph ",
        "content_type": " text ",
    }

    result = normalize_metadata(metadata)

    assert result["source_type"] == "pdf"
    assert result["element_type"] == "paragraph"
    assert result["content_type"] == "text"


def test_empty_string_metadata_is_preserved_as_empty_string():
    metadata = {
        "source_type": "",
        "url": "",
    }

    result = normalize_metadata(metadata)

    assert result["source_type"] == ""
    assert result["url"] == ""
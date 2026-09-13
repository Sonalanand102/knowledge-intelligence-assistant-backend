from __future__ import annotations

from typing import Any


_INT_FIELDS = {
    "page_number",
    "slide_number",
    "segment_index",
}

_FLOAT_FIELDS = {
    "timestamp_seconds",
    "start_seconds",
    "end_seconds",
}

_STRING_FIELDS = {
    "source_type",
    "element_type",
    "content_type",
}


def _normalize_numeric(
    value: Any,
    field_name: str,
    converter: type[int] | type[float],
) -> Any:
    """Normalize a known numeric metadata field without failing ingestion."""
    if value is None:
        return None

    try:
        return converter(value)
    except (TypeError, ValueError):
        # Preserve the original value rather than risking metadata loss.
        return value


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Return a normalized copy of metadata.

    Known fields are normalized to consistent types, while all unknown
    metadata keys and values are preserved unchanged.
    """
    normalized = metadata.copy()

    for field_name in _INT_FIELDS:
        if field_name in normalized:
            normalized[field_name] = _normalize_numeric(
                normalized[field_name],
                field_name,
                int,
            )

    for field_name in _FLOAT_FIELDS:
        if field_name in normalized:
            normalized[field_name] = _normalize_numeric(
                normalized[field_name],
                field_name,
                float,
            )

    for field_name in _STRING_FIELDS:
        if field_name not in normalized:
            continue

        value = normalized[field_name]

        if isinstance(value, str):
            normalized[field_name] = value.strip()

    return normalized
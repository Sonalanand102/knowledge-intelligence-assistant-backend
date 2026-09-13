from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize extracted text while preserving meaningful line structure.

    Operations:
    - Unicode normalization (NFC)
    - CRLF/CR -> LF
    - Remove control characters except newline/tab
    - Remove trailing whitespace from lines
    - Reduce excessive blank lines
    - Strip surrounding whitespace
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove control characters while preserving tabs and newlines.
    text = "".join(
        char
        for char in text
        if char in {"\n", "\t"} or not unicodedata.category(char).startswith("C")
    )

    # Remove trailing whitespace on each line.
    lines = [line.rstrip() for line in text.split("\n")]

    # Remove excessive blank lines while preserving paragraph separation.
    normalized_lines: list[str] = []
    blank_line_count = 0

    for line in lines:
        if line.strip():
            normalized_lines.append(line)
            blank_line_count = 0
            continue

        blank_line_count += 1

        if blank_line_count <= 1:
            normalized_lines.append("")

    text = "\n".join(normalized_lines)

    return text.strip()
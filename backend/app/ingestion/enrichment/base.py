from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ImageSemanticResult:
    text: str
    metadata: dict[str, object]


class ImageSemanticProvider(Protocol):
    def describe_image(
        self,
        image_path: str,
    ) -> ImageSemanticResult:
        """
        Convert an image into searchable semantic text.
        """
        ...
from dataclasses import dataclass


@dataclass
class TextContent:
    text: str


@dataclass
class ImageContent:
    path: str


@dataclass
class AudioContent:
    path: str


@dataclass
class VideoContent:
    path: str


@dataclass
class TableContent:
    text: str
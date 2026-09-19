from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredFile:
    original_filename: str
    storage_path: str
    size_bytes: int


class LocalFileStorage:
    """
    Stores uploaded files on the local filesystem.

    Layout:

        root/
          document_id/
            original_filename
    """

    def __init__(
        self,
        root_dir: str | Path,
        max_file_size_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError(
                "max_file_size_bytes must be greater than zero"
            )

        self.root_dir = Path(root_dir)
        self.max_file_size_bytes = max_file_size_bytes

        self.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _sanitize_filename(
        filename: str,
    ) -> str:
        """
        Prevent path traversal and normalize unsafe filenames.
        """
        name = Path(filename).name

        name = re.sub(
            r"[^A-Za-z0-9._-]",
            "_",
            name,
        )

        name = name.strip(".")

        if not name:
            raise ValueError(
                "Invalid filename"
            )

        return name

    async def save(
        self,
        file,
        *,
        document_id: str,
    ) -> StoredFile:
        if not document_id:
            raise ValueError(
                "document_id cannot be empty"
            )

        if not file.filename:
            raise ValueError(
                "Uploaded file must have a filename"
            )

        filename = self._sanitize_filename(
            file.filename
        )

        document_dir = (
            self.root_dir / document_id
        )

        document_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = document_dir / filename

        total_bytes = 0

        try:
            with destination.open("wb") as output:
                while True:
                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if (
                        total_bytes
                        > self.max_file_size_bytes
                    ):
                        raise ValueError(
                            f"File exceeds maximum allowed "
                            f"size of "
                            f"{self.max_file_size_bytes} bytes"
                        )

                    output.write(chunk)

        except Exception:
            destination.unlink(
                missing_ok=True
            )
            raise

        return StoredFile(
            original_filename=filename,
            storage_path=str(destination),
            size_bytes=total_bytes,
        )
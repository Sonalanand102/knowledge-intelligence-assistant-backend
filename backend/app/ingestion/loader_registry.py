from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable

from backend.app.ingestion.loaders.audio_loader import load_audio
from backend.app.ingestion.loaders.csv_loader import load_csv
from backend.app.ingestion.loaders.docx_loader import load_docx
from backend.app.ingestion.loaders.excel_loader import load_excel
from backend.app.ingestion.loaders.html_loader import load_html
from backend.app.ingestion.loaders.image_loader import load_image
from backend.app.ingestion.loaders.markdown_loader import load_markdown
from backend.app.ingestion.loaders.pdf_loader import load_pdf
from backend.app.ingestion.loaders.pptx_loader import load_pptx
from backend.app.ingestion.loaders.txt_loader import load_txt
from backend.app.ingestion.loaders.video_loader import load_video
from backend.app.ingestion.models.ingestion_result import IngestionResult


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file type is not supported."""


class LoaderConfigurationError(ValueError):
    """Raised when a loader cannot be constructed correctly."""


@dataclass(frozen=True)
class LoaderSpec:
    source_type: str
    extensions: frozenset[str]


class LoaderRegistry:
    """
    Resolves uploaded files to a normalized source type
    and creates the corresponding loader callable.

    The registry does not execute the loader itself.
    It only resolves the loader and binds the required arguments.

    Example:

        registry = LoaderRegistry()

        loader = registry.create_loader(
            source_type="pdf",
            file_path="/tmp/file.pdf",
            document_id="abc-123",
            output_dir="/tmp/processed",
        )

        ingestion_result = loader()
    """

    DEFAULT_SPECS: tuple[LoaderSpec, ...] = (
        LoaderSpec(
            source_type="pdf",
            extensions=frozenset({".pdf"}),
        ),
        LoaderSpec(
            source_type="docx",
            extensions=frozenset({".docx"}),
        ),
        LoaderSpec(
            source_type="pptx",
            extensions=frozenset({".pptx"}),
        ),
        LoaderSpec(
            source_type="excel",
            extensions=frozenset({".xlsx", ".xls"}),
        ),
        LoaderSpec(
            source_type="csv",
            extensions=frozenset({".csv"}),
        ),
        LoaderSpec(
            source_type="markdown",
            extensions=frozenset({".md", ".markdown"}),
        ),
        LoaderSpec(
            source_type="html",
            extensions=frozenset({".html", ".htm"}),
        ),
        LoaderSpec(
            source_type="txt",
            extensions=frozenset({".txt"}),
        ),
        LoaderSpec(
            source_type="image",
            extensions=frozenset(
                {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".gif",
                    ".bmp",
                    ".tiff",
                    ".tif",
                }
            ),
        ),
        LoaderSpec(
            source_type="audio",
            extensions=frozenset(
                {
                    ".mp3",
                    ".wav",
                    ".m4a",
                    ".aac",
                    ".flac",
                    ".ogg",
                }
            ),
        ),
        LoaderSpec(
            source_type="video",
            extensions=frozenset(
                {
                    ".mp4",
                    ".mov",
                    ".avi",
                    ".mkv",
                    ".webm",
                }
            ),
        ),
    )

    LOADERS: dict[str, Callable[..., IngestionResult]] = {
        "pdf": load_pdf,
        "docx": load_docx,
        "pptx": load_pptx,
        "excel": load_excel,
        "csv": load_csv,
        "markdown": load_markdown,
        "html": load_html,
        "txt": load_txt,
        "image": load_image,
        "audio": load_audio,
        "video": load_video,
    }

    def __init__(
        self,
        specs: tuple[LoaderSpec, ...] | None = None,
    ) -> None:
        self._specs = specs or self.DEFAULT_SPECS

        self._extension_map: dict[str, str] = {}

        for spec in self._specs:
            if spec.source_type not in self.LOADERS:
                raise LoaderConfigurationError(
                    "No loader registered for source type: "
                    f"{spec.source_type}"
                )

            for extension in spec.extensions:
                normalized_extension = extension.lower().strip()

                if not normalized_extension.startswith("."):
                    raise LoaderConfigurationError(
                        "File extension must start with '.': "
                        f"{extension}"
                    )

                if normalized_extension in self._extension_map:
                    raise LoaderConfigurationError(
                        "Duplicate extension registration: "
                        f"{normalized_extension}"
                    )

                self._extension_map[normalized_extension] = (
                    spec.source_type
                )

    def detect_source_type(
        self,
        filename: str,
    ) -> str:
        """
        Detect normalized source type from filename extension.
        """

        if not filename or not filename.strip():
            raise UnsupportedFileTypeError(
                "Filename cannot be empty"
            )

        extension = Path(filename).suffix.lower()

        if not extension:
            raise UnsupportedFileTypeError(
                f"File has no extension: {filename}"
            )

        source_type = self._extension_map.get(extension)

        if source_type is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {extension}"
            )

        return source_type

    def is_supported(
        self,
        filename: str,
    ) -> bool:
        """
        Return True when the filename extension is supported.
        """

        try:
            self.detect_source_type(filename)
            return True
        except UnsupportedFileTypeError:
            return False

    def get_loader(
        self,
        source_type: str,
    ) -> Callable[..., IngestionResult]:
        """
        Return the loader function for a normalized source type.
        """

        normalized_source_type = source_type.strip().lower()

        loader = self.LOADERS.get(
            normalized_source_type
        )

        if loader is None:
            raise LoaderConfigurationError(
                "No loader registered for source type: "
                f"{source_type}"
            )

        return loader

    def create_loader(
        self,
        *,
        file_path: str | Path,
        document_id: str,
        output_dir: str | Path | None = None,
        source_type: str | None = None,
        base_url: str | None = None,
        filename: str | None = None,
    ) -> Callable[[], IngestionResult]:
        """
        Create a zero-argument loader callable suitable for
        IngestionPipeline.ingest().

        The registry inspects the selected loader's signature and
        only passes arguments that the loader actually accepts.

        Supported argument names:

            file_path
            document_id
            output_dir
            file_name
            filename
            base_url

        This keeps the registry compatible with loaders having
        slightly different signatures.
        """

        resolved_source_type = (
            source_type.strip().lower()
            if source_type
            else self.detect_source_type(
                filename or str(file_path)
            )
        )

        loader = self.get_loader(
            resolved_source_type
        )

        path = Path(file_path)

        if not path.exists():
            raise LoaderConfigurationError(
                f"Loader input file does not exist: {path}"
            )

        if not document_id.strip():
            raise LoaderConfigurationError(
                "document_id cannot be empty"
            )

        effective_filename = (
            filename
            or path.name
        )

        available_kwargs: dict[str, Any] = {
            "file_path": str(path),
            "document_id": document_id,
            "output_dir": (
                str(output_dir)
                if output_dir is not None
                else None
            ),
            "file_name": effective_filename,
            "filename": effective_filename,
            "base_url": base_url,
        }

        bound_kwargs = self._build_loader_kwargs(
            loader=loader,
            available_kwargs=available_kwargs,
        )

        def execute_loader() -> IngestionResult:
            result = loader(**bound_kwargs)

            if not isinstance(
                result,
                IngestionResult,
            ):
                raise TypeError(
                    f"Loader '{resolved_source_type}' "
                    "must return an IngestionResult"
                )

            if result.document_id != document_id:
                raise ValueError(
                    "Loader returned an unexpected document_id: "
                    f"expected={document_id}, "
                    f"received={result.document_id}"
                )

            return result

        return execute_loader

    @staticmethod
    def _build_loader_kwargs(
        *,
        loader: Callable[..., IngestionResult],
        available_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Inspect a loader signature and construct only the kwargs
        that the loader accepts.

        Required parameters that cannot be supplied are rejected
        early with a clear configuration error.
        """

        try:
            loader_signature = signature(loader)
        except (TypeError, ValueError) as exc:
            raise LoaderConfigurationError(
                f"Could not inspect loader signature: {loader}"
            ) from exc

        bound_kwargs: dict[str, Any] = {}

        parameters = loader_signature.parameters

        has_var_keyword = any(
            parameter.kind
            == Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        for name, parameter in parameters.items():

            if parameter.kind in (
                Parameter.VAR_POSITIONAL,
                Parameter.VAR_KEYWORD,
            ):
                continue

            if name in available_kwargs:
                value = available_kwargs[name]

                if value is None:
                    if parameter.default is not Parameter.empty:
                        continue

                    raise LoaderConfigurationError(
                        "Required loader parameter cannot be "
                        f"resolved: '{name}' "
                        f"for loader '{loader.__name__}'"
                    )

                bound_kwargs[name] = value
                continue

            if parameter.default is not Parameter.empty:
                continue

            raise LoaderConfigurationError(
                "Unsupported required loader parameter "
                f"'{name}' in '{loader.__name__}'. "
                "Add support for this parameter in LoaderRegistry."
            )

        if has_var_keyword:
            for name, value in available_kwargs.items():
                if value is not None:
                    bound_kwargs.setdefault(
                        name,
                        value,
                    )

        return bound_kwargs
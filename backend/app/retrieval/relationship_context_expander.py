from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from backend.app.retrieval.base import VectorSearchResult


STRONG_RELATIONSHIPS = frozenset(
    {
        "contains",
        "parent_of",
        "child_of",
        "caption_of",
        "derived_from",
        "refers_to",
        "belongs_to",
    }
)


@dataclass(frozen=True)
class RelationshipContext:
    element_id: str
    document_id: str
    element_type: str
    relationship_type: str
    text: str | None
    asset_path: str | None
    metadata: dict[str, Any]


class RelationshipStore(Protocol):
    async def get_relationships(
        self,
        element_ids: set[str],
        document_id: str,
    ) -> list[Any]:
        ...

    async def get_elements(
        self,
        element_ids: set[str],
        document_id: str,
    ) -> list[Any]:
        ...


class RelationshipContextExpander:
    """
    Expands retrieved chunks through strong semantic relationships.

    Example:

        image_semantic
            ↓ derived_from
        image
            ↓ refers_to
        paragraph

    Only strong relationships are used for context expansion.

    Contextual relationships such as spatially_adjacent,
    temporally_adjacent, follows and precedes are intentionally ignored.
    """

    def __init__(
        self,
        store: RelationshipStore,
        max_hops: int = 1,
    ) -> None:
        if max_hops <= 0:
            raise ValueError(
                "max_hops must be greater than zero"
            )

        self.store = store
        self.max_hops = max_hops

    async def expand(
        self,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        if not results:
            return []

        expanded_results: list[VectorSearchResult] = []

        for result in results:
            expanded_results.append(
                await self._expand_result(result)
            )

        return expanded_results

    async def _expand_result(
        self,
        result: VectorSearchResult,
    ) -> VectorSearchResult:
        element_ids = self._get_element_ids(
            result
        )

        if not element_ids:
            return self._with_empty_context(
                result
            )

        document_id = result.document_id

        visited: set[str] = set(
            element_ids
        )

        frontier: set[str] = set(
            element_ids
        )

        contexts: list[RelationshipContext] = []

        for _ in range(self.max_hops):
            if not frontier:
                break

            relationships = (
                await self.store.get_relationships(
                    element_ids=frontier,
                    document_id=document_id,
                )
            )

            next_candidate_ids: set[str] = set()

            for relationship in relationships:
                relationship_type = str(
                    getattr(
                        relationship,
                        "relationship_type",
                        "",
                    )
                )

                if (
                    relationship_type
                    not in STRONG_RELATIONSHIPS
                ):
                    continue

                source_id = str(
                    getattr(
                        relationship,
                        "source_element_id",
                        "",
                    )
                )

                target_id = str(
                    getattr(
                        relationship,
                        "target_element_id",
                        "",
                    )
                )

                if not source_id or not target_id:
                    continue

                # Traverse the relationship from whichever side
                # is already in the current frontier.
                if source_id in frontier:
                    related_id = target_id
                elif target_id in frontier:
                    related_id = source_id
                else:
                    continue

                if related_id in visited:
                    continue

                next_candidate_ids.add(
                    related_id
                )

            if not next_candidate_ids:
                break

            elements = await self.store.get_elements(
                element_ids=next_candidate_ids,
                document_id=document_id,
            )

            element_by_id = {
                str(element.element_id): element
                for element in elements
                if str(
                    getattr(
                        element,
                        "document_id",
                        "",
                    )
                )
                == document_id
            }

            next_frontier: set[str] = set()

            for related_id in next_candidate_ids:
                element = element_by_id.get(
                    related_id
                )

                if element is None:
                    # This also protects us from accidentally
                    # crossing document boundaries.
                    continue

                relationship_type = (
                    self._find_relationship_type(
                        relationships=relationships,
                        current_ids=frontier,
                        related_id=related_id,
                    )
                )

                contexts.append(
                    RelationshipContext(
                        element_id=str(
                            element.element_id
                        ),
                        document_id=document_id,
                        element_type=str(
                            getattr(
                                element,
                                "element_type",
                                "",
                            )
                        ),
                        relationship_type=(
                            relationship_type
                        ),
                        text=self._get_text(
                            element
                        ),
                        asset_path=self._get_asset_path(
                            element
                        ),
                        metadata=self._get_metadata(
                            element
                        ),
                    )
                )

                visited.add(
                    related_id
                )

                next_frontier.add(
                    related_id
                )

            frontier = next_frontier

        return self._merge_context(
            result,
            contexts,
        )

    @staticmethod
    def _get_metadata(
        element: Any,
    ) -> dict[str, Any]:
        metadata = getattr(
            element,
            "metadata",
            None,
        )

        if metadata is None:
            metadata = getattr(
                element,
                "metadata_json",
                {},
            )

        if not isinstance(metadata, dict):
            return {}

        return dict(metadata)

    @staticmethod
    def _get_element_ids(
        result: VectorSearchResult,
    ) -> set[str]:
        raw_ids = result.metadata.get(
            "element_ids",
            [],
        )

        if not isinstance(raw_ids, list):
            return set()

        return {
            str(element_id)
            for element_id in raw_ids
            if str(element_id).strip()
        }

    @staticmethod
    def _find_relationship_type(
        relationships: list[Any],
        current_ids: set[str],
        related_id: str,
    ) -> str:
        for relationship in relationships:
            source_id = str(
                getattr(
                    relationship,
                    "source_element_id",
                    "",
                )
            )

            target_id = str(
                getattr(
                    relationship,
                    "target_element_id",
                    "",
                )
            )

            if (
                (
                    source_id in current_ids
                    and target_id == related_id
                )
                or (
                    target_id in current_ids
                    and source_id == related_id
                )
            ):
                return str(
                    getattr(
                        relationship,
                        "relationship_type",
                        "",
                    )
                )

        return ""

    @staticmethod
    def _get_text(
        element: Any,
    ) -> str | None:
        text = getattr(
            element,
            "text_content",
            None,
        )

        if text is None:
            return None

        text = str(text).strip()

        return text or None

    @staticmethod
    def _get_asset_path(
        element: Any,
    ) -> str | None:
        asset_path = getattr(
            element,
            "asset_path",
            None,
        )

        if asset_path is None:
            return None

        asset_path = str(
            asset_path
        ).strip()

        return asset_path or None

    @staticmethod
    def _merge_context(
        result: VectorSearchResult,
        contexts: list[RelationshipContext],
    ) -> VectorSearchResult:
        existing_metadata = dict(
            result.metadata
        )

        existing_context_ids = {
            str(item.get("element_id"))
            for item in existing_metadata.get(
                "relationship_context",
                [],
            )
            if isinstance(item, dict)
        }

        relationship_context = list(
            existing_metadata.get(
                "relationship_context",
                [],
            )
        )

        content_parts = [
            result.content
        ]

        for context in contexts:
            if context.element_id in existing_context_ids:
                continue

            context_payload = {
                "element_id": context.element_id,
                "document_id": context.document_id,
                "element_type": context.element_type,
                "relationship_type": (
                    context.relationship_type
                ),
                "text": context.text,
                "asset_path": context.asset_path,
                "metadata": dict(
                    context.metadata
                ),
            }

            relationship_context.append(
                context_payload
            )

            if context.text:
                content_parts.append(
                    (
                        "[Related context]\n"
                        f"{context.text}"
                    )
                )

        existing_metadata[
            "relationship_context"
        ] = relationship_context

        return VectorSearchResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            chunk_index=result.chunk_index,
            content="\n\n".join(
                content_parts
            ),
            score=result.score,
            metadata=existing_metadata,
        )

    @staticmethod
    def _with_empty_context(
        result: VectorSearchResult,
    ) -> VectorSearchResult:
        metadata = dict(
            result.metadata
        )

        metadata.setdefault(
            "relationship_context",
            [],
        )

        return VectorSearchResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            chunk_index=result.chunk_index,
            content=result.content,
            score=result.score,
            metadata=metadata,
        )
from backend.app.db.base import Base
from backend.app.db.models import (
    ElementRelationship,
    IngestionRun,
    SourceDocument,
    SourceElement,
)


def test_database_models_are_registered():
    expected_tables = {
        "source_documents",
        "source_elements",
        "element_relationships",
        "ingestion_runs",
    }

    assert expected_tables.issubset(
        set(Base.metadata.tables.keys())
    )


def test_source_document_table():
    table = SourceDocument.__table__

    assert "document_id" in table.columns
    assert "source_type" in table.columns
    assert "metadata" in table.columns


def test_source_element_table():
    table = SourceElement.__table__

    assert "element_id" in table.columns
    assert "document_id" in table.columns
    assert "text_content" in table.columns
    assert "asset_path" in table.columns


def test_relationship_table():
    table = ElementRelationship.__table__

    assert "source_element_id" in table.columns
    assert "relationship_type" in table.columns
    assert "target_element_id" in table.columns


def test_ingestion_run_table():
    table = IngestionRun.__table__

    assert "document_id" in table.columns
    assert "status" in table.columns
    assert "started_at" in table.columns
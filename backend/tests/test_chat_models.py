from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from backend.app.db.models.chat import Chat
from backend.app.db.models.chat_document import ChatDocument
from backend.app.db.models.source_document import SourceDocument
from backend.app.db.session import AsyncSessionLocal


@pytest.mark.asyncio
async def test_chat_can_have_multiple_documents():
    chat = Chat(
        title="Research Chat",
    )

    document_1 = SourceDocument(
        document_id=f"chat-doc-{uuid.uuid4().hex}",
        source_type="pdf",
        title="Research PDF",
        metadata_json={},
    )

    document_2 = SourceDocument(
        document_id=f"chat-doc-{uuid.uuid4().hex}",
        source_type="docx",
        title="Architecture DOCX",
        metadata_json={},
    )

    async with AsyncSessionLocal() as session:
        session.add(chat)
        session.add(document_1)
        session.add(document_2)

        await session.flush()

        chat_document_1 = ChatDocument(
            chat_id=chat.id,
            document_id=document_1.document_id,
        )

        chat_document_2 = ChatDocument(
            chat_id=chat.id,
            document_id=document_2.document_id,
        )

        session.add(chat_document_1)
        session.add(chat_document_2)

        await session.commit()

        result = await session.execute(
            select(ChatDocument).where(
                ChatDocument.chat_id == chat.id,
            )
        )

        associations = result.scalars().all()

        assert len(associations) == 2

        document_ids = {
            association.document_id
            for association in associations
        }

        assert document_ids == {
            document_1.document_id,
            document_2.document_id,
        }


@pytest.mark.asyncio
async def test_same_document_cannot_be_added_twice_to_same_chat():
    chat = Chat(
        title="Duplicate Test",
    )

    document = SourceDocument(
        document_id=f"chat-doc-{uuid.uuid4().hex}",
        source_type="pdf",
        title="Test PDF",
        metadata_json={},
    )

    async with AsyncSessionLocal() as session:
        session.add(chat)
        session.add(document)

        await session.flush()

        association_1 = ChatDocument(
            chat_id=chat.id,
            document_id=document.document_id,
        )

        association_2 = ChatDocument(
            chat_id=chat.id,
            document_id=document.document_id,
        )

        session.add(association_1)
        session.add(association_2)

        with pytest.raises(Exception):
            await session.commit()


@pytest.mark.asyncio
async def test_same_document_can_be_attached_to_two_chats():
    chat_1 = Chat(
        title="Chat One",
    )

    chat_2 = Chat(
        title="Chat Two",
    )

    document = SourceDocument(
        document_id=f"chat-doc-{uuid.uuid4().hex}",
        source_type="pdf",
        title="Shared Document",
        metadata_json={},
    )

    async with AsyncSessionLocal() as session:
        session.add(chat_1)
        session.add(chat_2)
        session.add(document)

        await session.flush()

        session.add(
            ChatDocument(
                chat_id=chat_1.id,
                document_id=document.document_id,
            )
        )

        session.add(
            ChatDocument(
                chat_id=chat_2.id,
                document_id=document.document_id,
            )
        )

        await session.commit()

        result = await session.execute(
            select(ChatDocument).where(
                ChatDocument.document_id == document.document_id,
            )
        )

        associations = result.scalars().all()

        assert len(associations) == 2

        chat_ids = {
            association.chat_id
            for association in associations
        }

        assert chat_ids == {
            chat_1.id,
            chat_2.id,
        }

@pytest.mark.asyncio
async def test_deleting_chat_deletes_chat_document_associations():
    chat = Chat(
        title="Cascade Test",
    )

    document = SourceDocument(
        document_id=f"chat-doc-{uuid.uuid4().hex}",
        source_type="pdf",
        title="Cascade PDF",
        metadata_json={},
    )

    async with AsyncSessionLocal() as session:
        session.add(chat)
        session.add(document)

        await session.flush()

        session.add(
            ChatDocument(
                chat_id=chat.id,
                document_id=document.document_id,
            )
        )

        await session.commit()

        chat_id = chat.id

        await session.delete(chat)
        await session.commit()

        result = await session.execute(
            select(ChatDocument).where(
                ChatDocument.chat_id == chat_id,
            )
        )

        associations = result.scalars().all()

        assert associations == []
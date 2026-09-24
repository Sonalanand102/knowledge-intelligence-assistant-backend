# Knowledge Intelligence Assistant — Backend

> **A knowledge intelligence layer for turning fragmented information into grounded, searchable knowledge.**

## Overview

KIA is not another RAG chatbot. It is a knowledge intelligence system that ingests information from multiple sources, processes and connects it, retrieves relevant knowledge, and generates evidence-grounded answers.

```text
Information → Knowledge → Retrieval → Intelligence → Evidence
```

## Core Pipeline

```text
Ingestion
   ↓
Extraction & Preprocessing
   ↓
Relationship-Aware Chunking
   ↓
Embeddings
   ↓
Hybrid Retrieval
   ↓
Reranking
   ↓
Grounded Generation
   ↓
Evidence
```

## Supported Sources

**Documents:** PDF · DOCX · PPTX · XLSX · CSV · Markdown · HTML · TXT

**Media:** Images · Audio · Video

**Web:** Web Pages · YouTube

## Tech Stack

* Python / FastAPI
* PostgreSQL
* Vector Search
* Redis
* SQLAlchemy / Alembic
* Docker
* LLM & Embedding Models

## API

```text
GET  /api/v1/chats
POST /api/v1/chats

POST /api/v1/chats/{chat_id}/documents
GET  /api/v1/chats/{chat_id}/documents

POST /api/v1/chats/{chat_id}/ask
GET  /api/v1/chats/{chat_id}/messages

GET  /api/v1/documents/{document_id}
POST /api/v1/documents/{document_id}/retry
```

Swagger is available at `/docs` when the API is running.

## Run Locally

```bash
uv sync
docker compose up -d
uv run uvicorn app.main:app --reload
```

Run tests:

```bash
uv run pytest -v
```

## Future Direction

The V1 foundation is designed to support **domain-specific intelligence** in future versions:

```text
KIA Foundation
      ↓
Domain Intelligence
 ┌────┼────┐
Legal Research Healthcare
```

Domain versions can introduce specialized retrieval, terminology, relationships, reasoning and workflows.

## Vision

> **KIA is a foundation for building intelligence over information — not just another chatbot.**

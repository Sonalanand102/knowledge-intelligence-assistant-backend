from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingEvaluationCase:
    query: str
    relevant_chunk_ids: set[str]


def get_embedding_evaluation_cases() -> list[EmbeddingEvaluationCase]:
    return [
        # ---------------------------------------------------------
        # RAG
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is RAG?",
            relevant_chunk_ids={
                "chunk_rag_001",
                "chunk_rag_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="How does retrieval augmented generation work?",
            relevant_chunk_ids={
                "chunk_rag_001",
                "chunk_rag_002",
                "chunk_rag_003",
            },
        ),
        EmbeddingEvaluationCase(
            query="Why is retrieval useful before LLM generation?",
            relevant_chunk_ids={
                "chunk_rag_002",
                "chunk_rag_003",
            },
        ),

        # ---------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What are embeddings?",
            relevant_chunk_ids={
                "chunk_embeddings_001",
                "chunk_embeddings_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="How is text converted into a vector representation?",
            relevant_chunk_ids={
                "chunk_embeddings_001",
                "chunk_embeddings_002",
                "chunk_embeddings_003",
            },
        ),
        EmbeddingEvaluationCase(
            query="Why are embeddings useful for semantic search?",
            relevant_chunk_ids={
                "chunk_embeddings_002",
                "chunk_embeddings_003",
            },
        ),

        # ---------------------------------------------------------
        # Vector databases
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is a vector database?",
            relevant_chunk_ids={
                "chunk_vector_db_001",
                "chunk_vector_db_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="Why do RAG systems use vector databases?",
            relevant_chunk_ids={
                "chunk_vector_db_001",
                "chunk_vector_db_002",
                "chunk_vector_db_003",
            },
        ),
        EmbeddingEvaluationCase(
            query="How are embeddings stored and searched?",
            relevant_chunk_ids={
                "chunk_vector_db_002",
                "chunk_vector_db_003",
            },
        ),

        # ---------------------------------------------------------
        # Hybrid search
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is hybrid search?",
            relevant_chunk_ids={
                "chunk_hybrid_search_001",
                "chunk_hybrid_search_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="How can keyword and semantic search be combined?",
            relevant_chunk_ids={
                "chunk_hybrid_search_001",
                "chunk_hybrid_search_002",
                "chunk_hybrid_search_003",
            },
        ),
        EmbeddingEvaluationCase(
            query="Why combine lexical search with vector search?",
            relevant_chunk_ids={
                "chunk_hybrid_search_002",
                "chunk_hybrid_search_003",
            },
        ),

        # ---------------------------------------------------------
        # Chunking
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is chunking in RAG?",
            relevant_chunk_ids={
                "chunk_chunking_001",
                "chunk_chunking_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="Why do documents need to be split into chunks?",
            relevant_chunk_ids={
                "chunk_chunking_001",
                "chunk_chunking_002",
                "chunk_chunking_003",
            },
        ),
        EmbeddingEvaluationCase(
            query="How does chunk size affect retrieval?",
            relevant_chunk_ids={
                "chunk_chunking_002",
                "chunk_chunking_003",
            },
        ),

        # ---------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is metadata?",
            relevant_chunk_ids={
                "chunk_metadata_001",
                "chunk_metadata_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="Why is metadata important in document retrieval?",
            relevant_chunk_ids={
                "chunk_metadata_001",
                "chunk_metadata_002",
                "chunk_metadata_003",
            },
        ),
        EmbeddingEvaluationCase(
            query="How can metadata help filter retrieved documents?",
            relevant_chunk_ids={
                "chunk_metadata_002",
                "chunk_metadata_003",
            },
        ),

        # ---------------------------------------------------------
        # FastAPI
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is FastAPI?",
            relevant_chunk_ids={
                "chunk_fastapi_001",
                "chunk_fastapi_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="Why would FastAPI be used for an AI backend?",
            relevant_chunk_ids={
                "chunk_fastapi_001",
                "chunk_fastapi_002",
            },
        ),

        # ---------------------------------------------------------
        # Semantic search
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is semantic search?",
            relevant_chunk_ids={
                "chunk_semantic_search_001",
                "chunk_semantic_search_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="How does semantic search understand meaning?",
            relevant_chunk_ids={
                "chunk_semantic_search_001",
                "chunk_semantic_search_002",
            },
        ),

        # ---------------------------------------------------------
        # Reranking
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is reranking?",
            relevant_chunk_ids={
                "chunk_reranking_001",
                "chunk_reranking_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="Why is reranking used after retrieval?",
            relevant_chunk_ids={
                "chunk_reranking_001",
                "chunk_reranking_002",
            },
        ),

        # ---------------------------------------------------------
        # Multimodal RAG
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="What is multimodal RAG?",
            relevant_chunk_ids={
                "chunk_multimodal_rag_001",
                "chunk_multimodal_rag_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="How can images and text be used together in RAG?",
            relevant_chunk_ids={
                "chunk_multimodal_rag_001",
                "chunk_multimodal_rag_002",
            },
        ),

        # ---------------------------------------------------------
        # Relationships
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="Why are relationships between document elements useful?",
            relevant_chunk_ids={
                "chunk_relationships_001",
                "chunk_relationships_002",
                "chunk_relationships_003",
            },
        ),
        EmbeddingEvaluationCase(
            query="How can relationships preserve context between extracted elements?",
            relevant_chunk_ids={
                "chunk_relationships_001",
                "chunk_relationships_002",
            },
        ),

        # ---------------------------------------------------------
        # Cross-topic queries
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="How do embeddings and vector databases work together?",
            relevant_chunk_ids={
                "chunk_embeddings_002",
                "chunk_vector_db_001",
                "chunk_vector_db_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="How do chunking and embeddings fit into a RAG pipeline?",
            relevant_chunk_ids={
                "chunk_chunking_001",
                "chunk_chunking_002",
                "chunk_embeddings_001",
                "chunk_rag_001",
            },
        ),
        EmbeddingEvaluationCase(
            query="How does hybrid search improve retrieval?",
            relevant_chunk_ids={
                "chunk_hybrid_search_001",
                "chunk_hybrid_search_002",
                "chunk_hybrid_search_003",
            },
        ),
        EmbeddingEvaluationCase(
            query="How do semantic search and reranking work together?",
            relevant_chunk_ids={
                "chunk_semantic_search_001",
                "chunk_semantic_search_002",
                "chunk_reranking_001",
                "chunk_reranking_002",
            },
        ),
        EmbeddingEvaluationCase(
            query="How can metadata and relationships improve document retrieval?",
            relevant_chunk_ids={
                "chunk_metadata_001",
                "chunk_metadata_002",
                "chunk_relationships_001",
                "chunk_relationships_002",
            },
        ),

        # ---------------------------------------------------------
        # Distractor-heavy conceptual queries
        # ---------------------------------------------------------

        EmbeddingEvaluationCase(
            query="How does semantic retrieval differ from keyword matching?",
            relevant_chunk_ids={
                "chunk_semantic_search_001",
                "chunk_semantic_search_002",
                "chunk_hybrid_search_001",
            },
        ),
        EmbeddingEvaluationCase(
            query="What happens between document ingestion and final answer generation?",
            relevant_chunk_ids={
                "chunk_chunking_001",
                "chunk_embeddings_001",
                "chunk_vector_db_001",
                "chunk_rag_001",
            },
        ),
    ]


def get_embedding_evaluation_corpus() -> dict[str, str]:
    return {
        # ---------------------------------------------------------
        # RAG
        # ---------------------------------------------------------

        "chunk_rag_001": (
            "Retrieval Augmented Generation, or RAG, combines "
            "information retrieval with language model generation. "
            "The system first retrieves relevant information from "
            "a knowledge source and then provides that context to "
            "the language model."
        ),
        "chunk_rag_002": (
            "A typical RAG pipeline contains document ingestion, "
            "text preprocessing, chunking, embedding generation, "
            "vector retrieval, and answer generation. Retrieval "
            "provides external context that can improve grounded answers."
        ),
        "chunk_rag_003": (
            "RAG is useful when an application needs to answer "
            "questions using information that is not reliably "
            "contained in the language model's original training data. "
            "Retrieved context can also provide source information "
            "for the generated response."
        ),

        # ---------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------

        "chunk_embeddings_001": (
            "Embeddings are numerical vector representations of "
            "data such as text. An embedding model maps semantically "
            "related content into nearby regions of a vector space."
        ),
        "chunk_embeddings_002": (
            "Text embeddings allow systems to compare semantic "
            "similarity between queries and documents using vector "
            "operations such as cosine similarity."
        ),
        "chunk_embeddings_003": (
            "Embedding models transform text into fixed-dimensional "
            "vectors that can be indexed and searched efficiently. "
            "The same embedding space should generally be used for "
            "documents and their corresponding queries."
        ),

        # ---------------------------------------------------------
        # Vector database
        # ---------------------------------------------------------

        "chunk_vector_db_001": (
            "A vector database stores numerical embeddings and "
            "supports similarity search over those vectors. It is "
            "commonly used to retrieve semantically similar chunks "
            "for RAG applications."
        ),
        "chunk_vector_db_002": (
            "Vector databases can associate each vector with metadata "
            "such as document identifiers, source information, page "
            "numbers, timestamps, and content types."
        ),
        "chunk_vector_db_003": (
            "During vector retrieval, a query embedding is compared "
            "against stored document embeddings. The database returns "
            "the vectors with the highest similarity according to "
            "the selected distance metric."
        ),

        # ---------------------------------------------------------
        # Hybrid search
        # ---------------------------------------------------------

        "chunk_hybrid_search_001": (
            "Hybrid search combines dense semantic retrieval with "
            "sparse lexical retrieval. Dense retrieval captures "
            "meaning while lexical retrieval can match exact words "
            "and identifiers."
        ),
        "chunk_hybrid_search_002": (
            "BM25 is a common sparse retrieval approach based on "
            "term matching. Dense vector retrieval instead represents "
            "queries and documents as embeddings and compares their "
            "semantic similarity."
        ),
        "chunk_hybrid_search_003": (
            "Combining sparse and dense retrieval can improve search "
            "coverage because exact keyword matches and semantic matches "
            "provide complementary retrieval signals."
        ),

        # ---------------------------------------------------------
        # Chunking
        # ---------------------------------------------------------

        "chunk_chunking_001": (
            "Chunking splits a larger document into smaller pieces "
            "that can be embedded and retrieved independently. "
            "Chunking is an important preprocessing step in many RAG systems."
        ),
        "chunk_chunking_002": (
            "Chunk size affects retrieval quality because very small "
            "chunks may lose context while very large chunks may contain "
            "too much unrelated information."
        ),
        "chunk_chunking_003": (
            "Structure-aware chunking can preserve useful document "
            "boundaries such as headings, paragraphs, sections, tables, "
            "and relationships between extracted elements."
        ),

        # ---------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------

        "chunk_metadata_001": (
            "Metadata is information describing a piece of content. "
            "Examples include source type, document identifier, page "
            "number, timestamp, URL, and content type."
        ),
        "chunk_metadata_002": (
            "Metadata can be used during retrieval to filter or "
            "constrain search results. For example, a system can "
            "retrieve only chunks belonging to a particular document "
            "or source type."
        ),
        "chunk_metadata_003": (
            "Good metadata preservation helps maintain provenance "
            "through ingestion, chunking, retrieval, and final answer "
            "generation."
        ),

        # ---------------------------------------------------------
        # FastAPI
        # ---------------------------------------------------------

        "chunk_fastapi_001": (
            "FastAPI is a Python web framework commonly used to build "
            "HTTP APIs. It provides request validation, automatic API "
            "documentation, and asynchronous endpoint support."
        ),
        "chunk_fastapi_002": (
            "FastAPI can serve as the API layer of an AI application, "
            "connecting frontend clients with ingestion, retrieval, "
            "embedding, and generation services."
        ),

        # ---------------------------------------------------------
        # Semantic search
        # ---------------------------------------------------------

        "chunk_semantic_search_001": (
            "Semantic search retrieves content based on meaning rather "
            "than requiring an exact keyword match. Embeddings are often "
            "used to represent queries and documents semantically."
        ),
        "chunk_semantic_search_002": (
            "A semantic search system can retrieve documents that use "
            "different wording from the query when their meanings are "
            "closely related in the embedding space."
        ),

        # ---------------------------------------------------------
        # Reranking
        # ---------------------------------------------------------

        "chunk_reranking_001": (
            "Reranking is a second-stage retrieval process that takes "
            "an initial candidate set and scores the candidates again "
            "to improve their ordering."
        ),
        "chunk_reranking_002": (
            "A reranker can use a more expensive relevance model after "
            "fast retrieval has produced a smaller candidate set. "
            "This separates broad candidate retrieval from precise ranking."
        ),

        # ---------------------------------------------------------
        # Multimodal RAG
        # ---------------------------------------------------------

        "chunk_multimodal_rag_001": (
            "Multimodal RAG extends retrieval augmented generation "
            "beyond plain text by allowing systems to retrieve and "
            "reason over content such as images, tables, audio, and video."
        ),
        "chunk_multimodal_rag_002": (
            "A multimodal knowledge system can preserve relationships "
            "between text, images, tables, and temporal media so that "
            "retrieval can return the relevant information together."
        ),

        # ---------------------------------------------------------
        # Relationships
        # ---------------------------------------------------------

        "chunk_relationships_001": (
            "Document elements can have structural relationships such "
            "as parent-child, contains, follows, and belongs-to. "
            "These relationships help preserve the original structure."
        ),
        "chunk_relationships_002": (
            "Spatial relationships can connect elements that appear "
            "near each other on a document page or presentation slide. "
            "Temporal relationships can connect events or media elements "
            "that occur near each other in time."
        ),
        "chunk_relationships_003": (
            "Relationship information can help a retrieval system "
            "recover context that would otherwise be lost when a "
            "multimodal document is split into independent chunks."
        ),

        # ---------------------------------------------------------
        # Distractors
        # ---------------------------------------------------------

        "chunk_postgres_001": (
            "PostgreSQL is a relational database system that supports "
            "structured data, SQL queries, indexes, transactions, "
            "and extensions."
        ),
        "chunk_redis_001": (
            "Redis is an in-memory data store commonly used for "
            "caching, queues, sessions, and other low-latency workloads."
        ),
        "chunk_docker_001": (
            "Docker packages applications and their dependencies "
            "into containers so that software can run consistently "
            "across development and deployment environments."
        ),
        "chunk_websocket_001": (
            "WebSockets provide a persistent communication channel "
            "between a client and server and can support real-time "
            "updates without repeatedly opening new HTTP requests."
        ),
        "chunk_auth_001": (
            "Authentication verifies the identity of a user or system, "
            "while authorization determines which resources or actions "
            "that authenticated identity is allowed to access."
        ),
        "chunk_testing_001": (
            "Automated tests verify that individual components and "
            "larger workflows behave as expected. Unit and integration "
            "tests help detect regressions during development."
        ),
    }
from __future__ import annotations

import uuid

from redis import Redis
from rq import Queue

from backend.app.core.config import settings


class DocumentJobQueue:
    def __init__(
        self,
        redis_url: str | None = None,
        queue_name: str = "document_ingestion",
    ) -> None:
        self.redis = Redis.from_url(
            redis_url or settings.redis_url
        )

        self.queue = Queue(
            name=queue_name,
            connection=self.redis,
        )

    def enqueue(
        self,
        document_id: str,
        *,
        retry: bool = False,
    ):
        from backend.app.workers.document_worker import (
            process_document_job,
        )

        if retry:
            job_id = (
                f"document-{document_id}-retry-"
                f"{uuid.uuid4().hex}"
            )
        else:
            job_id = (
                f"document-{document_id}"
            )

        return self.queue.enqueue(
            process_document_job,
            document_id,
            job_id=job_id,
            job_timeout="30m",
            result_ttl=86400,
        )

    def close(self) -> None:
        self.redis.close()
# import requests
# from bs4 import BeautifulSoup

# from backend.app.ingestion.models.source_document import SourceDocument
# from backend.app.ingestion.models.content import TextContent

# def load_web_page(
#     url: str,
#     document_id: str,
# ) -> list[SourceDocument]:
#     response = requests.get(
#         url,
#         timeout=10,
#         headers={
#             "User-Agent": "Knowledge-Intelligence-Assistant/1.0",
#         },
#     )

#     response.raise_for_status()

#     content_type = response.headers.get("content-type", "")

#     if "text/html" not in content_type:
#         raise ValueError(
#             f"Expected an HTML page, got: {content_type}"
#         )

#     soup = BeautifulSoup(response.text, "html.parser")

#     for element in soup(["script", "style", "noscript"]):
#         element.decompose()

#     content = soup.get_text(
#         separator="\n",
#         strip=True,
#     )

#     if not content:
#         return []

#     title = soup.title.get_text(strip=True) if soup.title else None

#     metadata = {
#         "url": url,
#         "content_type": "webpage",
#     }

#     if title:
#         metadata["title"] = title

#     return [
#         SourceDocument(
#             content=TextContent(text=content),
#             document_id=document_id,
#             source_type="web",
#             metadata=metadata,
#         )
#     ]

from __future__ import annotations

import requests

from backend.app.ingestion.loaders.html_loader import (
    _load_html_content,
)


def load_web(
    url: str,
    document_id: str,
) :
    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": (
                "KnowledgeIntelligenceAssistant/0.1"
            )
        },
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    if "text/html" not in content_type:
        raise ValueError(
            "URL did not return HTML content."
        )

    return _load_html_content(
        html=response.text,
        document_id=document_id,
        file_name=url,
        base_url=url,
    )
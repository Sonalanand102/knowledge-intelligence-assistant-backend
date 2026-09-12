# from unittest.mock import Mock, patch

# from backend.app.ingestion.loaders.web_loader import load_web_page
# from backend.app.ingestion.models.content import TextContent


# def test_load_web_page():
#     html = """
#     <!DOCTYPE html>
#     <html>
#         <head>
#             <title>Knowledge Intelligence Assistant</title>

#             <script>
#                 console.log("Ignore this");
#             </script>

#             <style>
#                 body { color: red; }
#             </style>
#         </head>

#         <body>
#             <h1>Knowledge Intelligence Assistant</h1>

#             <p>
#                 This is a multi-source RAG application.
#             </p>

#             <h2>Supported Sources</h2>

#             <p>
#                 PDF, DOCX, CSV, Excel, YouTube and web pages.
#             </p>
#         </body>
#     </html>
#     """

#     mock_response = Mock()
#     mock_response.text = html
#     mock_response.headers = {
#         "content-type": "text/html; charset=utf-8"
#     }

#     with patch(
#         "backend.app.ingestion.loaders.web_loader.requests.get",
#         return_value=mock_response,
#     ):
#         documents = load_web_page(
#             "https://example.com/test",
#             document_id="web-1",
#         )

#     assert len(documents) == 1

#     document = documents[0]

#     assert isinstance(document.content, TextContent)
#     assert document.document_id == "web-1"
#     assert document.source_type == "web"

#     assert document.metadata["url"] == "https://example.com/test"
#     assert document.metadata["content_type"] == "webpage"
#     assert document.metadata["title"] == "Knowledge Intelligence Assistant"

#     assert ("Knowledge Intelligence Assistant" in document.content.text)
#     assert ("multi-source RAG application" in document.content.text)
#     assert ("Supported Sources" in document.content.text)

#     assert ("Ignore this" not in document.content.text)
#     assert ("color: red" not in document.content.text)


from unittest.mock import Mock, patch

from backend.app.ingestion.loaders.web_loader import (
    load_web,
)


def test_load_web_extracts_html():
    html = """
    <html>
        <body>
            <article>
                <h1>Knowledge Assistant</h1>
                <p>Hello from the web.</p>
                <a href="/docs">Documentation</a>
            </article>
        </body>
    </html>
    """

    response = Mock()

    response.status_code = 200
    response.text = html
    response.headers = {
        "content-type": "text/html; charset=utf-8"
    }

    response.raise_for_status.return_value = None

    with patch(
        "backend.app.ingestion.loaders.web_loader.requests.get",
        return_value=response,
    ) as mock_get:
        result = load_web(
            url="https://example.com/page",
            document_id="web-123",
        )

    mock_get.assert_called_once()

    element_types = {
        element.element_type
        for element in result.elements
    }

    assert "heading" in element_types
    assert "paragraph" in element_types
    assert "link" in element_types

    links = [
        element
        for element in result.elements
        if element.element_type == "link"
    ]

    assert (
        links[0].metadata["url"]
        == "https://example.com/docs"
    )


def test_load_web_rejects_non_html():
    response = Mock()

    response.status_code = 200
    response.text = "not html"
    response.headers = {
        "content-type": "application/json"
    }

    response.raise_for_status.return_value = None

    with patch(
        "backend.app.ingestion.loaders.web_loader.requests.get",
        return_value=response,
    ):
        try:
            load_web(
                url="https://example.com/api",
                document_id="web-123",
            )
        except ValueError:
            return

    raise AssertionError(
        "Expected ValueError"
    )


def test_load_web_propagates_http_errors():
    response = Mock()

    response.raise_for_status.side_effect = (
        RuntimeError("HTTP error")
    )

    with patch(
        "backend.app.ingestion.loaders.web_loader.requests.get",
        return_value=response,
    ):
        try:
            load_web(
                url="https://example.com",
                document_id="web-123",
            )
        except RuntimeError:
            return

    raise AssertionError(
        "Expected HTTP error"
    )
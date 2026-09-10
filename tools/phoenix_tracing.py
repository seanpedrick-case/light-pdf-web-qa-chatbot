"""Arize Phoenix (local) OpenTelemetry tracing for Gradio chat turns.

Call :func:`setup_phoenix_tracing` once at app startup. Emit OpenInference-style
spans around retrieval and generation (this app is not LangChain-based).

Environment
-----------
- ``ARIZE_TRACING_ENABLED`` — ``true`` / ``1`` / ``yes`` / ``on`` to enable
- ``ARIZE_BACKEND`` — ``phoenix`` (default for this app) or ``ax``
- ``PHOENIX_COLLECTOR_ENDPOINT`` — default ``http://localhost:6006``
- ``PHOENIX_PROJECT_NAME`` / ``ARIZE_PROJECT_NAME`` — default ``light-pdf-qa-chatbot``
- ``PHOENIX_API_KEY`` — optional for secured Phoenix
- AX (``ARIZE_BACKEND=ax``): ``ARIZE_SPACE_ID`` / ``ARIZE_API_KEY``; optional
  ``ARIZE_ENDPOINT`` ``europe`` (default) or ``us``

Multi-turn chats: set ``session.id`` from Gradio ``session_hash`` so turns share
one Sessions row in Phoenix.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

_INITIALIZED = False
_TRACER_NAME = "light_pdf_qa_chatbot"
_MAX_ATTR_CHARS = 4000


def _env_truthy(name: str, default: str = "") -> bool:
    return (os.environ.get(name) or default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tracing_backend() -> str:
    raw = (os.environ.get("ARIZE_BACKEND") or "phoenix").strip().lower()
    if raw in {"ax", "arize", "cloud"}:
        return "ax"
    return "phoenix"


def _project_name() -> str:
    return (
        os.environ.get("PHOENIX_PROJECT_NAME")
        or os.environ.get("ARIZE_PROJECT_NAME")
        or "light-pdf-qa-chatbot"
    ).strip()


def _phoenix_otlp_endpoint(collector: str) -> str:
    ep = collector.strip().rstrip("/")
    if not ep:
        ep = "http://localhost:6006"
    if ep.endswith("/v1/traces") or ep.endswith(":4317"):
        return ep
    return f"{ep}/v1/traces"


def _truncate(value: str, limit: int = _MAX_ATTR_CHARS) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _json_attr(value: Any) -> str:
    try:
        return _truncate(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        return _truncate(str(value))


def tracing_initialized() -> bool:
    return _INITIALIZED


def arize_session_id(session_hash: str | None) -> str | None:
    sid = (session_hash or "").strip()
    return sid or None


def _setup_phoenix_tracing(project_name: str) -> bool:
    try:
        from phoenix.otel import register
    except ImportError as exc:
        warnings.warn(
            f"Phoenix tracing dependencies unavailable ({exc}); tracing skipped. "
            "Install arize-phoenix-otel.",
            UserWarning,
            stacklevel=3,
        )
        return False

    collector = (
        os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or "http://localhost:6006"
    ).strip()
    endpoint = _phoenix_otlp_endpoint(collector)
    register_kwargs: dict[str, Any] = {
        "project_name": project_name,
        "endpoint": endpoint,
    }
    api_key = (os.environ.get("PHOENIX_API_KEY") or "").strip()
    if api_key:
        register_kwargs["api_key"] = api_key

    register(**register_kwargs)
    print(f"Phoenix tracing enabled: project={project_name}, endpoint={endpoint}")
    return True


def _setup_ax_tracing(project_name: str) -> bool:
    space_id = (os.environ.get("ARIZE_SPACE_ID") or "").strip()
    api_key = (os.environ.get("ARIZE_API_KEY") or "").strip()
    if not space_id or not api_key:
        warnings.warn(
            "ARIZE_TRACING_ENABLED is set but ARIZE_SPACE_ID / ARIZE_API_KEY "
            "are missing; Arize AX tracing is skipped.",
            UserWarning,
            stacklevel=3,
        )
        return False

    try:
        from arize.otel import Endpoint, register
    except ImportError as exc:
        warnings.warn(
            f"Arize AX tracing dependencies unavailable ({exc}); tracing skipped. "
            "Install arize-otel.",
            UserWarning,
            stacklevel=3,
        )
        return False

    endpoint_raw = (os.environ.get("ARIZE_ENDPOINT") or "europe").strip().lower()
    if endpoint_raw in {"us", "arize", "default"}:
        endpoint = Endpoint.ARIZE
    else:
        endpoint = Endpoint.ARIZE_EUROPE

    register(
        space_id=space_id,
        api_key=api_key,
        project_name=project_name,
        endpoint=endpoint,
    )
    print(f"Arize AX tracing enabled: project={project_name}")
    return True


def setup_phoenix_tracing() -> bool:
    """Register Phoenix (or Arize AX) tracer for custom chat spans.

    Returns True if tracing was enabled (or already initialized). Soft-fails
    when disabled, partially configured, or packages are missing.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True

    if not _env_truthy("ARIZE_TRACING_ENABLED"):
        return False

    project_name = _project_name()
    backend = _tracing_backend()
    if backend == "phoenix":
        ok = _setup_phoenix_tracing(project_name)
    else:
        ok = _setup_ax_tracing(project_name)

    if ok:
        _INITIALIZED = True
    return ok


@contextmanager
def arize_session_context(session_hash: str | None) -> Iterator[None]:
    """Legacy no-op shim.

    Gradio resumes generators across threads/contexts, so attaching OpenInference
    session baggage via ContextVars is unsafe. Session id is set as a span
    attribute instead (see :func:`chat_span`).
    """
    yield


def _kind_value(kind: str) -> str:
    try:
        from openinference.semconv.trace import OpenInferenceSpanKindValues

        return OpenInferenceSpanKindValues[kind].value
    except (ImportError, KeyError):
        return kind


def _apply_span_base_attrs(
    span: Any,
    *,
    kind: str,
    session_hash: str | None,
    input_value: str | None,
    attributes: Optional[dict[str, Any]],
) -> None:
    try:
        from openinference.semconv.trace import SpanAttributes
    except ImportError:
        SpanAttributes = None  # type: ignore

    if SpanAttributes is not None:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, _kind_value(kind))
        sid = arize_session_id(session_hash)
        if sid:
            span.set_attribute(SpanAttributes.SESSION_ID, sid)
        if input_value is not None:
            span.set_attribute(SpanAttributes.INPUT_VALUE, _truncate(str(input_value)))
    if attributes:
        for key, value in attributes.items():
            if value is None:
                continue
            if isinstance(value, (bool, int, float)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, _truncate(str(value)))


def start_chat_span(
    name: str,
    *,
    kind: str,
    session_hash: str | None = None,
    input_value: str | None = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Any:
    """Start a span without attaching it as the current ContextVar span.

    Safe across Gradio generator yields / thread hops. Caller must
    :func:`end_chat_span`.
    """
    if not _INITIALIZED:
        return None
    try:
        from opentelemetry import trace
    except ImportError:
        return None

    tracer = trace.get_tracer(_TRACER_NAME)
    # Do not use start_as_current_span: Gradio may resume work in another
    # context, which breaks ContextVar detach ("Token was created in a
    # different Context").
    span = tracer.start_span(name)
    _apply_span_base_attrs(
        span,
        kind=kind,
        session_hash=session_hash,
        input_value=input_value,
        attributes=attributes,
    )
    return span


def end_chat_span(span: Any, *, error: BaseException | None = None) -> None:
    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        span.end()
        return
    if error is not None:
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
    span.end()


@contextmanager
def chat_span(
    name: str,
    *,
    kind: str,
    session_hash: str | None = None,
    input_value: str | None = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Iterator[Any]:
    """Open an OpenInference span; no-op when tracing is off.

    Uses non-attached spans so Gradio thread/context hops do not break OTEL
    ContextVar detach. Prefer this for ordinary (non-generator) call sites;
    for generators prefer :func:`start_chat_span` / :func:`end_chat_span`.

    ``kind`` should be an OpenInference span kind string such as ``CHAIN``,
    ``RETRIEVER``, or ``LLM``.
    """
    span = start_chat_span(
        name,
        kind=kind,
        session_hash=session_hash,
        input_value=input_value,
        attributes=attributes,
    )
    try:
        yield span
    except Exception as exc:
        end_chat_span(span, error=exc)
        raise
    else:
        end_chat_span(span)


def set_span_output(span: Any, output_value: str | None) -> None:
    if span is None or output_value is None:
        return
    try:
        from openinference.semconv.trace import SpanAttributes
    except ImportError:
        return
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, _truncate(str(output_value)))


def set_retrieval_documents(span: Any, passages: list[dict[str, Any]] | None) -> None:
    """Attach scored passage text using OpenInference retrieval.document.* attrs."""
    if span is None or not passages:
        return
    try:
        from openinference.semconv.trace import DocumentAttributes, SpanAttributes
    except ImportError:
        # Fallback: single JSON attribute for collectors without OpenInference semconv.
        set_span_attr(span, "qa.scored_passages", _json_attr(passages))
        return

    for i, passage in enumerate(passages):
        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
        text = passage.get("text")
        if text is not None:
            span.set_attribute(
                prefix + DocumentAttributes.DOCUMENT_CONTENT,
                _truncate(str(text)),
            )
        score = passage.get("score")
        if isinstance(score, (int, float)):
            span.set_attribute(prefix + DocumentAttributes.DOCUMENT_SCORE, float(score))
        source = passage.get("source") or passage.get("id")
        if source:
            span.set_attribute(
                prefix + DocumentAttributes.DOCUMENT_ID, _truncate(str(source))
            )
        metadata = passage.get("metadata")
        if metadata:
            span.set_attribute(
                prefix + DocumentAttributes.DOCUMENT_METADATA,
                _json_attr(metadata),
            )

    set_span_attr(span, "qa.scored_passages_count", len(passages))


def set_span_attr(span: Any, key: str, value: Any) -> None:
    if span is None or value is None:
        return
    if isinstance(value, (bool, int, float)):
        span.set_attribute(key, value)
    else:
        span.set_attribute(key, _truncate(str(value)))

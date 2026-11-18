"""HTML rendering for the Gradio Source paragraphs panel (Bedrock KB + local FAISS)."""

from __future__ import annotations

import html
import re
from typing import Any, Literal
from urllib.parse import urlparse

from tools.config import KB_SOURCE_LABEL_METADATA_KEY_LIST

PASSAGES_PLACEHOLDER = "No relevant source paragraphs currently loaded"

_SOURCES_PANEL_CSS = """
<style>
.kb-src-panel { font-family: inherit; font-size: 0.95em; line-height: 1.45; }
.kb-src-chunk { margin: 0 0 0.65em 0; border: 1px solid var(--border-color-primary, #e5e7eb);
  border-radius: 6px; padding: 0.35em 0.6em; background: var(--background-fill-secondary, #f9fafb); }
.kb-src-chunk summary { cursor: pointer; font-weight: 600; list-style: revert; }
.kb-src-chunk summary::-webkit-details-marker { font-size: 0.85em; }
.kb-src-passage { margin-top: 0.5em; padding-top: 0.45em; border-top: 1px solid var(--border-color-primary, #e5e7eb);
  font-weight: normal; color: inherit; }
.kb-src-outcome { margin: 0 0 0.75em 0; padding: 0.55em 0.7em; border-radius: 6px;
  border: 1px solid #f59e0b; background: #fffbeb; font-size: 0.92em; line-height: 1.4; }
.kb-src-outcome-detail { display: block; margin-top: 0.35em; color: #57534e; }
</style>
"""


def format_score(score: Any) -> str:
    if score is None:
        return "n/a"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def _value_non_empty(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text)


def _truncate_label(text: str, max_len: int = 72) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _source_basename(source: str) -> str:
    source = (source or "").strip()
    if not source:
        return ""
    if source.startswith("s3://"):
        return source.rstrip("/").split("/")[-1]
    if source.startswith("http://") or source.startswith("https://"):
        path = urlparse(source).path.rstrip("/")
        if path and path != "/":
            return path.split("/")[-1] or urlparse(source).netloc
        host = urlparse(source).netloc
        return host.removeprefix("www.") if host else source
    normalized = source.replace("\\", "/")
    return normalized.rstrip("/").split("/")[-1] or source


def _url_host_label(source: str) -> str:
    source = (source or "").strip()
    if not source:
        return ""
    if source.startswith("http://") or source.startswith("https://"):
        host = urlparse(source).netloc
        return host.removeprefix("www.") if host else ""
    if not source.startswith("s3://") and "." in source and "/" not in source:
        return source.removeprefix("www.")
    return ""


def short_label(passage: dict[str, Any], mode: Literal["bedrock", "local"]) -> str:
    meta = passage.get("metadata") or {}
    source = str(
        passage.get("source") or meta.get("source") or meta.get("meta_url") or ""
    )

    if mode == "bedrock":
        for key in KB_SOURCE_LABEL_METADATA_KEY_LIST:
            if _value_non_empty(meta.get(key)):
                return _truncate_label(str(meta[key]))
        base = _source_basename(source)
        if base:
            return _truncate_label(base)
        return "Retrieved passage"

    # local
    if _value_non_empty(meta.get("page")):
        return f"Page {meta['page']}"
    for key in ("row", "row_section"):
        if _value_non_empty(meta.get(key)):
            return f"Row {meta[key]}"
    if _value_non_empty(meta.get("section")):
        return _truncate_label(str(meta["section"]))
    base = _source_basename(source)
    if base:
        return _truncate_label(base)
    host = _url_host_label(source)
    if host:
        return _truncate_label(host)
    return "Retrieved passage"


def _summary_line(
    rank: int, passage: dict[str, Any], mode: Literal["bedrock", "local"]
) -> str:
    score_s = format_score(passage.get("score"))
    label = html.escape(short_label(passage, mode))
    return f"Source {rank} · {score_s} · {label}"


def _passage_body_html(text: str) -> str:
    escaped = html.escape(str(text or ""))
    return escaped.replace("\n", "<br>")


def passages_to_sources_html(
    passages: list[dict[str, Any]],
    *,
    mode: Literal["bedrock", "local"] = "bedrock",
) -> str:
    """Render scored retrieval passages as collapsible HTML chunks."""
    if not passages:
        return PASSAGES_PLACEHOLDER

    chunks: list[str] = []
    for i, passage in enumerate(passages, 1):
        open_attr = " open" if i == 1 else ""
        summary = _summary_line(i, passage, mode)
        body = _passage_body_html(passage.get("text") or "")
        chunks.append(
            f'<details class="kb-src-chunk"{open_attr}>'
            f"<summary>{summary}</summary>"
            f'<div class="kb-src-passage">{body}</div>'
            f"</details>"
        )

    return _SOURCES_PANEL_CSS + f'<div class="kb-src-panel">{"".join(chunks)}</div>'

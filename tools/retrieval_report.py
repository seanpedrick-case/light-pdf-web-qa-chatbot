"""Format hybrid retrieval diagnostics for the UI and logs."""

from __future__ import annotations

import html
import json
import logging
from typing import Any, List, Optional

from tools.bedrock_outcome import (
    diagnose_bedrock_kb_outcome,
    outcome_diagnosis_markdown_lines,
)
from tools.config import LOGGING
from tools.document import Document

logger = logging.getLogger(__name__)


def _format_score(score: Any) -> str:
    if score is None:
        return "n/a"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def doc_source(doc: Document) -> str:
    return (getattr(doc, "metadata", None) or {}).get("source", "") or ""


def faiss_index_size(vectorstore) -> int:
    index = getattr(vectorstore, "index", None)
    if index is not None and hasattr(index, "ntotal"):
        return int(index.ntotal)
    return 0


def build_vector_hits(
    docs: list,
    *,
    vec_score_cut_off: float,
    min_chars: int = 100,
) -> List[dict]:
    """Annotate FAISS hits with filter outcome."""
    hits: List[dict] = []
    for rank, (doc, cosine) in enumerate(docs, 1):
        chars = len(getattr(doc, "page_content", "") or "")
        passed_score = float(cosine) > vec_score_cut_off
        passed_length = chars >= min_chars
        flags: List[str] = []
        if not passed_score:
            flags.append(f"below cosine cut-off ({vec_score_cut_off})")
        if not passed_length:
            flags.append(f"shorter than {min_chars} chars")
        hits.append(
            {
                "rank": rank,
                "cosine_score": float(cosine),
                "source": doc_source(doc),
                "chars": chars,
                "kept": passed_score and passed_length,
                "filter_note": "; ".join(flags) if flags else "",
            }
        )
    return hits


def format_retrieval_details_markdown(report: Optional[dict]) -> str:
    """Render retrieval pipeline summary for the diagnostics accordion (Markdown)."""
    if not report:
        return "_No local retrieval diagnostics for this turn._"

    lines: List[str] = ["### Retrieval pipeline", ""]

    vector = report.get("vector") or {}
    lines.extend(
        [
            "#### 1. Query embedding + vector search",
            f"- **Embeddings model:** {vector.get('embeddings_model', 'n/a')}",
            f"- **Index chunks:** {vector.get('index_size', 'n/a')}",
            f"- **Compared (top-k):** {vector.get('k_requested', 'n/a')}",
            f"- **Metric:** {vector.get('metric', 'cosine similarity')}",
            f"- **After score (>{vector.get('score_cut_off')}) and length filters:** "
            f"{vector.get('candidates_after_filters', 0)} chunk(s)",
            "",
        ]
    )

    vector_hits = vector.get("hits") or []
    if vector_hits:
        lines.append("**Vector shortlist (FAISS order)**")
        for h in vector_hits:
            keep = "kept" if h.get("kept") else "dropped"
            note = h.get("filter_note") or ""
            suffix = f" — {note}" if note else ""
            lines.append(
                f"- #{h['rank']}: cosine `{_format_score(h.get('cosine_score'))}` "
                f"({keep}){suffix}"
            )
            lines.append(f"  - `{h.get('source', '')}` [{h.get('chars', 0)} chars]")
        lines.append("")

    hybrid = report.get("hybrid")
    if hybrid:
        lines.extend(
            [
                "#### 2. Hybrid fusion (vector rank + BM25 + SVM weights)",
                f"- **Pool size:** {hybrid.get('pool_size')}",
                f"- **Shortlist for rerank/output:** {hybrid.get('shortlist_size')}",
                "",
            ]
        )
        for row in hybrid.get("candidates") or []:
            sel = " → **selected**" if row.get("selected") else ""
            lines.append(
                f"- Fusion rank {row.get('fusion_rank')}: "
                f"score `{_format_score(row.get('fusion_score'))}`, "
                f"cosine `{_format_score(row.get('cosine_score'))}`{sel}"
            )
            lines.append(f"  - `{row.get('source', '')}`")
        lines.append("")

    rerank = report.get("rerank")
    if rerank:
        lines.append("#### 3. Reranker")
        if rerank.get("skipped"):
            lines.append(f"- {rerank.get('reason', 'skipped')}")
        elif rerank.get("enabled") and rerank.get("results"):
            lines.extend(
                [
                    f"- **Model:** {rerank.get('model', 'n/a')}",
                    f"- **Candidates:** {rerank.get('candidates', 0)}",
                    f"- **Returned:** {len(rerank.get('results') or [])}",
                    "",
                ]
            )
            for row in rerank.get("results") or []:
                lines.append(
                    f"- Rerank #{row.get('rank')}: "
                    f"relevance `{_format_score(row.get('relevance_score'))}`"
                )
                lines.append(f"  - `{row.get('source', '')}`")
        elif not rerank.get("enabled"):
            lines.append("- Disabled (`ENABLE_RERANKER=False`)")
        lines.append("")

    final = report.get("final") or []
    if final:
        lines.append("#### Final passages sent to the model")
        for row in final:
            lines.append(
                f"- #{row.get('rank')}: score `{_format_score(row.get('score'))}` "
                f"({row.get('score_kind', 'final')}) — `{row.get('source', '')}`"
            )

    return "\n".join(lines).strip()


def format_bedrock_kb_retrieval_details_markdown(result: Optional[dict]) -> str:
    """Bedrock KB retrieval steps for the diagnostics accordion (Markdown)."""
    from tools.bedrock_kb_filter import kb_metadata_filter_enabled
    from tools.config import (
        KB_METADATA_FILTER_KEY_SET,
        USE_KB_METADATA_FILTER,
    )

    if not result:
        return "_No Bedrock KB retrieval diagnostics for this turn._"

    lines: List[str] = ["### Bedrock Knowledge Base retrieval", ""]

    outcome = result.get("outcome_diagnosis") or diagnose_bedrock_kb_outcome(result)
    lines.extend(outcome_diagnosis_markdown_lines(outcome))

    query = result.get("user_message")
    if query:
        lines.append("#### Question sent to Bedrock")
        lines.append(f"- `{query}`")
        lines.append("")

    filter_feature_on = kb_metadata_filter_enabled(USE_KB_METADATA_FILTER)
    lines.append("#### 1. Metadata filter")
    if not filter_feature_on:
        lines.append("- **Auto filter:** disabled (`USE_KB_METADATA_FILTER` is off)")
    elif not KB_METADATA_FILTER_KEY_SET:
        lines.append(
            "- **Auto filter:** enabled but inactive (`KB_METADATA_FILTER_KEYS` is empty)"
        )
    elif result.get("metadata_filter_generation_attempted"):
        if result.get("metadataFilterApplied"):
            lines.append("- **Auto filter:** inferred and **applied** to retrieval")
        else:
            lines.append(
                "- **Auto filter:** generation ran; **no filter** used "
                "(model returned null or filter failed validation)"
            )
    elif result.get("metadata_filter_explicit"):
        if result.get("metadataFilterApplied"):
            lines.append("- **Filter:** supplied explicitly and **applied**")
        else:
            lines.append("- **Filter:** explicit filter was empty or not applied")
    else:
        lines.append("- **Auto filter:** not run for this turn")

    filt = result.get("metadata_filter")
    if filt:
        lines.append("")
        lines.append("**RetrievalFilter sent to Bedrock:**")
        lines.append("```json")
        lines.append(json.dumps(filt, indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("- **RetrievalFilter:** none (semantic search over full index)")
    lines.append("")

    lines.extend(
        [
            "#### 2. Retrieve API (sources panel)",
            f"- **Chunks returned:** {result.get('passages_count', 0)} "
            f"(requested up to {result.get('number_of_results_requested', 'n/a')})",
        ]
    )
    top = result.get("top_score")
    if top is not None:
        lines.append(f"- **Top relevance score:** `{_format_score(top)}`")
    if result.get("retrieve_error"):
        lines.append(f"- **Retrieve error:** {result['retrieve_error']}")
    lines.append(
        "- _Counts are retrieved text chunks after vector search, not total "
        "documents in the knowledge base. A metadata filter only restricts which "
        "chunks are eligible before ranking._"
    )
    lines.append("")

    lines.extend(
        [
            "#### 3. Retrieve and generate",
            f"- **Citation groups in answer:** {result.get('citations_count', 0)}",
        ]
    )
    if result.get("rag_skipped_after_retrieve_failure"):
        lines.append(
            "- **RetrieveAndGenerate:** skipped (Retrieve failed with the same filter)"
        )
    elif result.get("success") is False and result.get("generate_error"):
        lines.append("- **RetrieveAndGenerate:** failed before a response was returned")
    if result.get("generate_error"):
        lines.append(f"- **Error:** {result['generate_error']}")
    action = result.get("guardrail_action")
    if action:
        lines.append(f"- **Guardrail action (API):** `{action}`")
    if result.get("guardrail"):
        lines.append("- **Guardrail:** intervened on this response")
    elif action:
        lines.append("- **Guardrail:** did not intervene on this response")

    return "\n".join(lines).strip()


def format_retrieval_details_html(report: Optional[dict]) -> str:
    """HTML variant (legacy); prefer markdown for the UI."""
    md = format_retrieval_details_markdown(report)
    if not report:
        return ""
    escaped = html.escape(md).replace("\n", "<br>")
    return f'<div class="retrieval-details" style="font-size:0.92em;">{escaped}</div><hr><br>'


def log_retrieval_report(report: Optional[dict]) -> None:
    if LOGGING != "True" or not report:
        return
    try:
        payload = json.dumps(report, default=str, indent=2)
    except TypeError:
        payload = str(report)
    logger.info("Hybrid retrieval report:\n%s", payload)

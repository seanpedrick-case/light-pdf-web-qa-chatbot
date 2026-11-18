"""Classify Bedrock KB turn outcomes for UI diagnostics (filter vs guardrail vs empty retrieval)."""

from __future__ import annotations

import html
import re
from typing import Any, Optional

from tools.config import KB_DATE_METADATA_KEY_SET, KB_YEAR_METADATA_KEY_SET

# Bedrock / model refusals when guardrail flags are missing or generation had no context.
_REFUSAL_PHRASES = (
    "unable to assist",
    "can't assist",
    "cannot assist",
    "i am not able to",
    "i'm not able to",
)

_GUARDRAIL_INTERVENE_MARKERS = ("INTERVENED", "GUARDRAIL_INTERVENED")


def _guardrail_intervened(result: dict[str, Any]) -> bool:
    if result.get("guardrail"):
        return True
    action = str(result.get("guardrail_action") or "").upper()
    return any(m in action for m in _GUARDRAIL_INTERVENE_MARKERS)


def _response_looks_like_refusal(response_text: str) -> bool:
    text = (response_text or "").strip().lower()
    if not text:
        return False
    return any(phrase in text for phrase in _REFUSAL_PHRASES)


def _walk_filter_clauses(node: Any, out: list[dict[str, Any]]) -> None:
    if not isinstance(node, dict):
        return
    for op in (
        "equals",
        "notEquals",
        "greaterThan",
        "greaterThanOrEquals",
        "lessThan",
        "lessThanOrEquals",
        "in",
        "notIn",
        "startsWith",
        "listContains",
    ):
        if op in node and isinstance(node[op], dict):
            clause = dict(node[op])
            clause["_op"] = op
            out.append(clause)
    for key in ("andAll", "orAll"):
        for child in node.get(key) or []:
            _walk_filter_clauses(child, out)


def _filter_clause_warnings(metadata_filter: Any) -> list[str]:
    if not metadata_filter:
        return []
    clauses: list[dict[str, Any]] = []
    _walk_filter_clauses(metadata_filter, clauses)
    warnings: list[str] = []
    date_keys = {k.strip() for k in KB_DATE_METADATA_KEY_SET if k.strip()}

    for clause in clauses:
        key = str(clause.get("key") or "")
        value = clause.get("value")
        op = clause.get("_op", "equals")
        if op != "equals":
            continue
        value_s = str(value).strip() if value is not None else ""
        is_date_key = key in date_keys or "date" in key.lower()
        if is_date_key and re.fullmatch(r"\d{4}", value_s):
            warnings.append(
                f"Filter uses **equals** on `{key}` = `{value_s}` but ingested values "
                f"are often full ISO datetimes (e.g. `{value_s}-01-01T00:00:00Z`). "
                f"Exact match on the year alone usually returns **zero chunks**."
            )
    return warnings


def _filter_uses_year_metadata_keys(metadata_filter: Any) -> list[str]:
    if not metadata_filter or not KB_YEAR_METADATA_KEY_SET:
        return []
    clauses: list[dict[str, Any]] = []
    _walk_filter_clauses(metadata_filter, clauses)
    used: list[str] = []
    for clause in clauses:
        key = str(clause.get("key") or "")
        if key in KB_YEAR_METADATA_KEY_SET and key not in used:
            used.append(key)
    return used


def _filter_api_rejection_hint(error_text: str) -> Optional[str]:
    upper = error_text.upper()
    lower = error_text.lower()
    if "starts_with" in lower or "startswith" in lower:
        return (
            "**startsWith is not supported** on S3 Vectors knowledge bases. Use an "
            "**ISO-8601 year window** instead: `andAll` with `greaterThanOrEquals` "
            "`YYYY-01-01T00:00:00Z` and `lessThan` `(YYYY+1)-01-01T00:00:00Z` on "
            "date keys (same as the Lambda filter). The app rewrites legacy startsWith "
            "filters automatically when possible."
        )
    if "filter value type" in lower or "retrievalfilter" in lower:
        if "GREATER_THAN" in upper or "LESS_THAN" in upper:
            return (
                "Bedrock rejected a **range comparison**. Use **full ISO-8601** values on "
                "keys listed in `KB_DATE_METADATA_KEYS` (see Lambda filter). Bare years and "
                "DD/MM/YYYY are dropped or rewritten before the API call."
            )
        return (
            "Bedrock rejected the **RetrievalFilter** shape or value types. "
            "Compare the JSON below with "
            "[Bedrock RetrievalFilter](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalFilter.html) "
            "and your KB metadata schema."
        )
    return None


def diagnose_bedrock_kb_outcome(result: dict[str, Any]) -> dict[str, Any]:
    """Return headline, explanations, and whether to show a sources-panel alert."""
    passages = int(result.get("passages_count") or 0)
    citations = int(result.get("citations_count") or 0)
    filter_applied = bool(result.get("metadataFilterApplied"))
    retrieve_error = result.get("retrieve_error")
    generate_error = result.get("generate_error")
    response_text = str(result.get("response") or "")
    guardrail_hit = _guardrail_intervened(result)
    guardrail_action = result.get("guardrail_action")
    refusal_like = _response_looks_like_refusal(response_text)

    explanations: list[str] = []
    show_sources_alert = False
    headline: str

    filter_warnings = _filter_clause_warnings(result.get("metadata_filter"))
    for w in filter_warnings:
        explanations.append(w)

    headline = ""

    if generate_error:
        headline = "Bedrock rejected the request (filter or validation error)"
        show_sources_alert = True
        explanations.append(f"**RetrieveAndGenerate error:** `{generate_error}`")
        hint = _filter_api_rejection_hint(str(generate_error))
        if hint:
            explanations.append(hint)
        if result.get("rag_skipped_after_retrieve_failure"):
            explanations.append(
                "**RetrieveAndGenerate was not called** because **Retrieve** failed "
                "with the same filter (see Retrieve error below)."
            )
    elif guardrail_hit:
        headline = "Bedrock Guardrail intervened"
        explanations.append(
            "The Guardrail was applied to **RetrieveAndGenerate** output "
            f"(`guardrailAction`: `{guardrail_action or 'INTERVENED'}`). "
            "The chat text is likely a block or replacement message from the guardrail, "
            "not an answer grounded in your knowledge base."
        )
        show_sources_alert = True
    elif passages == 0 and citations == 0 and filter_applied and retrieve_error:
        headline = "Bedrock rejected the metadata filter on Retrieve"
        show_sources_alert = True
    elif passages == 0 and citations == 0 and filter_applied:
        headline = "No chunks matched the metadata filter"
        explanations.append(
            "Both **Retrieve** and **RetrieveAndGenerate** returned **zero chunks** while a "
            "**RetrievalFilter** was active. The model had little or no indexed text for this "
            "question—answers may be generic refusals."
        )
        year_keys = _filter_uses_year_metadata_keys(result.get("metadata_filter"))
        if year_keys:
            joined = ", ".join(f"`{k}`" for k in year_keys)
            explanations.append(
                f"The filter used {joined}, but **unfiltered Retrieve** on this knowledge base "
                f"returned chunks **without** those keys on metadata (e.g. a raw date field is present "
                f"but the calendar-year column from your schema is missing). That usually means the "
                f"**indexed copy** of the structured source was synced before the sidecar listed "
                f"`fieldsToInclude` for the year columns, or S3 still has an older file. Re-upload "
                f"the source file and its `.metadata.json`, run **Sync**, then confirm a sample chunk "
                f"shows the year keys from `KB_YEAR_METADATA_KEYS`. "
                f"No date-to-year mapping is required if those columns are already on each chunk."
            )
        show_sources_alert = True
    elif passages == 0 and citations == 0:
        headline = "No chunks retrieved from the knowledge base"
        explanations.append(
            "Vector search returned no eligible chunks (no metadata filter, or filter not applied). "
            "Check query wording and index content."
        )
        show_sources_alert = True
    elif passages == 0 and citations > 0:
        headline = "Retrieve returned no scored chunks; citations only"
        explanations.append(
            "**Retrieve** returned 0 passages for the sources panel, but **RetrieveAndGenerate** "
            "still attached citations—the answer may rely on a different retrieval path."
        )
    else:
        headline = "Retrieval and generation completed normally"
        if passages > 0:
            explanations.append(
                f"**{passages}** chunk(s) available for the sources panel; see scores below."
            )

    if retrieve_error:
        explanations.append(
            f"**Retrieve API error:** `{retrieve_error}` — the scored shortlist "
            "for the sources panel may be incomplete."
        )
        if not generate_error:
            hint = _filter_api_rejection_hint(str(retrieve_error))
            if hint:
                explanations.append(hint)

    if passages == 0 and citations == 0 and filter_applied and guardrail_hit:
        explanations.append(
            "**Also:** retrieval returned no chunks—filter or index mismatch may have left "
            "nothing to ground the answer before the guardrail fired."
        )

    if refusal_like and not guardrail_hit:
        explanations.append(
            "The reply text resembles a **policy/capability refusal**. "
            "With no guardrail intervene flag, this is **usually empty retrieval** or model behaviour "
            "without sources—not a separate guardrail block. Check chunk counts above."
        )
        if passages == 0 and not guardrail_hit:
            show_sources_alert = True
    elif refusal_like and guardrail_hit:
        explanations.append(
            "Refusal-style wording is **consistent with a guardrail intervention** (see guardrail action)."
        )
    elif refusal_like and guardrail_action and not guardrail_hit:
        explanations.append(
            f"Response text looks like a refusal, but `guardrailAction` was `{guardrail_action}` — "
            "treat guardrail as **unlikely** unless your guardrail uses a non-standard action value."
        )

    return {
        "headline": headline,
        "explanations": explanations,
        "show_sources_alert": show_sources_alert,
        "guardrail_intervened": guardrail_hit,
        "no_retrieval": passages == 0 and citations == 0,
    }


def outcome_diagnosis_markdown_lines(outcome: Optional[dict[str, Any]]) -> list[str]:
    if not outcome:
        return []
    lines = [
        "#### What happened (outcome)",
        "",
        f"- **Most likely:** {outcome.get('headline', 'Unknown')}",
    ]
    for item in outcome.get("explanations") or []:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def sources_panel_outcome_banner(
    base_html: str, outcome: Optional[dict[str, Any]]
) -> str:
    if not outcome or not outcome.get("show_sources_alert"):
        return base_html
    headline = html.escape(str(outcome.get("headline") or "Outcome issue"))
    detail_parts = outcome.get("explanations") or []
    detail = ""
    if detail_parts:
        # First explanation only in the banner; full list stays in diagnostics.
        detail = html.escape(re.sub(r"\*\*", "", str(detail_parts[0])).replace("`", ""))
    detail_html = (
        f'<br><span class="kb-src-outcome-detail">{detail}</span>' if detail else ""
    )
    return (
        '<div class="kb-src-outcome" role="note">'
        f"<strong>Why there are no sources:</strong> {headline}"
        f"{detail_html}"
        "<br><em>Expand <strong>Bedrock Knowledge Base retrieval</strong> "
        "below the chat for full detail.</em>"
        "</div>" + base_html
    )

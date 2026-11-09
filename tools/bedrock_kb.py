"""
AWS Bedrock Knowledge Base retrieve_and_generate backend.

When USE_BEDROCK_KB=1, the Gradio app calls this instead of local FAISS hybrid
retrieval and separate LLM generation.

Sources panel + scores come from the separate ``Retrieve`` API (which returns
``score``). ``RetrieveAndGenerate`` citations often omit unused chunks and do
not include retrieval scores.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import boto3

from tools.bedrock_kb_filter import (
    generate_metadata_filter,
    kb_metadata_filter_enabled,
    load_filter_group_options,
)
from tools.bedrock_outcome import (
    diagnose_bedrock_kb_outcome,
    sources_panel_outcome_banner,
)
from tools.config import (
    AWS_DEFAULT_REGION,
    AWS_REGION,
    BEDROCK_MODEL_ID,
    GUARDRAIL_ID,
    GUARDRAIL_VERSION,
    KB_DATE_METADATA_KEY_SET,
    KB_FILTER_MODEL_ID,
    KB_FILTER_PROMPT_PATH,
    KB_GROUP_FILTER_KEY,
    KB_GROUP_OPTIONS_PATH,
    KB_METADATA_EXACT_EQUAL_KEY_SET,
    KB_METADATA_FILTER_KEY_SET,
    KB_METADATA_IN_KEY_SET,
    KNOWLEDGE_BASE_ID,
    USE_BEDROCK_KB,
    USE_KB_METADATA_FILTER,
)
from tools.source_display import (
    passages_to_sources_html as render_passages_to_sources_html,
)

REGION = AWS_DEFAULT_REGION or AWS_REGION or "eu-west-2"
MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/{BEDROCK_MODEL_ID}"
NUMBER_OF_RESULTS = 5

_KB_GROUP_FILTER_KEY = (KB_GROUP_FILTER_KEY or "").strip()
_KB_GROUP_OPTIONS = (
    load_filter_group_options(KB_GROUP_OPTIONS_PATH)
    if _KB_GROUP_FILTER_KEY and _KB_GROUP_FILTER_KEY in KB_METADATA_FILTER_KEY_SET
    else frozenset()
)

SYSTEM_PROMPT = """You are a helpful UK local council service assistant for Sample Council.
You help residents with council services including waste collection, council tax,
planning, housing, parking, roads, schools, social care, and general enquiries.
Keep responses concise (under 200 words), friendly, and helpful.
Use bullet points for readability. Include contact details where appropriate.
Answer ONLY based on the following search results. If the answer is not in the
search results, say you don't have that specific information and suggest contacting
the council directly.

$search_results$"""

TOPIC_MAP = {
    "waste-collection": "Bin Collections",
    "council-tax": "Council Tax",
    "planning": "Planning",
    "housing": "Housing",
    "parking": "Parking",
    "roads-highways": "Roads & Highways",
    "contact": "Contact Us",
    "environmental": "Environmental",
    "libraries-leisure": "Libraries & Leisure",
    "benefits-support": "Benefits & Support",
    "elections": "Elections",
    "licensing": "Licensing",
    "schools-education": "Education",
    "adult-social-care": "Social Care",
    "childrens-services": "Children's Services",
    "community-safety": "Community Safety",
    "christmas-bin": "Bin Collections",
    "summer-activities": "Events",
    "flood": "Emergencies",
    "severe-weather": "Emergencies",
    "birth-death-marriage": "Registration",
    "dog-warden": "Animal Services",
    "parks-allotments": "Parks",
    "council-meeting": "Council Meetings",
    "accessibility": "Accessibility",
    "business-rates": "Business Rates",
    "building-control": "Building Control",
}

bedrock_agent = None
bedrock_runtime = None
if USE_BEDROCK_KB == "1":
    bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=REGION)
    if (
        kb_metadata_filter_enabled(USE_KB_METADATA_FILTER)
        and KB_METADATA_FILTER_KEY_SET
    ):
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)


def _exception_message(exc: BaseException) -> str:
    return str(exc)


def _is_filter_validation_error(message: str) -> bool:
    m = message.lower()
    return (
        "filter value type" in m
        or "retrievalfilter" in m
        or ("validationexception" in m and "filter" in m)
    )


def _is_session_retryable_error(exc: BaseException) -> bool:
    msg = _exception_message(exc).lower()
    if _is_filter_validation_error(msg):
        return False
    return "session" in msg


def _log_filter_attempt(
    metadata_filter: Optional[dict[str, Any]], user_message: str
) -> None:
    if metadata_filter:
        print(
            "Bedrock KB RetrievalFilter attempted:\n"
            f"{json.dumps(metadata_filter, indent=2, ensure_ascii=False)}"
        )
    print(f"Bedrock KB query text: {user_message!r}")


def _finalize_bedrock_result(
    *,
    user_message: str,
    base_payload: dict[str, Any],
    passages: list[dict[str, Any]],
) -> dict[str, Any]:
    outcome = diagnose_bedrock_kb_outcome(base_payload)
    sources_html = sources_panel_outcome_banner(
        passages_to_sources_html(passages),
        outcome,
    )
    base_payload["outcome_diagnosis"] = outcome
    base_payload["sources_html"] = sources_html
    base_payload["user_message"] = user_message
    return base_payload


def _bedrock_kb_failure_response(
    *,
    user_message: str,
    session_id: Optional[str],
    metadata_filter: Optional[dict[str, Any]],
    filter_passed_explicitly: bool,
    filter_generation_attempted: bool,
    retrieve_error: Optional[BaseException],
    generate_error: Optional[BaseException] = None,
    rag_skipped_after_retrieve_failure: bool = False,
) -> dict[str, Any]:
    _log_filter_attempt(metadata_filter, user_message)
    response_text = (
        "The knowledge base request could not be completed. "
        "Expand **Bedrock Knowledge Base retrieval** below this chat to see the "
        "**RetrievalFilter** JSON Bedrock received and the exact AWS error message."
    )
    base_payload: dict[str, Any] = {
        "success": False,
        "response": response_text,
        "sessionId": session_id or "",
        "citations": [],
        "citations_count": 0,
        "passages": [],
        "passages_count": 0,
        "retrieval_scores": [],
        "top_score": None,
        "topic": "General Enquiry",
        "guardrail": False,
        "guardrail_action": None,
        "retrieve_error": (
            _exception_message(retrieve_error) if retrieve_error else None
        ),
        "generate_error": (
            _exception_message(generate_error) if generate_error else None
        ),
        "rag_skipped_after_retrieve_failure": rag_skipped_after_retrieve_failure,
        "metadata_filter": metadata_filter,
        "metadataFilterApplied": bool(metadata_filter),
        "metadata_filter_explicit": filter_passed_explicitly,
        "metadata_filter_generation_attempted": filter_generation_attempted,
        "number_of_results_requested": NUMBER_OF_RESULTS,
    }
    return _finalize_bedrock_result(
        user_message=user_message,
        base_payload=base_payload,
        passages=[],
    )


def _vector_search_config(
    number_of_results: int, metadata_filter: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    vsc: dict[str, Any] = {"numberOfResults": number_of_results}
    if metadata_filter:
        vsc["filter"] = metadata_filter
    return vsc


def _resolve_metadata_filter(
    user_message: str, metadata_filter: Optional[dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], bool, bool]:
    """Return (filter, explicitly_passed, generation_attempted)."""
    if metadata_filter is not None:
        return metadata_filter, True, False
    if not (
        kb_metadata_filter_enabled(USE_KB_METADATA_FILTER)
        and KB_METADATA_FILTER_KEY_SET
    ):
        return None, False, False
    model_id = (KB_FILTER_MODEL_ID or "").strip() or BEDROCK_MODEL_ID
    generated = generate_metadata_filter(
        user_message,
        bedrock_runtime=bedrock_runtime,
        model_id=model_id,
        allowed_keys=KB_METADATA_FILTER_KEY_SET,
        date_keys=KB_DATE_METADATA_KEY_SET,
        prompt_path=KB_FILTER_PROMPT_PATH,
        group_filter_key=_KB_GROUP_FILTER_KEY,
        group_options=_KB_GROUP_OPTIONS,
    )
    return generated, False, True


def _ref_location_uri(location: dict) -> str:
    if not location:
        return ""
    for key in (
        "s3Location",
        "webLocation",
        "confluenceLocation",
        "sharePointLocation",
        "salesforceLocation",
        "oneDriveLocation",
        "googleDriveLocation",
        "kendraDocumentLocation",
        "customDocumentLocation",
    ):
        loc = location.get(key) or {}
        uri = loc.get("uri") or loc.get("url") or loc.get("id") or ""
        if uri:
            return str(uri)
    sql = location.get("sqlLocation") or {}
    if sql.get("query"):
        return f"sql:{sql.get('query')}"
    return ""


def get_topic_from_passages(passages: list[dict[str, Any]]) -> str:
    for p in passages:
        fname = (p.get("source") or "").split("/")[-1].replace(".txt", "")
        for key, topic in TOPIC_MAP.items():
            if key in fname:
                return topic
    return "General Enquiry"


def get_topic_from_citations(citations: list) -> str:
    passages = []
    for c in citations:
        for ref in c.get("retrievedReferences", []):
            uri = _ref_location_uri(ref.get("location") or {})
            passages.append({"source": uri})
    return get_topic_from_passages(passages)


def passages_to_sources_html(passages: list[dict[str, Any]]) -> str:
    """Render scored retrieval passages for the sources accordion."""
    return render_passages_to_sources_html(passages, mode="bedrock")


def citations_to_passages(citations: list) -> list[dict[str, Any]]:
    """Fallback passage list from RetrieveAndGenerate citations (no scores)."""
    passages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for citation in citations:
        for ref in citation.get("retrievedReferences", []):
            content_obj = ref.get("content") or {}
            text = content_obj.get("text") or ""
            source = _ref_location_uri(ref.get("location") or {})
            key = f"{source}|{text[:120]}"
            if key in seen:
                continue
            seen.add(key)
            passages.append(
                {
                    "text": text,
                    "source": source,
                    "score": None,
                    "metadata": ref.get("metadata") or {},
                    "origin": "citation",
                }
            )
    return passages


def citations_to_sources_html(citations: list) -> str:
    return passages_to_sources_html(citations_to_passages(citations))


def retrieve_passages(
    user_message: str,
    *,
    number_of_results: int = NUMBER_OF_RESULTS,
    metadata_filter: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Call Bedrock Knowledge Base Retrieve for scored chunks."""
    if bedrock_agent is None:
        raise RuntimeError(
            "Bedrock KB client is not initialised. Set USE_BEDROCK_KB=1 and ensure AWS credentials are available."
        )
    if not KNOWLEDGE_BASE_ID:
        raise ValueError("KNOWLEDGE_BASE_ID is required when USE_BEDROCK_KB=1")

    response = bedrock_agent.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": user_message},
        retrievalConfiguration={
            "vectorSearchConfiguration": _vector_search_config(
                number_of_results, metadata_filter
            )
        },
    )

    passages: list[dict[str, Any]] = []
    for item in response.get("retrievalResults", []):
        content_obj = item.get("content") or {}
        text = content_obj.get("text") or ""
        score = item.get("score")
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None
        passages.append(
            {
                "text": text,
                "source": _ref_location_uri(item.get("location") or {}),
                "score": score_f,
                "metadata": item.get("metadata") or {},
                "document_id": item.get("documentId") or "",
                "origin": "retrieve",
            }
        )

    passages.sort(
        key=lambda p: (
            p["score"] is not None,
            p["score"] if p["score"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return passages


def get_rag_response(
    user_message: str,
    session_id: Optional[str] = None,
    metadata_filter: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Call Bedrock KB Retrieve (scored shortlist) + RetrieveAndGenerate (answer)."""
    if bedrock_agent is None:
        raise RuntimeError(
            "Bedrock KB client is not initialised. Set USE_BEDROCK_KB=1 and ensure AWS credentials are available."
        )
    if not KNOWLEDGE_BASE_ID:
        raise ValueError("KNOWLEDGE_BASE_ID is required when USE_BEDROCK_KB=1")

    metadata_filter, filter_passed_explicitly, filter_generation_attempted = (
        _resolve_metadata_filter(user_message, metadata_filter)
    )
    if metadata_filter is not None:
        from tools.bedrock_kb_filter import prepare_retrieval_filter

        metadata_filter = prepare_retrieval_filter(
            metadata_filter,
            KB_METADATA_FILTER_KEY_SET,
            KB_DATE_METADATA_KEY_SET,
            in_keys=KB_METADATA_IN_KEY_SET or None,
            exact_equal_keys=KB_METADATA_EXACT_EQUAL_KEY_SET or None,
            group_filter_key=_KB_GROUP_FILTER_KEY,
            group_options=_KB_GROUP_OPTIONS,
        )

    vsc = _vector_search_config(NUMBER_OF_RESULTS, metadata_filter)
    _log_filter_attempt(metadata_filter, user_message)

    # Scored shortlist for the sources panel / monitoring (Retrieve API).
    passages: list[dict[str, Any]] = []
    retrieve_error: Optional[BaseException] = None
    try:
        passages = retrieve_passages(user_message, metadata_filter=metadata_filter)
    except Exception as e:
        retrieve_error = e
        print(f"Bedrock Retrieve failed: {e}")
        if _is_filter_validation_error(_exception_message(e)):
            return _bedrock_kb_failure_response(
                user_message=user_message,
                session_id=session_id,
                metadata_filter=metadata_filter,
                filter_passed_explicitly=filter_passed_explicitly,
                filter_generation_attempted=filter_generation_attempted,
                retrieve_error=retrieve_error,
                rag_skipped_after_retrieve_failure=True,
            )

    config: dict[str, Any] = {
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": KNOWLEDGE_BASE_ID,
            "modelArn": MODEL_ARN,
            "retrievalConfiguration": {"vectorSearchConfiguration": vsc},
            "generationConfiguration": {
                "inferenceConfig": {
                    "textInferenceConfig": {
                        "maxTokens": 1024,
                        "temperature": 0.7,
                    }
                },
                "promptTemplate": {"textPromptTemplate": SYSTEM_PROMPT},
            },
        },
    }

    if GUARDRAIL_ID and GUARDRAIL_VERSION:
        config["knowledgeBaseConfiguration"]["generationConfiguration"][
            "guardrailConfiguration"
        ] = {
            "guardrailId": GUARDRAIL_ID,
            "guardrailVersion": GUARDRAIL_VERSION,
        }

    kwargs: dict[str, Any] = {
        "input": {"text": user_message},
        "retrieveAndGenerateConfiguration": config,
    }
    if session_id:
        kwargs["sessionId"] = session_id

    try:
        response = bedrock_agent.retrieve_and_generate(**kwargs)
    except Exception as e:
        if session_id and _is_session_retryable_error(e):
            print(f"Session error, retrying without sessionId: {e}")
            kwargs.pop("sessionId", None)
            try:
                response = bedrock_agent.retrieve_and_generate(**kwargs)
            except Exception as retry_exc:
                return _bedrock_kb_failure_response(
                    user_message=user_message,
                    session_id=None,
                    metadata_filter=metadata_filter,
                    filter_passed_explicitly=filter_passed_explicitly,
                    filter_generation_attempted=filter_generation_attempted,
                    retrieve_error=retrieve_error,
                    generate_error=retry_exc,
                )
        else:
            return _bedrock_kb_failure_response(
                user_message=user_message,
                session_id=session_id,
                metadata_filter=metadata_filter,
                filter_passed_explicitly=filter_passed_explicitly,
                filter_generation_attempted=filter_generation_attempted,
                retrieve_error=retrieve_error,
                generate_error=e,
            )

    citations = response.get("citations", [])
    if not passages:
        passages = citations_to_passages(citations)

    topic = get_topic_from_passages(passages)
    if topic == "General Enquiry":
        topic = get_topic_from_citations(citations)

    scores = [p.get("score") for p in passages if p.get("score") is not None]
    guardrail_action = response.get("guardrailAction")
    response_text = response["output"]["text"]
    guardrail_intervened = guardrail_action in (
        "INTERVENED",
        "GUARDRAIL_INTERVENED",
    ) or (str(guardrail_action or "").upper().find("INTERVENED") >= 0)

    base_payload: dict[str, Any] = {
        "success": True,
        "response": response_text,
        "sessionId": response.get("sessionId", ""),
        "citations": citations,
        "citations_count": len(citations),
        "passages": passages,
        "passages_count": len(passages),
        "retrieval_scores": scores,
        "top_score": scores[0] if scores else None,
        "topic": topic,
        "guardrail": guardrail_intervened,
        "guardrail_action": guardrail_action,
        "retrieve_error": (
            _exception_message(retrieve_error) if retrieve_error else None
        ),
        "generate_error": None,
        "rag_skipped_after_retrieve_failure": False,
        "metadata_filter": metadata_filter,
        "metadataFilterApplied": bool(metadata_filter),
        "metadata_filter_explicit": filter_passed_explicitly,
        "metadata_filter_generation_attempted": filter_generation_attempted,
        "number_of_results_requested": NUMBER_OF_RESULTS,
    }
    return _finalize_bedrock_result(
        user_message=user_message,
        base_payload=base_payload,
        passages=passages,
    )

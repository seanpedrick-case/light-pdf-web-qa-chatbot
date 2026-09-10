"""
AWS Bedrock Knowledge Base retrieve_and_generate backend.

When USE_BEDROCK_KB=1, the Gradio app calls this instead of local FAISS hybrid
retrieval and separate LLM generation.

Sources panel + scores come from the separate ``Retrieve`` API (which returns
``score``). ``RetrieveAndGenerate`` citations often omit unused chunks and do
not include retrieval scores.
"""

from __future__ import annotations

import html
from typing import Any, Optional

import boto3

from tools.config import (
    AWS_DEFAULT_REGION,
    AWS_REGION,
    BEDROCK_MODEL_ID,
    GUARDRAIL_ID,
    GUARDRAIL_VERSION,
    KNOWLEDGE_BASE_ID,
    USE_BEDROCK_KB,
)

REGION = AWS_DEFAULT_REGION or AWS_REGION or "eu-west-2"
MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/{BEDROCK_MODEL_ID}"
NUMBER_OF_RESULTS = 5

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
if USE_BEDROCK_KB == "1":
    bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=REGION)


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


def _format_score(score: Any) -> str:
    if score is None:
        return "n/a"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def passages_to_sources_html(passages: list[dict[str, Any]]) -> str:
    """Render scored retrieval passages for the sources accordion."""
    if not passages:
        return "No relevant source paragraphs currently loaded"

    blocks: list[str] = []
    for i, p in enumerate(passages, 1):
        meta_parts = [
            f"rank: {i}",
            f"score: {_format_score(p.get('score'))}",
        ]
        source = p.get("source") or ""
        if source:
            meta_parts.append(f"source: {html.escape(str(source))}")
        for k, v in (p.get("metadata") or {}).items():
            if k == "page_section":
                continue
            meta_parts.append(f"{html.escape(str(k))}: {html.escape(str(v))}")

        meta_line = "  ".join(meta_parts)
        content = str(p.get("text") or "").replace("\n", "<br>")
        blocks.append(f"{meta_line}.<br><br>{content}")

    return "<br><br>".join(blocks)


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
    user_message: str, *, number_of_results: int = NUMBER_OF_RESULTS
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
            "vectorSearchConfiguration": {"numberOfResults": number_of_results}
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
    user_message: str, session_id: Optional[str] = None
) -> dict[str, Any]:
    """Call Bedrock KB Retrieve (scored shortlist) + RetrieveAndGenerate (answer)."""
    if bedrock_agent is None:
        raise RuntimeError(
            "Bedrock KB client is not initialised. Set USE_BEDROCK_KB=1 and ensure AWS credentials are available."
        )
    if not KNOWLEDGE_BASE_ID:
        raise ValueError("KNOWLEDGE_BASE_ID is required when USE_BEDROCK_KB=1")

    # Scored shortlist for the sources panel / monitoring (Retrieve API).
    passages: list[dict[str, Any]] = []
    retrieve_error = None
    try:
        passages = retrieve_passages(user_message)
    except Exception as e:
        retrieve_error = e
        print(f"Bedrock Retrieve failed (falling back to citations): {e}")

    config: dict[str, Any] = {
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": KNOWLEDGE_BASE_ID,
            "modelArn": MODEL_ARN,
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {"numberOfResults": NUMBER_OF_RESULTS}
            },
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
        if session_id and ("session" in str(e).lower() or "Validation" in str(e)):
            print(f"Session error, retrying without sessionId: {e}")
            kwargs.pop("sessionId", None)
            response = bedrock_agent.retrieve_and_generate(**kwargs)
        else:
            raise

    citations = response.get("citations", [])
    if not passages:
        passages = citations_to_passages(citations)

    topic = get_topic_from_passages(passages)
    if topic == "General Enquiry":
        topic = get_topic_from_citations(citations)

    scores = [p.get("score") for p in passages if p.get("score") is not None]
    return {
        "success": True,
        "response": response["output"]["text"],
        "sessionId": response.get("sessionId", ""),
        "citations": citations,
        "citations_count": len(citations),
        "passages": passages,
        "passages_count": len(passages),
        "retrieval_scores": scores,
        "top_score": scores[0] if scores else None,
        "topic": topic,
        "guardrail": response.get("guardrailAction") == "INTERVENED",
        "sources_html": passages_to_sources_html(passages),
        "retrieve_error": str(retrieve_error) if retrieve_error else None,
    }

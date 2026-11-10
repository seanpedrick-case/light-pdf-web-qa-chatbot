"""
Amazon Bedrock AgentCore Harness backend for the Gradio app.

When USE_AGENTCORE_HARNESS=1 (and USE_BEDROCK_KB is not taking precedence),
questions are answered via InvokeHarness — the configured harness runs its
tools (e.g. Gateway targets, KB) and returns the final assistant text.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from tools.config import (
    AGENTCORE_HARNESS_ARN,
    AGENTCORE_HARNESS_QUALIFIER,
    AGENTCORE_HARNESS_RUNTIME_USER_ID,
    AWS_DEFAULT_REGION,
    AWS_REGION,
    USE_AGENTCORE_HARNESS,
)

REGION = AWS_DEFAULT_REGION or AWS_REGION or "eu-west-2"

_harness_client: Any = None


def harness_mode_enabled() -> bool:
    return USE_AGENTCORE_HARNESS == "1"


def _client():
    global _harness_client
    if _harness_client is None:
        _harness_client = boto3.client("bedrock-agentcore", region_name=REGION)
    return _harness_client


def _new_runtime_session_id() -> str:
    # InvokeHarness: 33–100 chars; pattern [a-zA-Z0-9][a-zA-Z0-9-_]*
    return f"gradio-{uuid.uuid4().hex}-{uuid.uuid4().hex[:12]}"


def ensure_runtime_session_id(existing: Optional[str]) -> str:
    sid = (existing or "").strip()
    if len(sid) >= 33:
        return sid
    return _new_runtime_session_id()


def _event_top_level(event: dict[str, Any]) -> tuple[str, Any]:
    if not isinstance(event, dict):
        return "", event
    if len(event) == 1:
        key = next(iter(event))
        return key, event[key]
    return "", event


def _consume_harness_stream(stream: Any) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    stop_reason: Optional[str] = None
    stream_error: Optional[str] = None

    for raw in stream:
        kind, payload = _event_top_level(raw)
        if kind == "contentBlockDelta":
            delta = (payload or {}).get("delta") or {}
            chunk = delta.get("text")
            if chunk:
                text_parts.append(chunk)
        elif kind == "contentBlockStart":
            start = (payload or {}).get("start") or {}
            tu = start.get("toolUse")
            if tu:
                tool_uses.append(
                    {
                        "name": tu.get("name"),
                        "toolUseId": tu.get("toolUseId"),
                        "type": tu.get("type"),
                        "serverName": tu.get("serverName"),
                    }
                )
        elif kind == "messageStop":
            stop_reason = (payload or {}).get("stopReason")
        elif kind == "metadata":
            usage = ((payload or {}).get("usage") or {}) | usage
        elif kind in (
            "internalServerException",
            "validationException",
            "runtimeClientError",
            "serviceQuotaExceededException",
            "throttlingException",
        ):
            stream_error = (payload or {}).get("message") or kind

    return {
        "response": "".join(text_parts).strip(),
        "tool_uses": tool_uses,
        "usage": usage,
        "stop_reason": stop_reason,
        "stream_error": stream_error,
    }


def format_harness_details_markdown(result: Optional[dict[str, Any]]) -> str:
    if not result:
        return "No harness invocation details."
    lines = ["### AgentCore harness", ""]
    if result.get("invoke_error"):
        lines.append(f"**Invoke error:** {result['invoke_error']}")
        lines.append("")
    if result.get("stream_error"):
        lines.append(f"**Stream error:** {result['stream_error']}")
        lines.append("")
    stop = result.get("stop_reason")
    if stop:
        lines.append(f"**Stop reason:** `{stop}`")
    usage = result.get("usage") or {}
    if usage:
        lines.append(
            f"**Tokens:** in={usage.get('inputTokens', '—')}, "
            f"out={usage.get('outputTokens', '—')}, "
            f"total={usage.get('totalTokens', '—')}"
        )
    tools = result.get("tool_uses") or []
    if tools:
        lines.append("")
        lines.append("**Tools invoked:**")
        for t in tools:
            name = t.get("name") or "(unknown)"
            server = t.get("serverName")
            suffix = f" ({server})" if server else ""
            lines.append(f"- `{name}`{suffix}")
    lines.append("")
    return "\n".join(lines)


def format_harness_sources_html(result: dict[str, Any]) -> str:
    tools = result.get("tool_uses") or []
    if not tools:
        return (
            "<p><em>AgentCore harness — no tool-use events in the response stream. "
            "Answers may still have used configured harness tools.</em></p>"
        )
    items = "".join(
        "<li><strong>{}</strong>{}</li>".format(
            t.get("name") or "tool",
            f" — {t['serverName']}" if t.get("serverName") else "",
        )
        for t in tools
    )
    return f"<p>Harness tools observed in this turn:</p><ul>{items}</ul>"


def get_harness_response(
    user_message: str,
    *,
    session_id: Optional[str] = None,
    runtime_user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Invoke AgentCore harness; return shape aligned with bedrock_kb.get_rag_response."""
    if not harness_mode_enabled():
        raise RuntimeError("USE_AGENTCORE_HARNESS is not enabled")
    if not (AGENTCORE_HARNESS_ARN or "").strip():
        raise ValueError(
            "AGENTCORE_HARNESS_ARN is required when USE_AGENTCORE_HARNESS=1"
        )

    runtime_session_id = ensure_runtime_session_id(session_id)
    params: dict[str, Any] = {
        "harnessArn": AGENTCORE_HARNESS_ARN.strip(),
        "runtimeSessionId": runtime_session_id,
        "messages": [
            {"role": "user", "content": [{"text": user_message}]},
        ],
    }
    qualifier = (AGENTCORE_HARNESS_QUALIFIER or "").strip()
    if qualifier:
        params["qualifier"] = qualifier
    user_id = (runtime_user_id or AGENTCORE_HARNESS_RUNTIME_USER_ID or "").strip()
    if user_id:
        params["runtimeUserId"] = user_id

    try:
        response = _client().invoke_harness(**params)
    except ClientError as exc:
        msg = exc.response.get("Error", {}).get("Message", str(exc))
        return {
            "success": False,
            "response": (
                "The AgentCore harness request could not be completed. "
                f"AWS error: {msg}"
            ),
            "sessionId": runtime_session_id,
            "invoke_error": msg,
            "tool_uses": [],
            "usage": {},
            "stop_reason": None,
            "sources_html": format_harness_sources_html({}),
            "passages": [],
            "passages_count": 0,
            "topic": "General Enquiry",
        }

    stream = response.get("stream")
    if stream is None:
        return {
            "success": False,
            "response": "Harness returned no event stream.",
            "sessionId": runtime_session_id,
            "invoke_error": "missing stream",
            "tool_uses": [],
            "usage": {},
            "stop_reason": None,
            "sources_html": format_harness_sources_html({}),
            "passages": [],
            "passages_count": 0,
            "topic": "General Enquiry",
        }

    parsed = _consume_harness_stream(stream)
    success = not parsed.get("stream_error")
    answer = parsed.get("response") or ""
    if not answer and parsed.get("stream_error"):
        answer = f"Harness error: {parsed['stream_error']}"

    result: dict[str, Any] = {
        "success": success and bool(answer),
        "response": answer or "No response text from harness.",
        "sessionId": runtime_session_id,
        "invoke_error": parsed.get("stream_error"),
        "tool_uses": parsed.get("tool_uses") or [],
        "usage": parsed.get("usage") or {},
        "stop_reason": parsed.get("stop_reason"),
        "sources_html": format_harness_sources_html(parsed),
        "passages": [],
        "passages_count": 0,
        "topic": "General Enquiry",
    }
    return result

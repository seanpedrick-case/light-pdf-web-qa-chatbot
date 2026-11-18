"""Multi-turn chat memory helpers (Gradio messages and legacy tuple history)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from tools.config import CHAT_MEMORY_ENABLED


def chat_memory_enabled() -> bool:
    return CHAT_MEMORY_ENABLED in {"1", "true", "True", "yes", "on", "YES", "ON"}


def memory_turn_limit() -> int:
    """Max user/assistant pairs to retain; effectively unlimited until Clear chat when enabled."""
    return 10**9 if chat_memory_enabled() else 0


def message_text(turn) -> str:
    """Extract plain text from a Gradio chat message or legacy tuple."""
    if isinstance(turn, (list, tuple)):
        if len(turn) >= 1:
            return str(turn[0]).strip()
        return ""
    if not isinstance(turn, dict):
        return str(turn).strip()
    content = turn.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part)))
            else:
                parts.append(str(part))
        return " ".join(parts).strip()
    return str(content).strip()


def chat_history_turn_pairs(chat_history) -> List[Tuple[str, str]]:
    """Pair complete user/assistant turns from Gradio messages or legacy (user, bot) tuples."""
    if not chat_history:
        return []

    first = chat_history[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        pairs: List[Tuple[str, str]] = []
        for item in chat_history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
        return pairs

    pairs: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    for turn in chat_history:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("role") or "").lower()
        text = message_text(turn)
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, text))
            pending_user = None
    return pairs


def llm_conversation_history_from_pairs(
    pairs: List[Tuple[str, str]],
) -> List[dict]:
    """Format completed turn pairs for Gemini / legacy send_request history."""
    conv: List[dict] = []
    for user_text, assistant_text in pairs:
        conv.append({"role": "user", "parts": [user_text]})
        conv.append({"role": "assistant", "parts": [assistant_text]})
    return conv


def summarize_chat_history(
    chat_history, max_memory_length: int
) -> Tuple[str, str, str, int]:
    """Return conversation text and first Q/A for topic extraction; empty when memory is off."""
    if not chat_memory_enabled():
        return "", "", "", max_memory_length

    pairs = chat_history_turn_pairs(chat_history)
    if not pairs:
        return "", "", "", max_memory_length

    if max_memory_length > 0 and len(pairs) > max_memory_length:
        pairs = pairs[-max_memory_length:]

    first_q, first_ans = pairs[0]

    conversation = ""
    for human_s, ai_s in pairs:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        conversation += "\n" + "\n".join([human, ai])

    return conversation.strip(), first_q, first_ans, max_memory_length

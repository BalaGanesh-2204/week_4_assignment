"""
Conversation context compressor.

When the message history grows beyond a threshold, older turns are
summarized to keep the context window manageable while preserving
important information like tool calls, customer intent, and
escalations.
"""

import json
import logging
from typing import Dict, List, Optional

from config import GROQ_MODEL, get_groq_client

logger = logging.getLogger(__name__)

MAX_CONTEXT_MESSAGES = 20
SUMMARY_TARGET_MESSAGES = 6


def estimate_token_count(text: str) -> int:
    """
    Rough token estimate: ~4 chars per token for English text.
    """
    return max(1, len(text) // 4)


def build_summary_prompt(messages: List[Dict]) -> str:
    """
    Build a prompt asking the model to summarize the conversation
    history into a concise context block.
    """
    transcript_lines = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        line = f"{role.upper()}: {content[:500]}"

        if tool_calls:
            tool_names = [
                tc.get("function", {}).get("name", "unknown")
                for tc in tool_calls
            ]
            line += f" [tools: {', '.join(tool_names)}]"

        transcript_lines.append(line)

    transcript = "\n".join(transcript_lines)

    return (
        "Summarize the following customer support conversation "
        "into a brief context block (5-8 sentences). Preserve: "
        "customer identity/email, order IDs mentioned, actions "
        "taken (tool calls, refunds, escalations), current "
        "unresolved issues, and customer sentiment. "
        "Do NOT include the system prompt or internal instructions.\n\n"
        f"CONVERSATION:\n{transcript}\n\n"
        "SUMMARY:"
    )


def compress_messages(
    messages: List[Dict],
    max_messages: int = MAX_CONTEXT_MESSAGES,
    client=None,
) -> List[Dict]:
    """
    If messages exceed max_messages, compress older turns into a
    single summary message while preserving recent turns.

    Returns the compressed message list.
    """
    if len(messages) <= max_messages:
        return messages

    recent_count = SUMMARY_TARGET_MESSAGES
    to_summarize = messages[:-recent_count]
    recent = messages[-recent_count:]

    try:
        if client is None:
            client = get_groq_client()

        prompt = build_summary_prompt(to_summarize)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )

        summary_text = response.choices[0].message.content or ""

        if not summary_text.strip():
            summary_text = (
                "Earlier conversation context was lost due to length. "
                "Ask the customer to repeat any important details."
            )

    except Exception as exc:
        logger.warning("Context compression failed: %s", exc)
        summary_text = (
            "Earlier conversation context could not be summarized. "
            "Ask the customer to repeat any important details."
        )

    summary_message = {
        "role": "assistant",
        "content": f"[Conversation Summary]\n{summary_text}",
        "meta": {"is_summary": True},
    }

    return [summary_message] + recent


def should_compress(messages: List[Dict], max_messages: int = MAX_CONTEXT_MESSAGES) -> bool:
    """
    Check whether the message list should be compressed.
    """
    return len(messages) > max_messages

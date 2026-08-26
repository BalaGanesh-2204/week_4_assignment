"""
Structured output schemas for the agent.

JSON schemas are plain dicts so they can be passed directly to
Groq's response_format; Pydantic models validate the model output.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------
# FINAL ANSWER SCHEMA
# ---------------------------------------------------------

FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "The complete customer-facing reply text."
            ),
        },
        "intent": {
            "type": "string",
            "enum": [
                "order_status",
                "refund_return",
                "shipping",
                "product_question",
                "account",
                "promotion",
                "escalation",
                "other",
            ],
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "heading": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["source"],
            },
        },
        "tools_used": {
            "type": "array",
            "items": {"type": "string"},
        },
        "escalated": {"type": "boolean"},
        "needs_human": {"type": "boolean"},
        "customer_sentiment": {
            "type": "string",
            "enum": ["calm", "frustrated", "angry"],
        },
    },
    "required": [
        "answer",
        "intent",
        "confidence",
        "citations",
        "tools_used",
        "escalated",
        "needs_human",
        "customer_sentiment",
    ],
}


class Citation(BaseModel):
    source: str
    heading: str = ""
    path: str = ""


class FinalAnswer(BaseModel):
    answer: str
    intent: str = Field(
        default="other",
        description="Classified intent of the turn.",
    )
    confidence: float = Field(default=0.5, ge=0, le=1)
    citations: List[Citation] = []
    tools_used: List[str] = []
    escalated: bool = False
    needs_human: bool = False
    customer_sentiment: str = "calm"

    @field_validator("intent", "customer_sentiment")
    @classmethod
    def _coerce_enum(cls, value: str) -> str:
        return value.strip().lower()


# ---------------------------------------------------------
# TOPICAL GUARDRAIL SCHEMA
# ---------------------------------------------------------

TOPIC_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "on_topic": {
            "type": "boolean",
            "description": (
                "True only for e-commerce customer-support topics."
            ),
        },
        "reason": {"type": "string"},
    },
    "required": ["on_topic", "reason"],
}


class TopicDecision(BaseModel):
    on_topic: bool
    reason: str = ""
    degraded: bool = False


def schema_to_response_format(
    name: str,
    schema: dict,
):
    """
    Build the Groq response_format payload.
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
        },
    }

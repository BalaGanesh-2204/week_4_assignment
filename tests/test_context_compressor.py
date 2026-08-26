from unittest.mock import MagicMock

import pytest

from context_compressor import (
    compress_messages,
    should_compress,
    estimate_token_count,
    build_summary_prompt,
)


def _make_messages(n):
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"Message {i}"})
        msgs.append({"role": "assistant", "content": f"Reply {i}"})
    return msgs


class TestEstimateTokenCount:
    def test_empty(self):
        assert estimate_token_count("") == 1

    def test_short(self):
        assert estimate_token_count("hello") == 1

    def test_longer(self):
        assert estimate_token_count("a" * 200) == 50


class TestShouldCompress:
    def test_below_threshold(self):
        assert should_compress(_make_messages(5), max_messages=20) is False

    def test_above_threshold(self):
        assert should_compress(_make_messages(15), max_messages=20) is True

    def test_exact_threshold(self):
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        assert should_compress(msgs, max_messages=10) is False


class TestBuildSummaryPrompt:
    def test_contains_transcript(self):
        msgs = [
            {"role": "user", "content": "I need help with ORD-1041"},
            {"role": "assistant", "content": "Sure, let me check."},
        ]
        prompt = build_summary_prompt(msgs)
        assert "ORD-1041" in prompt
        assert "USER:" in prompt
        assert "ASSISTANT:" in prompt

    def test_includes_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_order_status"}}
                ],
            }
        ]
        prompt = build_summary_prompt(msgs)
        assert "get_order_status" in prompt


class TestCompressMessages:
    def test_no_compression_needed(self):
        msgs = _make_messages(5)
        result = compress_messages(msgs, max_messages=20)
        assert result == msgs

    def test_compression_reduces_messages(self):
        msgs = _make_messages(15)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of conversation."
        mock_client.chat.completions.create.return_value = mock_response

        result = compress_messages(msgs, max_messages=20, client=mock_client)
        assert len(result) < len(msgs)
        assert result[0]["meta"]["is_summary"] is True

    def test_preserves_recent_messages(self):
        msgs = _make_messages(15)
        recent = msgs[-6:]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary."
        mock_client.chat.completions.create.return_value = mock_response

        result = compress_messages(msgs, max_messages=20, client=mock_client)
        for msg in recent:
            assert msg in result

    def test_fallback_on_api_error(self):
        msgs = _make_messages(15)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        result = compress_messages(msgs, max_messages=20, client=mock_client)
        assert result[0]["meta"]["is_summary"] is True
        assert "could not be summarized" in result[0]["content"]

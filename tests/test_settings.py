import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from settings import Settings, get_settings


class TestSettings:
    def test_defaults_from_env(self):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "test",
            "GEMINI_API_KEY": "test",
            "PINECONE_API_KEY": "test",
        }, clear=False):
            s = Settings()
            assert s.groq_model == "openai/gpt-oss-120b"
            assert s.chunk_size == 1000
            assert s.max_steps == 5
            assert s.store_name == "ShopKart"

    def test_custom_env_values(self):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "k1",
            "GEMINI_API_KEY": "k2",
            "PINECONE_API_KEY": "k3",
            "GROQ_MODEL": "custom-model",
            "CHUNK_SIZE": "500",
            "MAX_STEPS": "3",
        }, clear=False):
            s = Settings()
            assert s.groq_model == "custom-model"
            assert s.chunk_size == 500
            assert s.max_steps == 3

    def test_chunk_overlap_validation(self):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "k",
            "GEMINI_API_KEY": "k",
            "PINECONE_API_KEY": "k",
            "CHUNK_SIZE": "100",
            "CHUNK_OVERLAP": "100",
        }, clear=False):
            with pytest.raises(ValidationError):
                Settings()

    def test_embedding_dimension_bounds(self):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "k",
            "GEMINI_API_KEY": "k",
            "PINECONE_API_KEY": "k",
            "EMBEDDING_DIMENSION": "50",
        }, clear=False):
            with pytest.raises(ValidationError):
                Settings()

    def test_max_steps_bounds(self):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "k",
            "GEMINI_API_KEY": "k",
            "PINECONE_API_KEY": "k",
            "MAX_STEPS": "0",
        }, clear=False):
            with pytest.raises(ValidationError):
                Settings()

    def test_negative_threshold_rejected(self):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "k",
            "GEMINI_API_KEY": "k",
            "PINECONE_API_KEY": "k",
            "ESCALATE_REFUND_THRESHOLD": "-10",
        }, clear=False):
            with pytest.raises(ValidationError):
                Settings()

    def test_get_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)
        assert s.groq_model

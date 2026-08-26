"""
Type-safe configuration validation using Pydantic.

Validates all environment variables at startup with clear error
messages for misconfigurations.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


class Settings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    groq_api_key: str = Field(default_factory=lambda: _env("GROQ_API_KEY"))
    gemini_api_key: str = Field(default_factory=lambda: _env("GEMINI_API_KEY"))
    pinecone_api_key: str = Field(default_factory=lambda: _env("PINECONE_API_KEY"))

    groq_model: str = Field(
        default_factory=lambda: _env("GROQ_MODEL", "openai/gpt-oss-120b"),
        alias="GROQ_MODEL",
    )
    embedding_model: str = Field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "gemini-embedding-001"),
        alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(
        default_factory=lambda: _env_int("EMBEDDING_DIMENSION", 768),
        alias="EMBEDDING_DIMENSION",
    )

    pinecone_index_name: str = Field(
        default_factory=lambda: _env("PINECONE_INDEX_NAME", "ecommerce-support-index"),
        alias="PINECONE_INDEX_NAME",
    )
    pinecone_namespace: str = Field(
        default_factory=lambda: _env("PINECONE_NAMESPACE", "support-docs"),
        alias="PINECONE_NAMESPACE",
    )
    pinecone_cloud: str = Field(
        default_factory=lambda: _env("PINECONE_CLOUD", "aws"),
        alias="PINECONE_CLOUD",
    )
    pinecone_region: str = Field(
        default_factory=lambda: _env("PINECONE_REGION", "us-east-1"),
        alias="PINECONE_REGION",
    )

    chunk_size: int = Field(
        default_factory=lambda: _env_int("CHUNK_SIZE", 1000),
        alias="CHUNK_SIZE",
    )
    chunk_overlap: int = Field(
        default_factory=lambda: _env_int("CHUNK_OVERLAP", 150),
        alias="CHUNK_OVERLAP",
    )

    vector_top_k: int = Field(
        default_factory=lambda: _env_int("VECTOR_TOP_K", 20),
        alias="VECTOR_TOP_K",
    )
    keyword_top_k: int = Field(
        default_factory=lambda: _env_int("KEYWORD_TOP_K", 20),
        alias="KEYWORD_TOP_K",
    )
    final_top_k: int = Field(
        default_factory=lambda: _env_int("FINAL_TOP_K", 5),
        alias="FINAL_TOP_K",
    )
    rrf_k: int = Field(
        default_factory=lambda: _env_int("RRF_K", 60),
        alias="RRF_K",
    )

    max_steps: int = Field(
        default_factory=lambda: _env_int("MAX_STEPS", 5),
        alias="MAX_STEPS",
    )

    store_name: str = Field(
        default_factory=lambda: _env("STORE_NAME", "ShopKart"),
        alias="STORE_NAME",
    )
    support_lead: str = Field(
        default_factory=lambda: _env("SUPPORT_LEAD", "Balaganesh"),
        alias="SUPPORT_LEAD",
    )
    return_window_days: int = Field(
        default_factory=lambda: _env_int("RETURN_WINDOW_DAYS", 30),
        alias="RETURN_WINDOW_DAYS",
    )
    escalate_refund_threshold: float = Field(
        default_factory=lambda: _env_float("ESCALATE_REFUND_THRESHOLD", 200.0),
        alias="ESCALATE_REFUND_THRESHOLD",
    )

    block_on_injection: bool = Field(
        default_factory=lambda: _env_bool("BLOCK_ON_INJECTION", True),
        alias="BLOCK_ON_INJECTION",
    )

    retry_max_attempts: int = Field(
        default_factory=lambda: _env_int("RETRY_MAX_ATTEMPTS", 3),
        alias="RETRY_MAX_ATTEMPTS",
    )
    retry_base_delay: float = Field(
        default_factory=lambda: _env_float("RETRY_BASE_DELAY", 1.0),
        alias="RETRY_BASE_DELAY",
    )
    retry_backoff_factor: float = Field(
        default_factory=lambda: _env_float("RETRY_BACKOFF_FACTOR", 2.0),
        alias="RETRY_BACKOFF_FACTOR",
    )

    cache_ttl_seconds: int = Field(
        default_factory=lambda: _env_int("CACHE_TTL_SECONDS", 300),
        alias="CACHE_TTL_SECONDS",
    )
    cache_max_size: int = Field(
        default_factory=lambda: _env_int("CACHE_MAX_SIZE", 128),
        alias="CACHE_MAX_SIZE",
    )

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        if self.embedding_dimension < 128 or self.embedding_dimension > 4096:
            raise ValueError(
                f"EMBEDDING_DIMENSION ({self.embedding_dimension}) must be 128-4096"
            )
        if self.max_steps < 1 or self.max_steps > 20:
            raise ValueError(f"MAX_STEPS ({self.max_steps}) must be 1-20")
        if self.escalate_refund_threshold <= 0:
            raise ValueError(
                f"ESCALATE_REFUND_THRESHOLD ({self.escalate_refund_threshold}) must be > 0"
            )
        if self.return_window_days < 1 or self.return_window_days > 365:
            raise ValueError(
                f"RETURN_WINDOW_DAYS ({self.return_window_days}) must be 1-365"
            )
        if self.final_top_k > max(self.vector_top_k, self.keyword_top_k):
            raise ValueError(
                f"FINAL_TOP_K ({self.final_top_k}) should not exceed "
                f"VECTOR_TOP_K or KEYWORD_TOP_K"
            )
        return self


def get_settings() -> Settings:
    """
    Get validated settings instance. Raises ValidationError on misconfiguration.
    """
    return Settings()

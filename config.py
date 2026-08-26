import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"

MOCK_DATA_DIR = BASE_DIR / "mock_data"

DATA_DIR = BASE_DIR / "data"

LOGS_DIR = BASE_DIR / "logs"


def ensure_dirs():
    """
    Create runtime directories if they are missing.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


ensure_dirs()


# ---------------------------------------------------------
# API KEYS
# ---------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", "768")
)


# ---------------------------------------------------------
# PINECONE CONFIGURATION
# ---------------------------------------------------------

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "ecommerce-support-index"
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "support-docs"
)

PINECONE_CLOUD = os.getenv(
    "PINECONE_CLOUD",
    "aws"
)

PINECONE_REGION = os.getenv(
    "PINECONE_REGION",
    "us-east-1"
)


# ---------------------------------------------------------
# CHUNKING CONFIGURATION
# ---------------------------------------------------------

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "150")
)


# ---------------------------------------------------------
# HYBRID RETRIEVAL CONFIGURATION
# ---------------------------------------------------------

VECTOR_TOP_K = int(
    os.getenv("VECTOR_TOP_K", "20")
)

KEYWORD_TOP_K = int(
    os.getenv("KEYWORD_TOP_K", "20")
)

FINAL_TOP_K = int(
    os.getenv("FINAL_TOP_K", "5")
)

RRF_K = int(
    os.getenv("RRF_K", "60")
)


# ---------------------------------------------------------
# AGENT HARNESS CONFIGURATION
# ---------------------------------------------------------

# Hard cap on model steps (tool-call rounds) per turn.
MAX_STEPS = int(
    os.getenv("MAX_STEPS", "5")
)


# ---------------------------------------------------------
# BUSINESS RULES
# ---------------------------------------------------------

STORE_NAME = os.getenv(
    "STORE_NAME",
    "ShopKart"
)

SUPPORT_LEAD = os.getenv(
    "SUPPORT_LEAD",
    "Balaganesh"
)

RETURN_WINDOW_DAYS = int(
    os.getenv("RETURN_WINDOW_DAYS", "30")
)

ESCALATE_REFUND_THRESHOLD = float(
    os.getenv("ESCALATE_REFUND_THRESHOLD", "200.0")
)


# ---------------------------------------------------------
# GUARDRAIL CONFIGURATION
# ---------------------------------------------------------

BLOCK_ON_INJECTION = os.getenv(
    "BLOCK_ON_INJECTION",
    "true"
).lower() in ("true", "1", "yes")


RESTOCKING_FEE_RATE = 0.10


# ---------------------------------------------------------
# RETRY CONFIGURATION
# ---------------------------------------------------------

RETRY_MAX_ATTEMPTS = int(
    os.getenv("RETRY_MAX_ATTEMPTS", "3")
)

RETRY_BASE_DELAY = float(
    os.getenv("RETRY_BASE_DELAY", "1.0")
)

RETRY_BACKOFF_FACTOR = float(
    os.getenv("RETRY_BACKOFF_FACTOR", "2.0")
)


# ---------------------------------------------------------
# CACHE CONFIGURATION
# ---------------------------------------------------------

CACHE_TTL_SECONDS = int(
    os.getenv("CACHE_TTL_SECONDS", "300")
)

CACHE_MAX_SIZE = int(
    os.getenv("CACHE_MAX_SIZE", "128")
)


# ---------------------------------------------------------
# VALIDATE REQUIRED KEYS
# ---------------------------------------------------------

def validate_config():
    """
    Validate that all required API keys are available.
    """

    missing = []

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------
# CLIENTS (with retry)
# ---------------------------------------------------------

def get_groq_client():
    """
    Return Groq client.
    """

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing.")

    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError(
            "The 'groq' package is required. "
            "Install it with 'pip install groq'."
        ) from exc

    return Groq(api_key=GROQ_API_KEY)


def get_gemini_client():
    """
    Return Gemini client.
    """

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing.")

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "The 'google-genai' package is required. "
            "Install it with 'pip install google-genai'."
        ) from exc

    return genai.Client(api_key=GEMINI_API_KEY)

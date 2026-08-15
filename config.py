import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env for local development
load_dotenv()


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"


# ---------------------------------------------------------
# API KEYS
# ---------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

# Groq LLM
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

# Gemini embedding model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)

# Gemini embedding dimensions
# 768 is sufficient for this project and reduces vector size.
EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", "768")
)


# ---------------------------------------------------------
# PINECONE CONFIGURATION
# ---------------------------------------------------------

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "rag-docs-index"
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "documents"
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
# RETRIEVAL CONFIGURATION
# ---------------------------------------------------------

TOP_K = int(
    os.getenv("TOP_K", "5")
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
# CLIENTS
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
            "The 'groq' package is required to create a Groq client. "
            "Install it with 'pip install groq' or add it to your environment."
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
            "The 'google-genai' package is required to create a Gemini client. "
            "Install it with 'pip install google-genai' or add it to your environment."
        ) from exc

    return genai.Client(api_key=GEMINI_API_KEY)
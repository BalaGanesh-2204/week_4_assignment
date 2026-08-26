from typing import List

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    get_gemini_client,
)


EMBED_BATCH_SIZE = 100


def _embed_raw(texts: List[str]) -> List[List[float]]:
    """
    Call Gemini embeddings for a single batch of texts.
    """

    client = get_gemini_client()

    from google.genai import types

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        ),
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a single piece of text.
    """

    return _embed_raw([text])[0]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for many texts, batching requests.
    """

    if not texts:
        return []

    vectors: List[List[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):

        batch = texts[start:start + EMBED_BATCH_SIZE]

        vectors.extend(_embed_raw(batch))

    return vectors


def embed_query(text: str) -> List[float]:
    """
    Embed a search query at retrieval time.
    """

    return get_embedding(text)


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed document chunks at ingestion time.
    """

    return get_embeddings(texts)

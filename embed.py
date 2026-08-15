from typing import List

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    get_gemini_client,
)


def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a single piece of text.
    """

    client = get_gemini_client()

    from google.genai import types

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        ),
    )

    return response.embeddings[0].values


def get_embeddings(
    texts: List[str]
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.

    Gemini accepts multiple contents in one request.
    """

    if not texts:
        return []

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


def embed_query(text: str) -> List[float]:
    """
    Alias for get_embedding to preserve a consistent public API.
    """

    return get_embedding(text)
 
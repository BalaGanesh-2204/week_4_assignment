from typing import List, Dict

from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
)


def get_pinecone_client():
    """
    Create Pinecone client.
    """

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing.")

    return Pinecone(api_key=PINECONE_API_KEY)


def index_exists() -> bool:
    """
    Check whether the support knowledge index exists.
    """

    pc = get_pinecone_client()

    indexes = pc.list_indexes()

    existing_names = [index["name"] for index in indexes]

    return PINECONE_INDEX_NAME in existing_names


def create_index_if_not_exists():
    """
    Create the serverless index if it doesn't exist.
    """

    pc = get_pinecone_client()

    if index_exists():
        return

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )


def get_index():
    """
    Return the Pinecone index object.
    """

    pc = get_pinecone_client()

    create_index_if_not_exists()

    return pc.Index(PINECONE_INDEX_NAME)


def upsert_chunks(
    chunks: List[Dict],
    embeddings: List[List[float]],
) -> int:
    """
    Upsert embedded chunks into Pinecone.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    index = get_index()

    vectors = []

    for chunk, embedding in zip(chunks, embeddings):

        vectors.append(
            {
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "path": chunk["path"],
                    "chunk_index": chunk["chunk_index"],
                    "heading": chunk.get("heading", ""),
                },
            }
        )

    batch_size = 100
    total_upserted = 0

    for start in range(0, len(vectors), batch_size):

        batch = vectors[start:start + batch_size]

        response = index.upsert(
            vectors=batch,
            namespace=PINECONE_NAMESPACE,
        )

        total_upserted += response.upserted_count

    return total_upserted


def dense_search(
    query_embedding: List[float],
    top_k: int,
) -> List[Dict]:
    """
    Dense vector search. Returns normalized hits with metadata.
    """

    index = get_index()

    results = index.query(
        namespace=PINECONE_NAMESPACE,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    hits = []

    for match in results.matches:

        hits.append(
            {
                "id": match.id,
                "score": float(match.score),
                "metadata": dict(match.metadata or {}),
            }
        )

    return hits


def delete_namespace():
    """
    Delete all vectors from the namespace (full re-index).
    """

    index = get_index()

    index.delete(
        delete_all=True,
        namespace=PINECONE_NAMESPACE,
    )


def get_index_stats():
    """
    Return Pinecone index statistics.
    """

    index = get_index()

    return index.describe_index_stats()

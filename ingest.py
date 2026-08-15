from chunk import create_chunks
from embed import get_embeddings
from store import (
    create_index_if_not_exists,
    upsert_chunks,
)


def ingest_documents():
    """
    Complete document ingestion pipeline.
    """

    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    # -----------------------------------------------------
    # Step 1: Create Pinecone index
    # -----------------------------------------------------

    print("\n[1/4] Checking Pinecone index...")

    create_index_if_not_exists()

    print("Pinecone index ready.")

    # -----------------------------------------------------
    # Step 2: Load and chunk documents
    # -----------------------------------------------------

    print("\n[2/4] Loading and chunking documents...")

    chunks = create_chunks()

    print(
        f"Created {len(chunks)} chunks."
    )

    if not chunks:
        print("No chunks found.")
        return 0

    # -----------------------------------------------------
    # Step 3: Generate embeddings
    # -----------------------------------------------------

    print("\n[3/4] Generating Gemini embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = get_embeddings(texts)

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    # -----------------------------------------------------
    # Step 4: Store in Pinecone
    # -----------------------------------------------------

    print("\n[4/4] Uploading to Pinecone...")

    count = upsert_chunks(
        chunks,
        embeddings,
    )

    print(
        f"Successfully upserted {count} vectors."
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)

    return count


if __name__ == "__main__":
    ingest_documents()
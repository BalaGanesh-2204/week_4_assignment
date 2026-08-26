"""
Ingestion pipeline entry point.

Reads Markdown from docs/, chunks it, embeds it with Gemini and
stores vectors in a dedicated Pinecone index. Also builds the
local BM25 keyword index used for hybrid search.
"""

import sys

import chunker
import embedder
import vector_store
import keyword_search

from config import validate_config, PINECONE_INDEX_NAME


def run_ingest(full_rebuild: bool = False) -> dict:
    """
    Run the full ingestion pipeline.

    Returns a small stats dict for callers (CLI or Streamlit).
    """

    validate_config()

    print("Loading Markdown files...")
    documents = chunker.load_markdown_files()

    print(f"Found {len(documents)} document(s).")

    print("Creating knowledge chunks...")
    chunks = chunker.create_chunks()

    if not chunks:
        raise ValueError(
            "No chunks were produced. Check the contents of docs/."
        )

    print(f"Created {len(chunks)} chunk(s).")

    if full_rebuild:
        print("Full rebuild requested - clearing namespace...")
        try:
            vector_store.delete_namespace()
        except Exception as exc:
            print(f"Namespace clear skipped: {exc}")

    print("Generating embeddings with Gemini...")
    embeddings = embedder.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    print(f"Upserting into Pinecone index '{PINECONE_INDEX_NAME}'...")
    upserted = vector_store.upsert_chunks(chunks, embeddings)

    print("Writing local chunk store + BM25 keyword index...")
    keyword_search.write_chunk_store(chunks)
    keyword_search.build_keyword_index(chunks)

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "upserted": upserted,
        "index": PINECONE_INDEX_NAME,
    }

    print("\nIngestion complete:")
    print(f"  Documents : {stats['documents']}")
    print(f"  Chunks    : {stats['chunks']}")
    print(f"  Upserted  : {stats['upserted']}")

    return stats


def main():
    """
    CLI entry point.
    """

    try:
        run_ingest()
    except Exception as exc:
        print(f"\nIngestion failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

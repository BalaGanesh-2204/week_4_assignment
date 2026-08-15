from pathlib import Path
from typing import List, Dict
import hashlib

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_markdown_files() -> List[Dict]:
    """
    Load all Markdown files from the docs directory.

    Returns:
        List of dictionaries containing:
        - filename
        - path
        - content
    """

    documents = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Docs directory not found: {DOCS_DIR}"
        )

    markdown_files = list(DOCS_DIR.rglob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(
            f"No Markdown files found inside {DOCS_DIR}"
        )

    for file_path in markdown_files:

        content = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            continue

        documents.append(
            {
                "filename": file_path.name,
                "path": str(file_path),
                "content": content,
            }
        )

    return documents


def clean_text(text: str) -> str:
    """
    Basic Markdown text cleanup.
    """

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks.

    Uses character-based chunking for simplicity.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    text = clean_text(text)

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        # Try to avoid cutting in the middle of a word.
        if end < text_length:

            last_space = chunk.rfind(" ")

            if last_space > chunk_size * 0.5:
                chunk = chunk[:last_space]

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        actual_length = len(chunk)

        if actual_length == 0:
            break

        if end >= text_length:
            break

        step = actual_length - chunk_overlap
        if step <= 0:
            step = 1

        start += step

    return chunks


def create_chunk_id(
    filename: str,
    chunk_index: int,
    content: str
) -> str:
    """
    Create a deterministic ID for a chunk.
    """

    raw_id = (
        f"{filename}:"
        f"{chunk_index}:"
        f"{content}"
    )

    hash_value = hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()[:16]

    return f"{filename}-{chunk_index}-{hash_value}"


def create_chunks() -> List[Dict]:
    """
    Load all Markdown files and create chunks.

    Returns:
        List of chunk dictionaries.
    """

    documents = load_markdown_files()

    all_chunks = []

    for document in documents:

        chunks = split_text(
            document["content"]
        )

        for index, chunk in enumerate(chunks):

            chunk_id = create_chunk_id(
                document["filename"],
                index,
                chunk
            )

            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "source": document["filename"],
                    "path": document["path"],
                    "chunk_index": index,
                }
            )

    return all_chunks
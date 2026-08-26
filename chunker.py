import hashlib
import re
from typing import List, Dict

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*)$")


def load_markdown_files() -> List[Dict]:
    """
    Load all Markdown files from the docs directory.
    """

    documents = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Docs directory not found: {DOCS_DIR}"
        )

    markdown_files = sorted(DOCS_DIR.rglob("*.md"))

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


def parse_sections(markdown_text: str) -> List[Dict]:
    """
    Split a Markdown document into sections at headings.

    Returns a list of:
    - heading: breadcrumb of heading titles ("Page / Section / Sub")
    - body: the raw text belonging to that section
    """

    lines = markdown_text.splitlines()

    sections = []

    current_heading = ""
    current_lines = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body or current_heading:
            sections.append(
                {
                    "heading": current_heading.strip(),
                    "body": body,
                }
            )

    for line in lines:

        match = HEADING_PATTERN.match(line)

        if match:

            flush()

            level = len(match.group(1))
            title = match.group(2).strip()

            # Build a breadcrumb from the existing heading path.
            parent_parts = current_heading.split(" > ")
            base = parent_parts[0] if parent_parts else ""

            if level == 1:
                current_heading = title
            else:
                prefix = base if base else ""
                current_heading = (
                    f"{prefix} > {title}" if prefix else title
                )

            current_lines = [line]

        else:

            current_lines.append(line)

    flush()

    return sections


def split_long_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a long section into overlapping character chunks,
    avoiding cuts in the middle of words where possible.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    text_length = len(text)

    if text_length <= chunk_size:
        return [text]

    pieces = []
    start = 0

    while start < text_length:

        end = start + chunk_size
        piece = text[start:end]

        if end < text_length:
            last_space = piece.rfind(" ")
            if last_space > chunk_size * 0.5:
                piece = piece[:last_space]

        piece = piece.strip()

        if piece:
            pieces.append(piece)

        actual = len(piece)

        if actual == 0 or end >= text_length:
            break

        step = max(actual - chunk_overlap, 1)
        start += step

    return pieces


def create_chunk_id(filename: str, chunk_index: int) -> str:
    """
    Deterministic chunk id so re-ingesting updates in place.
    """

    hash_value = hashlib.sha256(
        filename.encode("utf-8")
    ).hexdigest()[:12]

    return f"{filename}-{chunk_index}-{hash_value}"


def create_chunks() -> List[Dict]:
    """
    Load all Markdown files and build knowledge chunks.

    Each chunk keeps its heading breadcrumb for better retrieval
    context and citation display.
    """

    documents = load_markdown_files()

    all_chunks = []

    for document in documents:

        sections = parse_sections(document["content"])

        doc_chunks = []

        for section in sections:

            # Prepend the heading breadcrumb so every chunk is self-contained.
            prefix = (
                f"[{section['heading']}]\n"
                if section["heading"]
                else ""
            )

            full_text = prefix + section["body"]

            if not full_text.strip():
                continue

            pieces = split_long_text(full_text)

            for piece in pieces:
                doc_chunks.append(
                    {
                        "text": piece,
                        "heading": section["heading"] or document["filename"],
                    }
                )

        for index, chunk in enumerate(doc_chunks):

            all_chunks.append(
                {
                    "id": create_chunk_id(document["filename"], index),
                    "text": chunk["text"],
                    "source": document["filename"],
                    "path": document["path"],
                    "chunk_index": index,
                    "heading": chunk["heading"],
                }
            )

    return all_chunks

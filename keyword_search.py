"""
Keyword (BM25) search over ingested chunks.

Uses a self-contained Okapi BM25 implementation (no external
dependency) over the local chunk store written at ingest time.
"""

import json
import math
import pickle
import re
from collections import Counter
from typing import Dict, List, Tuple

from config import DATA_DIR


CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

KEYWORD_INDEX_FILE = DATA_DIR / "keyword_index.pkl"


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    """
    Simple lowercase alphanumeric tokenizer.
    """

    return TOKEN_PATTERN.findall(text.lower())


class _BM25:
    """
    Minimal Okapi BM25 index.

    score(D,Q) = sum over query terms of
        idf(q) * f(q,D) * (K1 + 1) /
        (f(q,D) + K1 * (1 - B + B * len(D)/avgdl))
    with idf(q) = ln(1 + (N - df + 0.5) / (df + 0.5)).
    """

    def __init__(self, corpus_tokens: List[List[str]]):

        self.doc_count = len(corpus_tokens)

        self.doc_lens = [len(tokens) for tokens in corpus_tokens]

        total = sum(self.doc_lens)

        self.avgdl = (total / self.doc_count) if self.doc_count else 0.0

        self.doc_freqs: List[Counter] = [
            Counter(tokens) for tokens in corpus_tokens
        ]

        document_frequency: Counter = Counter()

        for freqs in self.doc_freqs:
            document_frequency.update(freqs.keys())

        self.idf: Dict[str, float] = {}

        for term, df in document_frequency.items():

            self.idf[term] = math.log(
                1.0
                + (self.doc_count - df + 0.5)
                / (df + 0.5)
            )

    def get_scores(self, query_tokens: List[str]) -> List[float]:

        scores = [0.0] * self.doc_count

        if not self.avgdl:
            return scores

        for term in set(query_tokens):

            idf = self.idf.get(term)

            if not idf:
                continue

            for index, freqs in enumerate(self.doc_freqs):

                frequency = freqs.get(term, 0)

                if not frequency:
                    continue

                denominator = (
                    frequency
                    + K1
                    * (
                        1.0
                        - B
                        + B * self.doc_lens[index] / self.avgdl
                    )
                )

                scores[index] += (
                    idf * frequency * (K1 + 1.0) / denominator
                )

        return scores


# ---------------------------------------------------------
# CHUNK STORE
# ---------------------------------------------------------

def write_chunk_store(chunks: List[Dict]):
    """
    Persist the local chunk store used by keyword search.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNKS_FILE.open("w", encoding="utf-8") as handle:

        for chunk in chunks:

            handle.write(
                json.dumps(chunk, ensure_ascii=False)
            )
            handle.write("\n")


def load_chunk_store() -> List[Dict]:
    """
    Load chunks from the local store.
    """

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Local chunk store not found. Run 'python ingest.py' first."
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as handle:

        for line in handle:

            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


# ---------------------------------------------------------
# INDEX BUILD / LOAD
# ---------------------------------------------------------

def build_keyword_index(chunks: List[Dict]):
    """
    Build and persist the BM25 keyword index.
    """

    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    payload = {
        "chunks": chunks,
        "bm25": _BM25(corpus),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with KEYWORD_INDEX_FILE.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_or_rebuild():
    """
    Load the pickled index; rebuild from the chunk store if missing.
    """

    if KEYWORD_INDEX_FILE.exists():

        with KEYWORD_INDEX_FILE.open("rb") as handle:
            return pickle.load(handle)

    chunks = load_chunk_store()

    build_keyword_index(chunks)

    return {"chunks": chunks}


def keyword_search(
    query: str,
    top_k: int,
) -> List[Tuple[str, float]]:
    """
    BM25 keyword search over ingested chunks.

    Returns list of (chunk_id, score), best first.
    """

    payload = _load_or_rebuild()

    bm25 = payload["bm25"]
    chunks = payload["chunks"]

    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        zip(chunks, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]

    return [
        (chunk["id"], float(score))
        for chunk, score in ranked
        if score > 0
    ]


def get_chunks_by_ids(ids: List[str]) -> Dict[str, Dict]:
    """
    Map of chunk id -> chunk record from the local store.
    """

    payload = _load_or_rebuild()

    wanted = set(ids)

    return {
        chunk["id"]: chunk
        for chunk in payload["chunks"]
        if chunk["id"] in wanted
    }

from typing import List, Dict

import embedder
import vector_store
import keyword_search

from config import (
    VECTOR_TOP_K,
    KEYWORD_TOP_K,
    FINAL_TOP_K,
    RRF_K,
)


def _reciprocal_rank_fusion(
    dense_hits: List[Dict],
    keyword_hits,
) -> Dict[str, float]:
    """
    Fuse ranked lists with Reciprocal Rank Fusion.
    """

    fused: Dict[str, float] = {}

    for rank, hit in enumerate(dense_hits):
        chunk_id = hit["id"]
        fused[chunk_id] = fused.get(chunk_id, 0.0)
        fused[chunk_id] += 1.0 / (RRF_K + rank + 1)

    for rank, (chunk_id, _kw_score) in enumerate(keyword_hits):
        fused[chunk_id] = fused.get(chunk_id, 0.0)
        fused[chunk_id] += 1.0 / (RRF_K + rank + 1)

    return fused


def hybrid_search(query: str) -> List[Dict]:
    """
    Hybrid retrieval: dense vector search + BM25 keyword search,
    fused with RRF.

    Returns top chunks with text, source and score breakdown.
    """

    # Dense leg.
    try:
        query_vector = embedder.embed_query(query)
        dense_hits = vector_store.dense_search(
            query_embedding=query_vector,
            top_k=VECTOR_TOP_K,
        )
    except Exception as exc:
        raise RuntimeError(
            "Dense search failed. Is the index ingested? "
            f"Run 'python ingest.py'. ({exc})"
        ) from exc

    # Keyword leg.
    keyword_hits = keyword_search.keyword_search(
        query=query,
        top_k=KEYWORD_TOP_K,
    )

    # Fusion.
    fused_scores = _reciprocal_rank_fusion(dense_hits, keyword_hits)

    dense_score_map = {hit["id"]: hit["score"] for hit in dense_hits}
    keyword_score_map = {
        chunk_id: score for chunk_id, score in keyword_hits
    }

    ordered_ids = [
        chunk_id
        for chunk_id, _score in sorted(
            fused_scores.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )[:FINAL_TOP_K]
    ]

    chunk_map = keyword_search.get_chunks_by_ids(ordered_ids)

    results = []

    for chunk_id in ordered_ids:

        chunk = chunk_map.get(chunk_id)

        if not chunk:
            continue

        results.append(
            {
                "id": chunk_id,
                "text": chunk["text"],
                "source": chunk["source"],
                "path": chunk.get("path", ""),
                "heading": chunk.get("heading", ""),
                "dense_score": round(dense_score_map.get(chunk_id, 0.0), 4),
                "keyword_score": round(keyword_score_map.get(chunk_id, 0.0), 4),
                "rrf_score": round(fused_scores[chunk_id], 6),
            }
        )

    return results

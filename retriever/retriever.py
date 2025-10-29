# retriever/retriever.py
from typing import List, Dict
from embeddings.indexer import FAISSIndexer
from config import TOP_K

class HybridRetriever:
    """
    Minimal hybrid retriever: uses FAISS dense retrieval and simple keyword filtering.
    """
    def __init__(self, indexer: FAISSIndexer):
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = TOP_K):
        # dense retrieval
        dense_hits = self.indexer.query(query, top_k=top_k)
        # simple re-ranking by exact keyword hits in text (boost)
        q_tokens = set(query.lower().split())
        for h in dense_hits:
            txt_tokens = set(h["text"].lower().split())
            overlap = q_tokens.intersection(txt_tokens)
            # boost score by overlap fraction
            boost = len(overlap) / (len(q_tokens) + 1e-6)
            h["score"] = h["score"] + 0.1 * boost
        dense_hits.sort(key=lambda x: x["score"], reverse=True)
        return dense_hits[:top_k]

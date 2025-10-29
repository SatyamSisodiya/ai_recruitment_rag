# embeddings/indexer.py
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
from typing import List, Dict
from config import EMBEDDING_MODEL, INDEX_PATH, METADATA_DB, CHUNK_SIZE, CHUNK_OVERLAP
from utils import ensure_dir

class FAISSIndexer:
    def __init__(self, model_name: str = EMBEDDING_MODEL, dim: int = None):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = pd.DataFrame(columns=["doc_id", "chunk_id", "text", "section", "type"])
        self.index_path = INDEX_PATH
        self.metadata_db = METADATA_DB
        ensure_dir(self.index_path)
        ensure_dir(self.metadata_db)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True), dtype=np.float32)

    def init_index(self, d: int):
        # L2 index
        self.index = faiss.IndexFlatIP(d)  # use inner product on normalized vectors
        # we will store vectors normalized; IP=cosine if vectors normalized
        # if large scale: replace with IndexIVFFlat etc.

    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        return vectors / norms

    def add_documents(self, docs: List[Dict]):
        """
        docs: list of dicts with keys: doc_id, text, section, type
        """
        texts = [d["text"] for d in docs]
        vectors = self.encode(texts)
        vectors = self.normalize_vectors(vectors)
        if self.index is None:
            self.init_index(vectors.shape[1])
        self.index.add(vectors)
        # append metadata
        start = len(self.metadata)
        rows = []
        for i, d in enumerate(docs):
            rows.append({
                "doc_id": d.get("doc_id"),
                "chunk_id": start + i,
                "text": d.get("text"),
                "section": d.get("section"),
                "type": d.get("type")
            })
        self.metadata = pd.concat([self.metadata, pd.DataFrame(rows)], ignore_index=True)

    def save(self):
        ensure_dir(self.index_path)
        faiss.write_index(self.index, self.index_path)
        self.metadata.to_parquet(self.metadata_db, index=False)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_db):
            self.index = faiss.read_index(self.index_path)
            self.metadata = pd.read_parquet(self.metadata_db)
        else:
            raise FileNotFoundError("Index or metadata db not found.")

    def query(self, query_text: str, top_k: int = 5):
        v = self.encode([query_text])
        v = self.normalize_vectors(v)
        D, I = self.index.search(v, top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            meta = self.metadata.iloc[idx].to_dict()
            meta["score"] = float(score)
            hits.append(meta)
        return hits

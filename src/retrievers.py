"""Day 7 Retrieval implementations: Dense, BM25 Lexical, and Hybrid RRF."""
import math
import re
import unicodedata
from collections import Counter
from typing import Any
import chromadb
from chromadb.utils import embedding_functions

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words with Unicode normalization."""
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())

def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Check if chunk matches any gold evidence in source_id, page, and contains evidence substring."""
    return any(
        chunk.get('source_id') == g.get('source_id')
        and chunk.get('page') == g.get('page')
        and g.get('evidence', '') in chunk.get('text', '')
        for g in gold
    )

class DenseRetriever:
    """Dense retriever querying ChromaDB with all-MiniLM-L6-v2."""
    def __init__(self, persist_dir: str = "starter/chroma_store", collection_name: str = "strategy_b"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_collection(collection_name, embedding_function=self.ef)
        self.strategy = "dense"

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        where_clause = None
        if filters:
            # Handle ChromaDB where format: if multiple keys, use $and
            if len(filters) == 1:
                where_clause = filters
            else:
                where_clause = {"$and": [{k: v} for k, v in filters.items()]}

        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            # If filter causes no matches or error, fallback to query without filter
            res = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        results = []
        if not res or not res["ids"] or not res["ids"][0]:
            return results

        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns L2/cosine distance; convert to similarity score
            sim = 1.0 / (1.0 + float(dist))
            results.append({
                "id": cid,
                "text": doc,
                "source_id": "vinuni-handbook-2024",
                "source": meta.get("source", "vinuni_handbook.pdf"),
                "page": int(meta.get("page")),
                "metadata": {k: v for k, v in meta.items() if k not in ("source", "page", "chunk_id")},
                "score": sim,
                "strategy": self.strategy
            })
        return results


class BM25Retriever:
    """Okapi BM25 Lexical retriever for Day 7."""
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.strategy = "bm25"
        
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / (len(self.doc_lens) or 1)
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        
        # Inverted index & IDF
        df = Counter()
        for freq in self.doc_freqs:
            for term in freq:
                df[term] += 1
                
        N = len(chunks)
        self.idf = {
            term: math.log((N - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
            
        scored = []
        for idx, (chunk, freq, doc_len) in enumerate(zip(self.chunks, self.doc_freqs, self.doc_lens)):
            # Check metadata filter
            if filters:
                match = True
                chunk_meta = chunk.get("metadata", {})
                for fk, fv in filters.items():
                    if chunk_meta.get(fk) != fv and chunk.get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue
                    
            score = 0.0
            for term in query_tokens:
                if term not in freq:
                    continue
                tf = freq[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)
                
            if score > 0:
                scored.append((score, chunk))
                
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        results = []
        for score, chunk in scored[:top_k]:
            results.append({
                **chunk,
                "score": score,
                "strategy": self.strategy
            })
        return results


class HybridRetriever:
    """Hybrid Retriever combining Dense and BM25 using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.strategy = "hybrid_rrf"

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidate_k = max(top_k * 4, 10)
        dense_results = self.dense.search(query, filters=filters, top_k=candidate_k)
        bm25_results = self.bm25.search(query, filters=filters, top_k=candidate_k)

        chunk_map = {}
        rrf_scores = Counter()

        for rank, r in enumerate(dense_results):
            cid = r["id"]
            chunk_map[cid] = r
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank + 1)

        for rank, r in enumerate(bm25_results):
            cid = r["id"]
            chunk_map[cid] = r
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        results = []
        for cid, score in ranked:
            c = chunk_map[cid]
            results.append({
                **c,
                "score": score,
                "strategy": self.strategy
            })
        return results

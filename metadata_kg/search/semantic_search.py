"""Semantic search engine over MetadataKnowledgeGraph (PHASE 6).

Features:
- index_kg(kg) → embed all nodes (sentence-transformers, optional)
- search(query, top_k) → cosine similarity ranking
- keyword_fallback(query) → BM25
- hybrid_search(query) → linear combine semantic + BM25
- expand_query(query) → synonyms + KG concept neighbors
- explain_result(entity_id, query) → match rationale
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph

# Optional heavy deps — lazy import
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore

_BM25_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None  # type: ignore


# ---------------------------------------------------------------------------
# Lightweight fallback embedding (when sentence-transformers not installed)
# ---------------------------------------------------------------------------
def _hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Cheap deterministic embedding via hashed bag-of-words (offline fallback)."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"\w+", text.lower()):
        h = hash(tok) % dim
        vec[h] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    entity_id: str
    score: float
    method: str           # semantic | keyword | hybrid
    snippet: str = ""
    matched_terms: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "score": round(self.score, 4),
            "method": self.method,
            "snippet": self.snippet,
            "matched_terms": self.matched_terms or [],
        }


# ---------------------------------------------------------------------------
# Synonym table for query expansion (small, illustrative)
# ---------------------------------------------------------------------------
SYNONYMS: dict[str, list[str]] = {
    "covid": ["coronavirus", "sars-cov-2", "pandemic"],
    "weather": ["climate", "meteorology", "temperature"],
    "fintech": ["finance", "banking", "payments", "digital banking"],
    "health": ["medical", "healthcare", "clinical"],
    "education": ["learning", "academic", "school"],
    "energy": ["electricity", "power", "renewable"],
    "transport": ["transportation", "mobility", "logistics"],
}


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------
class SemanticMetadataSearch:
    """Hybrid search over a MetadataKnowledgeGraph."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 256,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.entity_ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.documents: list[str] = []    # one text per entity (for BM25)
        self.bm25: Any | None = None      # BM25Okapi if available
        self._kg: MetadataKnowledgeGraph | None = None
        self._use_st = _ST_AVAILABLE
        self._st_model: Any | None = None
        if self._use_st:
            try:
                self._st_model = SentenceTransformer(model_name)
                logger.info(f"Loaded SentenceTransformer model={model_name}")
            except Exception as exc:
                logger.warning(f"SentenceTransformer load failed ({exc}); using hash embeddings.")
                self._use_st = False

    # ------------------------------------------------------------------
    def _entity_text(self, kg: MetadataKnowledgeGraph, entity_id: str) -> str:
        props = kg.get_entity(entity_id)
        parts: list[str] = []
        for k, v in props.items():
            if isinstance(v, (list, tuple, set)):
                parts.append(" ".join(str(x) for x in v))
            else:
                parts.append(str(v))
        return " ".join(parts).strip()

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._use_st and self._st_model is not None:
            return np.asarray(self._st_model.encode(texts, show_progress_bar=False))
        return np.stack([_hash_embed(t, self.embedding_dim) for t in texts]) if texts else np.zeros((0, self.embedding_dim))

    # ------------------------------------------------------------------
    def index_kg(self, kg: MetadataKnowledgeGraph) -> None:
        """Build semantic + BM25 index over all entities."""
        self._kg = kg
        self.entity_ids = kg.all_entities()
        self.documents = [self._entity_text(kg, eid) for eid in self.entity_ids]

        if self.entity_ids:
            self.embeddings = self._embed(self.documents)
        else:
            self.embeddings = np.zeros((0, self.embedding_dim))

        if _BM25_AVAILABLE and self.documents:
            tokens = [re.findall(r"\w+", d.lower()) for d in self.documents]
            self.bm25 = BM25Okapi(tokens) if any(tokens) else None
        else:
            self.bm25 = None

        logger.info(
            f"Indexed KG: {len(self.entity_ids)} entities, "
            f"embed_dim={self.embeddings.shape[1] if self.embeddings.size else 0}, "
            f"bm25={'on' if self.bm25 else 'off'}"
        )

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Semantic search via cosine similarity."""
        if not self.entity_ids or self.embeddings is None or self.embeddings.size == 0:
            return []
        q_emb = self._embed([query])[0]
        # Vectorized cosine
        norm_docs = np.linalg.norm(self.embeddings, axis=1)
        norm_q = np.linalg.norm(q_emb)
        if norm_q == 0:
            return []
        scores = (self.embeddings @ q_emb) / (norm_docs * norm_q + 1e-12)

        ranked = np.argsort(-scores)[:top_k]
        return [
            SearchResult(
                entity_id=self.entity_ids[i],
                score=float(scores[i]),
                method="semantic",
                snippet=self.documents[i][:200],
            )
            for i in ranked
        ]

    # ------------------------------------------------------------------
    def keyword_fallback(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """BM25 fallback (or TF overlap if rank_bm25 unavailable)."""
        if not self.entity_ids:
            return []
        q_tokens = re.findall(r"\w+", query.lower())
        if not q_tokens:
            return []

        if self.bm25 is not None:
            scores = self.bm25.get_scores(q_tokens)
            ranked = np.argsort(-scores)[:top_k]
            results = []
            for i in ranked:
                matched = [t for t in q_tokens if t in self.documents[i].lower()]
                if not matched and scores[i] <= 0:
                    continue
                results.append(SearchResult(
                    entity_id=self.entity_ids[i],
                    score=float(scores[i]),
                    method="keyword",
                    snippet=self.documents[i][:200],
                    matched_terms=matched,
                ))
            return results

        # Manual TF-overlap fallback
        results: list[SearchResult] = []
        for i, doc in enumerate(self.documents):
            doc_tokens = re.findall(r"\w+", doc.lower())
            tf = sum(doc_tokens.count(t) for t in q_tokens)
            if tf > 0:
                results.append(SearchResult(
                    entity_id=self.entity_ids[i],
                    score=float(tf),
                    method="keyword",
                    snippet=doc[:200],
                    matched_terms=[t for t in q_tokens if t in doc.lower()],
                ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.6,  # weight for semantic
    ) -> list[SearchResult]:
        """Linear combination of semantic + keyword scores (normalized)."""
        sem = self.search(query, top_k=max(top_k * 3, 30))
        kw = self.keyword_fallback(query, top_k=max(top_k * 3, 30))

        # Normalize each set to [0,1]
        def _norm(items: list[SearchResult]) -> dict[str, float]:
            if not items:
                return {}
            mx = max(r.score for r in items)
            return {r.entity_id: (r.score / mx) if mx > 0 else 0.0 for r in items}

        s_map = _norm(sem)
        k_map = _norm(kw)

        all_ids = set(s_map) | set(k_map)
        combined = {
            eid: alpha * s_map.get(eid, 0.0) + (1 - alpha) * k_map.get(eid, 0.0)
            for eid in all_ids
        }
        ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]

        # Build snippets/matched terms from kw results when available
        snippets = {r.entity_id: r.snippet for r in (sem + kw)}
        terms = {r.entity_id: r.matched_terms or [] for r in kw}

        return [
            SearchResult(
                entity_id=eid,
                score=float(score),
                method="hybrid",
                snippet=snippets.get(eid, ""),
                matched_terms=terms.get(eid),
            )
            for eid, score in ranked
        ]

    # ------------------------------------------------------------------
    def expand_query(self, query: str) -> list[str]:
        """Return query expansions: synonyms + KG-neighbor concepts."""
        q_tokens = re.findall(r"\w+", query.lower())
        expansions: list[str] = [query]

        # Synonym lookup
        for tok in q_tokens:
            for syn in SYNONYMS.get(tok, []):
                expansions.append(query + " " + syn)

        # KG concept neighbors
        if self._kg is not None:
            related: set[str] = set()
            for eid in self._kg.all_entities():
                props = self._kg.get_entity(eid)
                text = " ".join(str(v) for v in props.values()).lower()
                if any(t in text for t in q_tokens):
                    # Add keywords/themes as expansion candidates
                    for k in ("dcat:keyword", "dcat:theme", "dct:subject"):
                        v = props.get(k)
                        if isinstance(v, list):
                            related.update(v)
                        elif isinstance(v, str):
                            related.add(v)
            for kw in list(related)[:10]:
                expansions.append(query + " " + kw)

        return list(dict.fromkeys(expansions))  # dedupe, preserve order

    # ------------------------------------------------------------------
    def explain_result(self, entity_id: str, query: str) -> str:
        """Explain why this entity matched a query."""
        if entity_id not in self.entity_ids:
            return f"Entity {entity_id} not in current index."
        idx = self.entity_ids.index(entity_id)
        doc = self.documents[idx]
        q_tokens = re.findall(r"\w+", query.lower())
        matched = [t for t in q_tokens if t in doc.lower()]

        sem_score: float | None = None
        if self.embeddings is not None and self.embeddings.size > 0:
            q_emb = self._embed([query])[0]
            sem_score = _cosine(q_emb, self.embeddings[idx])

        lines = [f"Entity `{entity_id}` matched query `{query}`:"]
        if sem_score is not None:
            lines.append(f"- semantic similarity: {sem_score:.3f}")
        lines.append(f"- keyword overlap: {len(matched)}/{len(q_tokens)} tokens ({matched})")
        snippet = doc[:200] + "\u2026" if len(doc) > 200 else doc
        lines.append(f"- snippet: {snippet}")
        return "\n".join(lines)


__all__ = ["SearchResult", "SemanticMetadataSearch", "SYNONYMS"]

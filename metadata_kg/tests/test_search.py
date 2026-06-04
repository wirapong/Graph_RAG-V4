"""Search tests (PHASE 8 bonus)."""

from __future__ import annotations

import pytest

from metadata_kg.search.semantic_search import SemanticMetadataSearch


def test_index_kg(sample_kg):
    s = SemanticMetadataSearch()
    s.index_kg(sample_kg)
    assert len(s.entity_ids) == 2
    assert s.embeddings is not None
    assert s.embeddings.shape[0] == 2


def test_semantic_search_ranks_relevant_first(sample_kg):
    s = SemanticMetadataSearch()
    s.index_kg(sample_kg)
    results = s.search("covid health", top_k=2)
    assert results
    # The COVID entry should come first
    assert "ds:001" in results[0].entity_id or "covid" in results[0].snippet.lower()


def test_keyword_search_finds_match(sample_kg):
    s = SemanticMetadataSearch()
    s.index_kg(sample_kg)
    results = s.keyword_fallback("Bangkok stations", top_k=2)
    assert results
    # Bangkok entry should be in results
    assert any("ds:002" in r.entity_id for r in results)


def test_hybrid_search_returns_results(sample_kg):
    s = SemanticMetadataSearch()
    s.index_kg(sample_kg)
    results = s.hybrid_search("Bangkok air quality", top_k=2)
    assert results
    assert results[0].method == "hybrid"


def test_expand_query_adds_synonyms(sample_kg):
    s = SemanticMetadataSearch()
    s.index_kg(sample_kg)
    expansions = s.expand_query("covid")
    assert any("coronavirus" in e for e in expansions)


def test_explain_result(sample_kg):
    s = SemanticMetadataSearch()
    s.index_kg(sample_kg)
    eid = s.entity_ids[0]
    txt = s.explain_result(eid, "covid")
    assert eid in txt
    assert "snippet" in txt

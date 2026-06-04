"""Tests for kg_builder.py (PHASE 8.1)."""

from __future__ import annotations

import pytest

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph, link_cross_domain
from metadata_kg.core.metadata_schema import DCATEntityType


def test_add_and_query_entity(empty_kg: MetadataKnowledgeGraph):
    empty_kg.add_entity(
        "ds:test",
        DCATEntityType.Dataset,
        {"dct:title": "Test", "dct:description": "Hello", "dcat:keyword": ["a", "b"]},
    )
    props = empty_kg.get_entity("ds:test")
    assert "rdf:type" in props
    assert props.get("dcmi:title") == "Test" or props.get("dct:title") == "Test"
    assert "ds:test" in empty_kg.all_entities()[0]


def test_relation_added(empty_kg):
    empty_kg.add_entity("a", "dcat:Dataset", {"dct:title": "A", "dct:description": "x"})
    empty_kg.add_entity("b", "dcat:Dataset", {"dct:title": "B", "dct:description": "y"})
    empty_kg.add_relation("a", "mkg:relatedTo", "b")
    # In NX, the edge exists
    nx_edges = list(empty_kg.to_networkx().edges(keys=True))
    assert any("relatedTo" in str(k) for _, _, k in nx_edges)


def test_validate_consistency_flags_missing_mandatory(empty_kg):
    empty_kg.add_entity("ds:bad", DCATEntityType.Dataset, {"dct:keyword": ["x"]})  # no title/desc
    conflicts = empty_kg.validate_consistency()
    rules = [c["rule"] for c in conflicts]
    assert "DCAT_MANDATORY_FIELD" in rules


def test_export_to_turtle(sample_kg, tmp_path):
    out = tmp_path / "out.ttl"
    ttl = sample_kg.export_to_turtle(out)
    assert "dcat:Dataset" in ttl or "@prefix" in ttl
    assert out.exists()
    # Round-trip
    kg2 = MetadataKnowledgeGraph(name="loaded")
    kg2.load_from_turtle(out)
    assert len(kg2) == len(sample_kg)


def test_to_networkx_returns_multigraph(sample_kg):
    g = sample_kg.to_networkx()
    assert g.is_multigraph()
    assert g.number_of_nodes() >= 2


def test_stats(sample_kg):
    s = sample_kg.stats()
    assert s["triples"] > 0
    assert s["entities"] >= 2


def test_link_cross_domain_creates_same_as():
    kg_a = MetadataKnowledgeGraph(name="a")
    kg_a.add_entity("a:1", "dcat:Dataset", {"dct:title": "Shared Concept", "dct:description": "from A"})
    kg_b = MetadataKnowledgeGraph(name="b")
    kg_b.add_entity("b:1", "dcat:Dataset", {"dct:title": "Shared Concept", "dct:description": "from B"})
    merged = link_cross_domain(kg_a, kg_b, shared_key="title")
    # Look for sameAs triples
    ttl = merged.export_to_turtle()
    assert "sameAs" in ttl or "mkg:sameAs" in ttl
    assert merged.stats()["entities"] >= 2


def test_dangling_reference_flagged(empty_kg):
    empty_kg.add_entity("a", "dcat:Dataset", {"dct:title": "A", "dct:description": "d"})
    empty_kg.add_relation("a", "mkg:linksTo", "nonexistent")
    conflicts = empty_kg.validate_consistency()
    rules = {c["rule"] for c in conflicts}
    assert "DANGLING_REFERENCE" in rules

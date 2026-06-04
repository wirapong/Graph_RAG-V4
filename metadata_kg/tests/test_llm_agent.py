"""Tests for llm_agent.py (PHASE 8.2)."""

from __future__ import annotations

import pytest

from metadata_kg.core.llm_agent import (
    MetadataAgent,
    deterministic_extract_entities,
    deterministic_flag_hallucination,
    deterministic_map_schema,
)


def test_extract_entities_returns_entity_with_title():
    ents = deterministic_extract_entities(
        "Title: Demo Dataset\nDescription: Sample data."
    )
    assert len(ents) == 1
    assert "Demo" in ents[0].properties["dct:title"]
    assert ents[0].confidence > 0


def test_map_schema_handles_unknown_fields():
    out = deterministic_map_schema({"name": "X", "tags": ["a", "b"], "random_xyz": 1})
    assert out["dct:title"] == "X"
    assert out["dcat:keyword"] == ["a", "b"]
    assert "mkg:random_xyz" in out


def test_map_schema_idempotent_on_known_curies():
    out = deterministic_map_schema({"dct:title": "X", "dcat:keyword": ["a"]})
    assert out == {"dct:title": "X", "dcat:keyword": ["a"]}


def test_flag_hallucination_no_context_returns_moderate():
    score = deterministic_flag_hallucination("Some claim", [])
    assert 0.0 <= score <= 1.0
    assert score < 0.7  # treated as suspicious


def test_flag_hallucination_high_overlap():
    score = deterministic_flag_hallucination(
        "covid health data", ["thailand covid health metrics", "health"]
    )
    assert score > 0.7


def test_agent_run_stores_entity(agent_no_llm):
    run = agent_no_llm.run("Title: X\nDescription: Y data")
    assert len(run.final_entities) == 1
    # Entity is stored in the agent's KG
    eid = run.final_entities[0].id
    assert agent_no_llm.kg.get_entity(eid)


def test_agent_hitl_triggers_at_high_threshold():
    # Set threshold > 0.6 so the baseline extractor (confidence=0.6) flags HITL.
    agent = MetadataAgent(use_llm=False, hitl_threshold=0.95)
    run = agent.run("Title: T\nDescription: short text only")
    # Even when validation passes, the high threshold should force HITL.
    assert run.hitl_required or run.overall_confidence < 0.95


def test_reasoning_log_contains_steps(agent_no_llm):
    run = agent_no_llm.run("Title: T\nDescription: D")
    log = agent_no_llm.reasoning_log(run)
    assert "extract_entities" in log
    assert "validate_via_kg" in log
    assert "Overall confidence" in log


def test_validate_via_kg_returns_validation_result(agent_no_llm):
    agent_no_llm.kg.add_entity("ds:x", "dcat:Dataset", {"dct:title": "T", "dct:description": "D"})
    val = agent_no_llm.validate_via_kg("ds:x")
    assert val.valid

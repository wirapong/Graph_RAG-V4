"""Tests for lifecycle.py (PHASE 8.3)."""

from __future__ import annotations

import pytest


def test_phase1_creates_entity_and_event(lifecycle):
    r = lifecycle.phase1_creation(
        "Title: X\nDescription: Y data with sufficient text for hallucination grounding."
    )
    assert r.entity_ids, "Phase 1 must create at least one entity"
    eid = r.entity_ids[0]
    chain = lifecycle.lineage.get_provenance_chain(eid)
    actions = [e.action for e in chain]
    assert "creation" in actions


def test_phase2_dedup_detects_similar_pair(lifecycle):
    # Two near-identical entries
    lifecycle.phase1_creation("Title: Iris Dataset\nDescription: Classic iris flower data with 150 samples sepal petal.")
    lifecycle.phase1_creation("Title: Iris Dataset\nDescription: Classic iris flower data with 150 samples sepal petal.")
    r2 = lifecycle.phase2_cleaning()
    assert len(r2.duplicates_merged) >= 1


def test_phase3_maintenance_records_update(lifecycle):
    r = lifecycle.phase1_creation("Title: A\nDescription: B")
    eid = r.entity_ids[0]
    r3 = lifecycle.phase3_maintenance(eid, {"dct:license": "MIT"}, reason="add license")
    chain = lifecycle.lineage.get_provenance_chain(eid)
    assert any(e.action == "update" for e in chain)
    assert r3.events


def test_phase4_archive_and_purge(lifecycle):
    r = lifecycle.phase1_creation("Title: tmp\nDescription: temporary record")
    eid = r.entity_ids[0]

    # archive
    r4a = lifecycle.phase4_retirement([eid], purge=False)
    assert r4a.retired == [eid]
    chain = lifecycle.lineage.get_provenance_chain(eid)
    assert any(e.action == "archive" for e in chain)

    # purge
    r4p = lifecycle.phase4_retirement([eid], purge=True, reason="gdpr test")
    assert eid in r4p.retired
    # After purge, entity should be gone from KG
    assert not lifecycle.kg.get_entity(eid)


def test_all_4_phases_emit_lineage(lifecycle):
    r = lifecycle.phase1_creation("Title: full pipeline\nDescription: covering every lifecycle phase.")
    eid = r.entity_ids[0]
    lifecycle.phase2_cleaning()
    lifecycle.phase3_maintenance(eid, {"dct:license": "CC0"})
    lifecycle.phase4_retirement([eid], purge=False)
    chain = [e.action for e in lifecycle.lineage.get_provenance_chain(eid)]
    assert "creation" in chain
    assert "update" in chain
    assert "archive" in chain


def test_run_full_returns_creation_and_cleaning(lifecycle):
    results = lifecycle.run_full("Title: foo\nDescription: bar")
    assert "creation" in results
    assert "cleaning" in results

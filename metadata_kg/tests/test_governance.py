"""Tests for governance (policy + lineage + xai) — PHASE 8.4."""

from __future__ import annotations

import pytest

from metadata_kg.governance.lineage import DataLineage
from metadata_kg.governance.policy import PolicyEngine, detect_pii
from metadata_kg.governance.xai import ExplainabilityLayer


# ---------- policy ----------
def test_detect_pii_email():
    cats = detect_pii("Contact me at user@example.com please")
    assert "email" in cats


def test_detect_pii_phone():
    cats = detect_pii("Call 081-234-5678 anytime")
    assert "phone" in cats


def test_policy_passes_clean_metadata():
    pe = PolicyEngine()
    result = pe.check_compliance({
        "id": "x", "title": "X", "description": "Public open data set."
    })
    assert result["pass"]


def test_policy_fails_with_pii_in_description():
    pe = PolicyEngine()
    result = pe.check_compliance({
        "id": "x", "title": "X", "description": "Email user@example.com"
    })
    assert not result["pass"]
    rules = [v["rule_id"] for v in result["violated_rules"]]
    assert "GDPR_PII_DESCRIPTION" in rules


def test_policy_fails_missing_mandatory():
    pe = PolicyEngine()
    result = pe.check_compliance({"id": "x"})
    assert not result["pass"]
    assert result["summary"]["error"] >= 2  # missing title + description


def test_policy_load_custom_rules(tmp_path):
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text(
        "rules:\n"
        "  - id: CUSTOM_FORBID_DRAFT\n"
        "    description: \"No draft licenses\"\n"
        "    severity: warning\n"
        "    check: forbid_value\n"
        "    field: license\n"
        "    forbid_values: [draft]\n"
    )
    pe = PolicyEngine()
    pe.load_rules(rules_yaml)
    result = pe.check_compliance({"id": "x", "title": "T", "description": "D", "license": "draft"})
    rules = [v["rule_id"] for v in result["violated_rules"]]
    assert "CUSTOM_FORBID_DRAFT" in rules


# ---------- lineage ----------
def test_lineage_records_chain_in_order():
    lin = DataLineage()
    lin.record_event("e1", "creation", "agent")
    lin.record_event("e1", "update", "agent")
    chain = lin.get_provenance_chain("e1")
    assert [e.action for e in chain] == ["creation", "update"]
    assert chain[1].prev_event_id == chain[0].event_id


def test_lineage_export_jsonld(tmp_path):
    lin = DataLineage()
    lin.record_event("e1", "creation", "agent")
    out = tmp_path / "lineage.jsonld"
    doc = lin.export_lineage_graph(out)
    assert out.exists()
    assert "@context" in doc and "@graph" in doc


# ---------- xai ----------
def test_xai_explanation_non_empty(sample_kg):
    lin = DataLineage()
    lin.record_event(sample_kg.all_entities()[0], "creation", "agent", confidence=0.8)
    xai = ExplainabilityLayer(sample_kg, lin)
    md = xai.explain_decision(sample_kg.all_entities()[0])
    assert "Decision explanation" in md
    assert len(md) > 100


def test_xai_confidence_report(sample_kg):
    lin = DataLineage()
    eid = sample_kg.all_entities()[0]
    lin.record_event(eid, "creation", "agent", confidence=0.9)
    xai = ExplainabilityLayer(sample_kg, lin)
    rep = xai.confidence_report(eid)
    assert rep["overall_confidence"] == pytest.approx(0.9)
    assert rep["events_considered"] == 1

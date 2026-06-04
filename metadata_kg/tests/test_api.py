"""Smoke tests for FastAPI routes (PHASE 8 bonus)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from metadata_kg.api import routes as routes_mod


@pytest.fixture
def client():
    # Reset state for isolation
    routes_mod.state = routes_mod.AppState()
    return TestClient(routes_mod.app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()["data"]


def test_ingest_and_metadata_roundtrip(client):
    r = client.post(
        "/ingest",
        json={"text": "Title: API Test\nDescription: For routing tests with sufficient text.", "source": "test"},
    )
    assert r.status_code == 200, r.text
    eid = r.json()["data"]["entity_ids"][0]
    r2 = client.get(f"/metadata/{eid}")
    assert r2.status_code == 200
    assert "metadata" in r2.json()["data"]


def test_search_endpoint(client):
    client.post("/ingest", json={"text": "Title: Search target\nDescription: about climate."})
    r = client.get("/search", params={"q": "climate"})
    assert r.status_code == 200
    assert "results" in r.json()["data"]


def test_policy_check_detects_pii(client):
    r = client.post(
        "/policy/check",
        json={"metadata": {"id": "x", "title": "T", "description": "Email user@example.com"}},
    )
    assert r.status_code == 200
    assert r.json()["data"]["pass"] is False


def test_hitl_queue_and_review(client):
    ingest_resp = client.post("/ingest", json={"text": "Title: hitl-test\nDescription: short text."})
    eid = ingest_resp.json()["data"]["entity_ids"][0]
    qr = client.get("/hitl/queue")
    assert qr.status_code == 200

    rr = client.post(
        "/hitl/review",
        json={
            "entity_id": eid,
            "approved": True,
            "corrections": {"dct:license": "MIT"},
            "reviewer": "tester@kku.ac.th",
        },
    )
    assert rr.status_code == 200
    assert rr.json()["data"]["approved"]


def test_explain_endpoint(client):
    ing = client.post("/ingest", json={"text": "Title: explain-me\nDescription: testing xai endpoint."})
    eid = ing.json()["data"]["entity_ids"][0]
    r = client.get(f"/explain/{eid}")
    assert r.status_code == 200
    assert "explanation_md" in r.json()["data"]


def test_stats_endpoint(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert "kg" in r.json()["data"]

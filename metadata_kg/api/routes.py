"""REST API + Human-in-the-Loop endpoints (PHASE 7).

Endpoints:
- POST   /ingest          submit raw document for metadata extraction
- GET    /metadata/{id}   retrieve metadata + provenance chain
- GET    /search          semantic search (query param)
- POST   /validate/{id}   trigger KG consistency check
- GET    /explain/{id}    XAI explanation
- POST   /hitl/review     human expert submits correction/approval
- GET    /hitl/queue      list items flagged for human review
- POST   /policy/check    run policy compliance

All endpoints return: { data, confidence, lineage_url, warnings[] }
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph
from metadata_kg.core.llm_agent import MetadataAgent
from metadata_kg.governance.lineage import DataLineage
from metadata_kg.governance.policy import PolicyEngine
from metadata_kg.governance.xai import ExplainabilityLayer
from metadata_kg.pipeline.lifecycle import MetadataLifecycle
from metadata_kg.pipeline.validate import validate_entity, validate_graph
from metadata_kg.search.semantic_search import SemanticMetadataSearch


# ---------------------------------------------------------------------------
# Shared in-process state (single-tenant by default)
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self) -> None:
        self.kg = MetadataKnowledgeGraph(name="api_kg")
        self.lineage = DataLineage()
        self.agent = MetadataAgent(kg=self.kg)
        self.lifecycle = MetadataLifecycle(kg=self.kg, agent=self.agent, lineage=self.lineage)
        self.policy = PolicyEngine()
        self.xai = ExplainabilityLayer(self.kg, self.lineage)
        self.search = SemanticMetadataSearch()
        self.hitl_queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def reindex(self) -> None:
        with self._lock:
            self.search.index_kg(self.kg)


# Singleton state for the FastAPI app
state = AppState()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    text: str = Field(..., description="Raw document or text")
    source: str | None = None


class HITLReview(BaseModel):
    entity_id: str
    approved: bool
    corrections: dict[str, Any] = Field(default_factory=dict)
    reviewer: str
    comment: str | None = None


class PolicyCheckRequest(BaseModel):
    metadata: dict[str, Any]


def _envelope(
    data: Any,
    *,
    confidence: float | None = None,
    entity_id: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Standard API response envelope."""
    return {
        "data": data,
        "confidence": confidence,
        "lineage_url": f"/explain/{entity_id}" if entity_id else None,
        "warnings": warnings or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Metadata KG API",
    description="AI Metadata Automation with Knowledge Graphs (DCAT2/DCMI).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return _envelope({
        "service": "metadata-kg",
        "version": "0.1.0",
        "endpoints": [
            "POST /ingest", "GET /metadata/{id}", "GET /search",
            "POST /validate/{id}", "GET /explain/{id}",
            "POST /hitl/review", "GET /hitl/queue",
            "POST /policy/check", "GET /stats",
        ],
    })


@app.get("/stats")
def stats() -> dict[str, Any]:
    return _envelope({
        "kg": state.kg.stats(),
        "lineage_events": len(state.lineage),
        "hitl_queue_size": len(state.hitl_queue),
        "indexed_entities": len(state.search.entity_ids),
    })


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, Any]:
    """Submit raw document → extract metadata → store in KG → return summary."""
    results = state.lifecycle.run_full(req.text, source=req.source)
    creation = results["creation"]
    state.reindex()

    warnings = []
    if creation.notes:
        warnings.extend(creation.notes)
        # Enqueue HITL items
        for eid in creation.entity_ids:
            chain = state.lineage.get_provenance_chain(eid)
            if any(e.action == "hitl_flag" for e in chain):
                state.hitl_queue.append({
                    "entity_id": eid,
                    "reasons": creation.notes,
                    "enqueued_at": datetime.now(timezone.utc).isoformat(),
                })

    return _envelope(
        {
            "entity_ids": creation.entity_ids,
            "creation_events": creation.events,
            "cleaning": {
                "duplicates_merged": results["cleaning"].duplicates_merged,
                "suggestions_count": len(results["cleaning"].suggestions),
            },
        },
        confidence=1.0 if not warnings else 0.7,
        entity_id=creation.entity_ids[0] if creation.entity_ids else None,
        warnings=warnings,
    )


@app.get("/metadata/{entity_id:path}")
def get_metadata(entity_id: str) -> dict[str, Any]:
    props = state.kg.get_entity(entity_id)
    if not props:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    chain = state.lineage.get_provenance_chain(entity_id)
    return _envelope(
        {
            "entity_id": entity_id,
            "metadata": props,
            "provenance": [ev.to_dict() for ev in chain],
        },
        entity_id=entity_id,
    )


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
    method: str = Query("hybrid", pattern="^(semantic|keyword|hybrid)$"),
) -> dict[str, Any]:
    state.reindex()
    if method == "semantic":
        results = state.search.search(q, top_k=top_k)
    elif method == "keyword":
        results = state.search.keyword_fallback(q, top_k=top_k)
    else:
        results = state.search.hybrid_search(q, top_k=top_k)

    return _envelope(
        {
            "query": q,
            "method": method,
            "results": [r.to_dict() for r in results],
            "expansions": state.search.expand_query(q)[:5],
        }
    )


@app.post("/validate/{entity_id:path}")
def validate(entity_id: str) -> dict[str, Any]:
    if not state.kg.get_entity(entity_id):
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    report = validate_entity(state.kg, entity_id)
    consistency_issues = [
        c for c in state.kg.validate_consistency() if c.get("entity") == entity_id
    ]
    return _envelope(
        {"validation": report, "consistency_issues": consistency_issues},
        entity_id=entity_id,
        warnings=report.get("errors", []),
    )


@app.get("/explain/{entity_id:path}")
def explain(entity_id: str) -> dict[str, Any]:
    if not state.kg.get_entity(entity_id) and not state.lineage.get_provenance_chain(entity_id):
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    md = state.xai.explain_decision(entity_id)
    conf = state.xai.confidence_report(entity_id)
    return _envelope(
        {"explanation_md": md, "confidence_report": conf},
        confidence=conf.get("overall_confidence"),
        entity_id=entity_id,
    )


# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------
@app.post("/hitl/review")
def hitl_review(review: HITLReview) -> dict[str, Any]:
    """Human expert submits a correction/approval."""
    if not state.kg.get_entity(review.entity_id):
        raise HTTPException(status_code=404, detail=f"Entity {review.entity_id} not found")

    if review.corrections:
        state.lifecycle.phase3_maintenance(
            review.entity_id,
            review.corrections,
            agent=f"human:{review.reviewer}",
            reason="HITL correction",
        )

    state.lineage.record_event(
        entity_id=review.entity_id,
        action="hitl_review",
        agent=f"human:{review.reviewer}",
        details={"approved": review.approved, "comment": review.comment, "corrections": review.corrections},
        confidence=1.0 if review.approved else 0.5,
    )

    # Remove from queue
    state.hitl_queue = [q for q in state.hitl_queue if q["entity_id"] != review.entity_id]

    return _envelope(
        {"entity_id": review.entity_id, "approved": review.approved},
        confidence=1.0 if review.approved else 0.5,
        entity_id=review.entity_id,
    )


@app.get("/hitl/queue")
def hitl_queue() -> dict[str, Any]:
    return _envelope({"queue_size": len(state.hitl_queue), "items": state.hitl_queue})


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
@app.post("/policy/check")
def policy_check(req: PolicyCheckRequest) -> dict[str, Any]:
    result = state.policy.check_compliance(req.metadata)
    return _envelope(
        result,
        confidence=1.0 if result["pass"] else 0.0,
        warnings=[v["message"] for v in result["violated_rules"] if v["severity"] == "error"],
    )


# ---------------------------------------------------------------------------
# Graph-wide
# ---------------------------------------------------------------------------
@app.get("/graph/validate")
def graph_validate() -> dict[str, Any]:
    return _envelope(validate_graph(state.kg))


@app.get("/graph/turtle")
def graph_turtle() -> dict[str, Any]:
    return _envelope({"turtle": state.kg.export_to_turtle()})


@app.get("/graph/lineage")
def graph_lineage() -> dict[str, Any]:
    return _envelope(state.lineage.export_lineage_graph())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run() -> None:  # pragma: no cover
    """Entry point: `metadata-kg-api`."""
    import os

    import uvicorn
    uvicorn.run(
        "metadata_kg.api.routes:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()

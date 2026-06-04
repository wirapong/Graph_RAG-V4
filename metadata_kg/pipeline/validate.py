"""KG-based consistency validation (PHASE 4 — used by Phase 2 Cleaning)."""

from __future__ import annotations

from typing import Any

from loguru import logger

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph
from metadata_kg.core.metadata_schema import DCAT_MANDATORY_FIELDS


def validate_entity(kg: MetadataKnowledgeGraph, entity_id: str) -> dict[str, Any]:
    """Per-entity validation report."""
    props = kg.get_entity(entity_id)
    if not props:
        return {"entity_id": entity_id, "valid": False, "errors": ["entity not found"]}

    normalized = {MetadataKnowledgeGraph._strip_prefix(k) for k in props}
    missing = [f for f in DCAT_MANDATORY_FIELDS if f != "id" and f not in normalized]
    suggestions: list[dict[str, Any]] = []
    for f in missing:
        suggestions.append({
            "field": f,
            "action": "fill_in",
            "confidence": 0.5,
            "note": f"Mandatory DCAT field '{f}' missing; HITL recommended.",
        })

    return {
        "entity_id": entity_id,
        "valid": not missing,
        "errors": [f"missing:{f}" for f in missing],
        "suggestions": suggestions,
    }


def validate_graph(kg: MetadataKnowledgeGraph) -> dict[str, Any]:
    """Full-graph validation summary."""
    issues = kg.validate_consistency()
    per_entity = [validate_entity(kg, eid) for eid in kg.all_entities()]
    invalid_count = sum(1 for r in per_entity if not r["valid"])
    logger.info(f"validate_graph: {invalid_count}/{len(per_entity)} invalid; {len(issues)} consistency issues")
    return {
        "summary": {
            "entities_total": len(per_entity),
            "entities_invalid": invalid_count,
            "consistency_issues": len(issues),
        },
        "consistency_issues": issues,
        "per_entity": per_entity,
    }


__all__ = ["validate_entity", "validate_graph"]

"""Explainability layer (PHASE 5.3).

Generates human-readable decision explanations by combining:
  - KG entity state
  - Lineage chain (DataLineage)
  - Agent reasoning log
  - Per-field confidence scores
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph
from metadata_kg.governance.lineage import DataLineage


class ExplainabilityLayer:
    """XAI explanations grounded in lineage + KG state."""

    def __init__(self, kg: MetadataKnowledgeGraph, lineage: DataLineage) -> None:
        self.kg = kg
        self.lineage = lineage

    # ------------------------------------------------------------------
    def explain_decision(self, entity_id: str) -> str:
        """Produce a Markdown explanation of how an entity reached its current state."""
        props = self.kg.get_entity(entity_id)
        chain = self.lineage.get_provenance_chain(entity_id)

        if not props and not chain:
            return f"No information found for {entity_id}."

        lines = [f"# Decision explanation for `{entity_id}`", ""]

        if props:
            lines.append("## Current state")
            for k, v in sorted(props.items()):
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        if chain:
            lines.append("## Lineage")
            lines.append(f"Total events: **{len(chain)}**")
            lines.append("")
            for i, ev in enumerate(chain, 1):
                lines.append(f"### {i}. {ev.action} — {ev.timestamp.isoformat()}")
                lines.append(f"- agent: `{ev.agent}`")
                if ev.confidence is not None:
                    lines.append(f"- confidence: {ev.confidence:.2f}")
                if ev.details:
                    lines.append("- details:")
                    for k, v in ev.details.items():
                        v_str = str(v)
                        if len(v_str) > 200:
                            v_str = v_str[:200] + "…"
                        lines.append(f"  - {k}: {v_str}")
                lines.append("")
        else:
            lines.append("_No lineage events recorded._")

        # Reasoning summary
        actions = [ev.action for ev in chain]
        lines.append("## Reasoning summary")
        lines.append(
            f"This entity was produced through the following actions: "
            f"`{' → '.join(actions)}`. "
            + (f"It is currently flagged for HITL review." if "hitl_flag" in actions else "")
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def confidence_report(self, entity_id: str) -> dict[str, Any]:
        """Per-field confidence dict for an entity."""
        chain = self.lineage.get_provenance_chain(entity_id)
        # Aggregate: use the most recent confidence per detail key, fallback to event.confidence
        per_field: dict[str, float] = {}
        overall: list[float] = []

        for ev in chain:
            if ev.confidence is not None:
                overall.append(ev.confidence)
            # Look for delta entries (Phase 3 maintenance pattern)
            delta = ev.details.get("delta") if ev.details else None
            if isinstance(delta, dict):
                for k in delta:
                    per_field[k] = ev.confidence if ev.confidence is not None else 1.0

        # Also reflect KG fields with no explicit confidence as 1.0 default
        for k in self.kg.get_entity(entity_id):
            per_field.setdefault(k, 1.0)

        return {
            "entity_id": entity_id,
            "overall_confidence": (sum(overall) / len(overall)) if overall else None,
            "per_field": per_field,
            "events_considered": len(chain),
        }


__all__ = ["ExplainabilityLayer"]

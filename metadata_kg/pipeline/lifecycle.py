"""Metadata Lifecycle (PHASE 4).

Four phases:
  1. Creation     — LLM extraction, auto-tag with DCAT2 vocabulary
  2. Cleaning     — Deduplicate via KG similarity, flag missing, AI suggestions
  3. Maintenance  — Incremental updates with version stamps
  4. Retirement   — Archive / GDPR purge

Every phase emits lifecycle_event → DataLineage.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph
from metadata_kg.core.llm_agent import AgentRun, MetadataAgent
from metadata_kg.core.metadata_schema import DCAT_MANDATORY_FIELDS
from metadata_kg.governance.lineage import DataLineage
from metadata_kg.pipeline.extract import extract_from_input
from metadata_kg.pipeline.validate import validate_entity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _entity_tokens(props: dict[str, Any]) -> set[str]:
    text_blob = " ".join(str(v) for v in props.values() if isinstance(v, (str, int, float)))
    return set(re.findall(r"\w+", text_blob.lower()))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class LifecycleResult:
    entity_ids: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)   # event_ids
    duplicates_merged: list[tuple[str, str]] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"LifecycleResult(entities={len(self.entity_ids)}, events={len(self.events)}, "
            f"dups={len(self.duplicates_merged)}, retired={len(self.retired)})"
        )


# ---------------------------------------------------------------------------
# MetadataLifecycle
# ---------------------------------------------------------------------------
class MetadataLifecycle:
    """End-to-end lifecycle manager."""

    AGENT_ID = "MetadataAgent/v1"

    def __init__(
        self,
        kg: MetadataKnowledgeGraph | None = None,
        agent: MetadataAgent | None = None,
        lineage: DataLineage | None = None,
        dedup_threshold: float = 0.85,
        retire_after_days: int = 180,
    ) -> None:
        self.kg = kg if kg is not None else MetadataKnowledgeGraph(name="lifecycle_kg")
        # Make sure the agent writes to the same KG
        if agent is None:
            agent = MetadataAgent(kg=self.kg)
        else:
            agent.kg = self.kg
        self.agent = agent
        self.lineage = lineage if lineage is not None else DataLineage()
        self.dedup_threshold = dedup_threshold
        self.retire_after_days = retire_after_days

    # ------------------------------------------------------------------
    # Phase 1 — Creation
    # ------------------------------------------------------------------
    def phase1_creation(self, input: str | dict[str, Any], *, source: str | None = None) -> LifecycleResult:
        result = LifecycleResult()
        run: AgentRun = extract_from_input(input, agent=self.agent, source=source)

        for ent in run.final_entities:
            # Already stored by the agent into self.kg
            result.entity_ids.append(ent.id)
            ev = self.lineage.record_event(
                entity_id=ent.id,
                action="creation",
                agent=self.AGENT_ID,
                details={
                    "source": source or "<inline>",
                    "extracted_keys": list(ent.properties.keys()),
                    "run_id": run.run_id,
                },
                confidence=ent.confidence,
            )
            result.events.append(ev.event_id)

        if run.hitl_required:
            result.notes.append(f"HITL required: {run.hitl_reasons}")
            for ent_id in result.entity_ids:
                ev = self.lineage.record_event(
                    entity_id=ent_id,
                    action="hitl_flag",
                    agent=self.AGENT_ID,
                    details={"reasons": run.hitl_reasons},
                    confidence=run.overall_confidence,
                )
                result.events.append(ev.event_id)

        logger.info(f"Phase 1 creation → {result}")
        return result

    # ------------------------------------------------------------------
    # Phase 2 — Cleaning
    # ------------------------------------------------------------------
    def phase2_cleaning(self, entity_ids: list[str] | None = None) -> LifecycleResult:
        result = LifecycleResult()
        entity_ids = entity_ids or self.kg.all_entities()

        # 2a. Validate + suggestions
        for eid in entity_ids:
            report = validate_entity(self.kg, eid)
            if not report["valid"]:
                result.suggestions.append(report)
                ev = self.lineage.record_event(
                    entity_id=eid,
                    action="cleaning_validation",
                    agent=self.AGENT_ID,
                    details=report,
                    confidence=0.5,
                )
                result.events.append(ev.event_id)

        # 2b. Deduplication (Jaccard on token sets — stand-in for embedding cosine)
        ids = list(entity_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ta = _entity_tokens(self.kg.get_entity(a))
                tb = _entity_tokens(self.kg.get_entity(b))
                sim = _jaccard(ta, tb)
                if sim >= self.dedup_threshold:
                    result.duplicates_merged.append((a, b))
                    self.kg.add_relation(a, "mkg:sameAs", b)
                    ev = self.lineage.record_event(
                        entity_id=a,
                        action="deduplicate",
                        agent=self.AGENT_ID,
                        details={"duplicate_of": b, "similarity": round(sim, 3), "method": "jaccard"},
                        confidence=sim,
                    )
                    result.events.append(ev.event_id)

        logger.info(f"Phase 2 cleaning → {result}")
        return result

    # ------------------------------------------------------------------
    # Phase 3 — Maintenance
    # ------------------------------------------------------------------
    def phase3_maintenance(
        self,
        entity_id: str,
        updates: dict[str, Any],
        agent: str | None = None,
        *,
        reason: str | None = None,
    ) -> LifecycleResult:
        """Apply incremental updates with version stamping.

        Implements `continual learning stub`: instead of overwriting, every
        change is logged as a versioned event so older state can be reconstructed.
        """
        result = LifecycleResult()
        agent_id = agent or self.AGENT_ID

        before = self.kg.get_entity(entity_id)
        from metadata_kg.core.kg_builder import _ensure_uri  # local import to avoid cycles
        if not before:
            result.notes.append(f"Entity {entity_id} not found; creating new.")
            self.kg.add_entity(entity_id, "dcat:Dataset", updates)
        else:
            subj = _ensure_uri(entity_id)
            for k, v in updates.items():
                self.kg._add_property(subj, k, v)

        delta = {k: {"new": updates[k], "previous": before.get(k) if before else None} for k in updates}
        ev = self.lineage.record_event(
            entity_id=entity_id,
            action="update",
            agent=agent_id,
            details={"delta": delta, "reason": reason},
            confidence=1.0,
        )
        result.events.append(ev.event_id)
        result.entity_ids.append(entity_id)
        logger.info(f"Phase 3 maintenance: {entity_id} updated with {list(updates.keys())}")
        return result

    # ------------------------------------------------------------------
    # Phase 4 — Retirement
    # ------------------------------------------------------------------
    def phase4_retirement(
        self,
        entity_ids: list[str] | None = None,
        *,
        purge: bool = False,
        reason: str | None = None,
    ) -> LifecycleResult:
        """Archive unused (or purge on GDPR request)."""
        result = LifecycleResult()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.retire_after_days)

        candidates = entity_ids or self._stale_entities(cutoff)

        for eid in candidates:
            if purge:
                self.purge_entity(eid, reason=reason or "GDPR purge")
                result.retired.append(eid)
                ev = self.lineage.record_event(
                    entity_id=eid,
                    action="purge",
                    agent=self.AGENT_ID,
                    details={"reason": reason or "GDPR purge", "irreversible": True},
                )
            else:
                # Mark as archived
                from metadata_kg.core.kg_builder import _ensure_uri  # type: ignore
                subj = _ensure_uri(eid)
                self.kg._add_property(subj, "mkg:archived", True)
                self.kg._add_property(subj, "mkg:archivedAt", now.isoformat())
                result.retired.append(eid)
                ev = self.lineage.record_event(
                    entity_id=eid,
                    action="archive",
                    agent=self.AGENT_ID,
                    details={"reason": reason or "stale", "cutoff": cutoff.isoformat()},
                )
            result.events.append(ev.event_id)

        logger.info(f"Phase 4 retirement → {result}")
        return result

    def purge_entity(self, entity_id: str, reason: str = "GDPR") -> None:
        """GDPR-compliant: remove all triples about an entity."""
        from metadata_kg.core.kg_builder import _ensure_uri  # type: ignore
        subj = _ensure_uri(entity_id)
        triples_to_remove = list(self.kg.rdf.triples((subj, None, None))) + \
                            list(self.kg.rdf.triples((None, None, subj)))
        for t in triples_to_remove:
            self.kg.rdf.remove(t)
        # Remove from NX
        if str(subj) in self.kg.nx:
            self.kg.nx.remove_node(str(subj))
        logger.warning(f"Purged entity {entity_id}: {len(triples_to_remove)} triples removed (reason: {reason})")

    def _stale_entities(self, cutoff: datetime) -> list[str]:
        """Entities with no event after `cutoff`."""
        stale = []
        for eid in self.kg.all_entities():
            chain = self.lineage.get_provenance_chain(eid)
            if not chain or chain[-1].timestamp < cutoff:
                stale.append(eid)
        return stale

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------
    def run_full(self, input: str | dict[str, Any], *, source: str | None = None) -> dict[str, LifecycleResult]:
        """Execute Phase 1 + Phase 2 in sequence for a fresh input."""
        r1 = self.phase1_creation(input, source=source)
        r2 = self.phase2_cleaning(r1.entity_ids)
        return {"creation": r1, "cleaning": r2}


__all__ = ["LifecycleResult", "MetadataLifecycle"]

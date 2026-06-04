"""Data lineage tracker (PHASE 5.1).

Records every lifecycle event with PROV-O semantics and exports JSON-LD.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class LineageEvent:
    """A single provenance event (PROV-O Activity)."""

    event_id: str
    entity_id: str
    action: str            # creation | mapping | validation | update | retire | hitl_review | ...
    agent: str             # agent_id (e.g. 'MetadataAgent/v1' or 'human:wirach@kku.ac.th')
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    prev_event_id: str | None = None  # forms a chain per entity

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class DataLineage:
    """In-memory lineage store with JSON-LD export.

    Thread-safe via lock. Can persist to disk if storage_dir given.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._events: list[LineageEvent] = []
        self._by_entity: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_event(
        self,
        entity_id: str,
        action: str,
        agent: str,
        timestamp: datetime | None = None,
        details: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> LineageEvent:
        timestamp = timestamp or datetime.now(timezone.utc)
        with self._lock:
            prev = self._by_entity.get(entity_id, [])
            prev_id = prev[-1] if prev else None
            event = LineageEvent(
                event_id=str(uuid.uuid4()),
                entity_id=entity_id,
                action=action,
                agent=agent,
                timestamp=timestamp,
                details=details or {},
                confidence=confidence,
                prev_event_id=prev_id,
            )
            self._events.append(event)
            self._by_entity.setdefault(entity_id, []).append(event.event_id)

            if self.storage_dir:
                self._persist_event(event)

        logger.debug(f"Lineage: {entity_id} → {action} by {agent}")
        return event

    def _persist_event(self, event: LineageEvent) -> None:
        """Append to per-entity JSONL file."""
        safe_id = event.entity_id.replace("/", "_").replace(":", "_")
        path = self.storage_dir / f"{safe_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_provenance_chain(self, entity_id: str) -> list[LineageEvent]:
        """Return all events for an entity, oldest first."""
        with self._lock:
            ids = self._by_entity.get(entity_id, [])
            events_by_id = {e.event_id: e for e in self._events}
            return [events_by_id[i] for i in ids if i in events_by_id]

    def get_event(self, event_id: str) -> LineageEvent | None:
        for e in self._events:
            if e.event_id == event_id:
                return e
        return None

    def all_events(self) -> list[LineageEvent]:
        return list(self._events)

    def entities_touched(self) -> list[str]:
        return list(self._by_entity.keys())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_lineage_graph(self, path: str | Path | None = None) -> dict[str, Any]:
        """Export as JSON-LD using PROV-O vocabulary."""
        context = {
            "@vocab": "http://www.w3.org/ns/prov#",
            "mkg": "https://wirapongc.kku.ac.th/ns/metadata-kg#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "timestamp": {"@id": "http://www.w3.org/ns/prov#atTime", "@type": "xsd:dateTime"},
            "agent": "http://www.w3.org/ns/prov#wasAssociatedWith",
            "entity_id": "http://www.w3.org/ns/prov#used",
            "prev": "http://www.w3.org/ns/prov#wasInformedBy",
        }
        graph_nodes: list[dict[str, Any]] = []
        for e in self._events:
            node: dict[str, Any] = {
                "@id": f"mkg:event/{e.event_id}",
                "@type": "Activity",
                "action": e.action,
                "agent": e.agent,
                "entity_id": e.entity_id,
                "timestamp": e.timestamp.isoformat(),
            }
            if e.confidence is not None:
                node["confidence"] = e.confidence
            if e.prev_event_id:
                node["prev"] = f"mkg:event/{e.prev_event_id}"
            if e.details:
                node["details"] = e.details
            graph_nodes.append(node)

        doc = {"@context": context, "@graph": graph_nodes}
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
            logger.info(f"Lineage exported to {path}")
        return doc

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"<DataLineage events={len(self._events)} entities={len(self._by_entity)}>"


__all__ = ["DataLineage", "LineageEvent"]

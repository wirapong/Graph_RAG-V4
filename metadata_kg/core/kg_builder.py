"""Knowledge Graph construction with rdflib + networkx.

PHASE 2 implementation.

Supports:
- add_entity / add_relation
- validate_consistency
- export_to_turtle / load_from_turtle
- to_networkx (for GNN / visualization)
- DCAT2 / DCMI schema alignment via metadata_schema.NAMESPACES
- Cross-domain linking
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger
from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCAT, DCTERMS, FOAF

from metadata_kg.core.metadata_schema import (
    DCAT_MANDATORY_FIELDS,
    NAMESPACES,
    DCATEntityType,
    DomainExtension,
)

MKG = NAMESPACES["mkg"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_uri(value: str, default_ns: Namespace = MKG) -> URIRef:
    """Resolve a string to a URIRef.

    - If `value` looks like a full URI (has scheme), return as-is.
    - If `value` has a prefix like 'dcat:Dataset', expand using NAMESPACES.
    - Otherwise, mint a URI under the default namespace.
    """
    if "://" in value:
        return URIRef(value)
    if ":" in value:
        prefix, local = value.split(":", 1)
        ns = NAMESPACES.get(prefix)
        if ns is not None:
            return URIRef(str(ns) + local)
    return URIRef(str(default_ns) + value)


def _python_to_rdf(value: Any) -> Literal | URIRef:
    """Convert a Python value to an rdflib node."""
    if isinstance(value, URIRef):
        return value
    if isinstance(value, str) and ("://" in value or value.startswith(("mkg:", "dct:", "dcat:"))):
        try:
            return _ensure_uri(value)
        except Exception:
            return Literal(value)
    return Literal(value)


# ---------------------------------------------------------------------------
# MetadataKnowledgeGraph
# ---------------------------------------------------------------------------
class MetadataKnowledgeGraph:
    """Dual-store KG: rdflib for semantics, networkx for graph analytics."""

    def __init__(self, base_uri: str = str(MKG), name: str = "metadata_kg") -> None:
        self.name = name
        self.base_uri = base_uri
        self.rdf = Graph()
        self.nx = nx.MultiDiGraph()

        # Bind common prefixes
        for prefix, ns in NAMESPACES.items():
            self.rdf.bind(prefix, ns)

        logger.debug(f"Initialized MetadataKnowledgeGraph(name={name}, base_uri={base_uri})")

    # ------------------------------------------------------------------
    # Entity / relation API
    # ------------------------------------------------------------------
    def add_entity(
        self,
        id: str,
        type: str | DCATEntityType,
        properties: dict[str, Any] | None = None,
    ) -> URIRef:
        """Add an entity (RDF subject) with rdf:type and properties.

        Args:
            id: identifier (will be resolved to URI)
            type: entity type, e.g. 'dcat:Dataset' or DCATEntityType.Dataset
            properties: flat dict of predicate → value(s).
              Predicates may be 'dct:title', 'dcat:keyword', or full URIs.

        Returns:
            URIRef of the entity.
        """
        properties = properties or {}
        type_str = type.value if isinstance(type, DCATEntityType) else type

        subj = _ensure_uri(id)
        type_uri = _ensure_uri(type_str)

        # RDF
        self.rdf.add((subj, RDF.type, type_uri))
        # NX
        self.nx.add_node(str(subj), type=type_str, **{k: v for k, v in properties.items() if isinstance(v, (str, int, float, bool))})

        for pred, value in properties.items():
            self._add_property(subj, pred, value)

        logger.debug(f"add_entity({id}, {type_str}) → {subj}")
        return subj

    def add_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
    ) -> tuple[URIRef, URIRef, URIRef]:
        """Add a typed relation between two entities."""
        s = _ensure_uri(subject)
        p = _ensure_uri(predicate)
        o = _ensure_uri(object)

        self.rdf.add((s, p, o))
        self.nx.add_edge(str(s), str(o), key=str(p), predicate=str(p))

        logger.debug(f"add_relation({subject} --{predicate}--> {object})")
        return s, p, o

    def _add_property(self, subj: URIRef, pred: str, value: Any) -> None:
        """Add a literal or URI-valued property."""
        p_uri = _ensure_uri(pred)
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for v in value:
                self.rdf.add((subj, p_uri, _python_to_rdf(v)))
        else:
            self.rdf.add((subj, p_uri, _python_to_rdf(value)))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_entity(self, id: str) -> dict[str, Any]:
        """Return all properties for an entity as a flat dict."""
        subj = _ensure_uri(id)
        result: dict[str, list[Any]] = {}
        for _, p, o in self.rdf.triples((subj, None, None)):
            key = self.rdf.namespace_manager.normalizeUri(p)
            result.setdefault(key, []).append(str(o))
        # Flatten singleton lists
        return {k: (v[0] if len(v) == 1 else v) for k, v in result.items()}

    def entities_of_type(self, type: str | DCATEntityType) -> list[str]:
        """Return all entity URIs of a given type."""
        type_str = type.value if isinstance(type, DCATEntityType) else type
        type_uri = _ensure_uri(type_str)
        return [str(s) for s in self.rdf.subjects(RDF.type, type_uri)]

    def all_entities(self) -> list[str]:
        """Return all distinct entity URIs in the graph."""
        return list({str(s) for s in self.rdf.subjects()})

    # ------------------------------------------------------------------
    # Consistency validation
    # ------------------------------------------------------------------
    def validate_consistency(self) -> list[dict[str, Any]]:
        """Run lightweight consistency checks.

        Returns a list of conflict descriptors:
            { entity, rule, severity, message }
        """
        conflicts: list[dict[str, Any]] = []

        # 1. Every Dataset must have all DCAT_MANDATORY_FIELDS
        for ds_uri in self.entities_of_type(DCATEntityType.Dataset):
            props = self.get_entity(ds_uri)
            normalized_keys = {self._strip_prefix(k) for k in props}
            for f in DCAT_MANDATORY_FIELDS:
                if f == "id":
                    continue
                if f not in normalized_keys:
                    conflicts.append({
                        "entity": ds_uri,
                        "rule": "DCAT_MANDATORY_FIELD",
                        "field": f,
                        "severity": "error",
                        "message": f"dcat:Dataset {ds_uri} missing mandatory field '{f}'",
                    })

        # 2. Conflicting rdf:type assignments
        for s in set(self.rdf.subjects()):
            types = list(self.rdf.objects(s, RDF.type))
            if len(types) > 1:
                conflicts.append({
                    "entity": str(s),
                    "rule": "MULTIPLE_TYPES",
                    "severity": "warning",
                    "message": f"Entity has multiple rdf:type values: {[str(t) for t in types]}",
                })

        # 3. Dangling relations (object referenced but never typed)
        # Build set of typed subjects (entities that have rdf:type).
        typed_subjects = {str(s) for s in self.rdf.subjects(RDF.type, None)}
        # Skip well-known vocabulary URIs (the entity URIs we mint live under MKG
        # but only count as 'known' if they were explicitly typed).
        vocab_prefixes = tuple(
            str(ns) for prefix, ns in NAMESPACES.items() if prefix != "mkg"
        )
        for s, p, o in self.rdf:
            if p == RDF.type:
                continue  # rdf:type targets are class URIs, not entities
            if isinstance(o, URIRef) and str(o) not in typed_subjects:
                if str(o).startswith(vocab_prefixes):
                    continue
                conflicts.append({
                    "entity": str(o),
                    "rule": "DANGLING_REFERENCE",
                    "severity": "info",
                    "message": f"URI {o} referenced by {s} via {p} but not typed in graph",
                })

        logger.info(f"validate_consistency: {len(conflicts)} issues found")
        return conflicts

    @staticmethod
    def _strip_prefix(curie_or_uri: str) -> str:
        """Reduce 'dct:title' or 'http://.../title' to 'title'."""
        for sep in ("#", "/"):
            if sep in curie_or_uri:
                curie_or_uri = curie_or_uri.rsplit(sep, 1)[-1]
        if ":" in curie_or_uri:
            curie_or_uri = curie_or_uri.split(":", 1)[-1]
        return curie_or_uri.lower()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def export_to_turtle(self, path: str | Path | None = None) -> str:
        """Serialize to Turtle. If path provided, also write to disk."""
        data = self.rdf.serialize(format="turtle")
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(data, encoding="utf-8")
            logger.info(f"Exported KG to {path}")
        return data

    def load_from_turtle(self, path: str | Path) -> None:
        """Load triples from a Turtle file (merges into current graph)."""
        self.rdf.parse(str(path), format="turtle")
        self._rebuild_nx_from_rdf()
        logger.info(f"Loaded KG from {path}: {len(self.rdf)} triples")

    def _rebuild_nx_from_rdf(self) -> None:
        """Reconstruct networkx graph from the RDF store."""
        self.nx.clear()
        for s, p, o in self.rdf:
            if isinstance(o, URIRef):
                self.nx.add_edge(str(s), str(o), key=str(p), predicate=str(p))
            else:
                self.nx.add_node(str(s))

    def to_networkx(self) -> nx.MultiDiGraph:
        """Return the underlying networkx MultiDiGraph (for GNN/viz)."""
        return self.nx

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        return {
            "triples": len(self.rdf),
            "entities": len(self.all_entities()),
            "nx_nodes": self.nx.number_of_nodes(),
            "nx_edges": self.nx.number_of_edges(),
        }

    def __len__(self) -> int:
        return len(self.rdf)

    def __repr__(self) -> str:
        s = self.stats()
        return f"<MetadataKnowledgeGraph name={self.name!r} triples={s['triples']} entities={s['entities']}>"


# ---------------------------------------------------------------------------
# Cross-domain linking
# ---------------------------------------------------------------------------
def link_cross_domain(
    graph_a: MetadataKnowledgeGraph,
    graph_b: MetadataKnowledgeGraph,
    shared_key: str = "title",
    similarity_threshold: float = 0.85,
) -> MetadataKnowledgeGraph:
    """Merge overlapping concepts from two KGs into a new KG.

    Strategy:
        For each entity in A and B, compare `shared_key` (case-insensitive,
        whitespace-normalized). If equal, link via mkg:sameAs and copy both
        entities + properties into the merged graph.

    Args:
        graph_a, graph_b: source KGs
        shared_key: property name to match on (default: title)
        similarity_threshold: kept for future fuzzy matching; exact equality used now

    Returns:
        New MetadataKnowledgeGraph containing the union plus mkg:sameAs links.
    """
    merged = MetadataKnowledgeGraph(name=f"{graph_a.name}+{graph_b.name}")

    # Copy all triples
    for triple in graph_a.rdf:
        merged.rdf.add(triple)
    for triple in graph_b.rdf:
        merged.rdf.add(triple)
    merged._rebuild_nx_from_rdf()

    # Build index: shared_key value → list of URIs
    def _index(kg: MetadataKnowledgeGraph) -> dict[str, list[str]]:
        idx: dict[str, list[str]] = {}
        for s in set(kg.rdf.subjects()):
            for _, _p, o in kg.rdf.triples((s, None, None)):
                pred_short = kg.rdf.namespace_manager.normalizeUri(_p)
                if MetadataKnowledgeGraph._strip_prefix(pred_short) == shared_key.lower():
                    key = str(o).strip().lower()
                    idx.setdefault(key, []).append(str(s))
        return idx

    idx_a = _index(graph_a)
    idx_b = _index(graph_b)

    link_count = 0
    same_as = MKG["sameAs"]
    for key, uris_a in idx_a.items():
        uris_b = idx_b.get(key, [])
        for ua in uris_a:
            for ub in uris_b:
                if ua == ub:
                    continue
                merged.rdf.add((URIRef(ua), same_as, URIRef(ub)))
                merged.nx.add_edge(ua, ub, key=str(same_as), predicate=str(same_as))
                link_count += 1

    logger.info(
        f"link_cross_domain: linked {link_count} pairs on '{shared_key}' "
        f"(threshold={similarity_threshold})"
    )
    return merged


__all__ = [
    "MetadataKnowledgeGraph",
    "link_cross_domain",
]
